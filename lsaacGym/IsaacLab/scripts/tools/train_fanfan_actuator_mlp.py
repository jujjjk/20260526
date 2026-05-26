#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train a Fanfan actuator MLP from open-loop hardware CSV logs.

The exported TorchScript model is compatible with IsaacLab's ActuatorNetMLPCfg
when using:
    input_idx=[0, 1, 2], input_order="pos_vel",
    pos_scale=1.0, vel_scale=1.0, torque_scale=1.0

CSV rows are expected to come from mydog_openloop_gait_node.py.  The script
converts real motor signs into policy/URDF joint signs before building samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


JOINT_SIGNS = {
    "FR_hip_joint": -1.0,
    "FR_thigh_joint": 1.0,
    "FR_calf_joint": 1.0,
    "FL_hip_joint": -1.0,
    "FL_thigh_joint": -1.0,
    "FL_calf_joint": -1.0,
    "RR_hip_joint": 1.0,
    "RR_thigh_joint": 1.0,
    "RR_calf_joint": 1.0,
    "RL_hip_joint": 1.0,
    "RL_thigh_joint": -1.0,
    "RL_calf_joint": -1.0,
}


@dataclass
class DatasetStats:
    num_files: int
    num_samples: int
    input_dim: int
    input_idx: list[int]
    input_order: str
    csv_hz_mean: float
    q_error_abs_mean: float
    q_error_abs_p95: float
    torque_abs_mean: float
    torque_abs_p95: float
    modes: list[str]
    joint_names: list[str]


class NormalizedActuatorMLP(nn.Module):
    def __init__(self, input_mean: torch.Tensor, input_std: torch.Tensor, torque_mean: torch.Tensor, torque_std: torch.Tensor):
        super().__init__()
        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_std", input_std)
        self.register_buffer("torque_mean", torque_mean)
        self.register_buffer("torque_std", torque_std)
        self.net = nn.Sequential(
            nn.Linear(input_mean.numel(), 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.input_mean) / self.input_std
        y = self.net(x)
        return y * self.torque_std + self.torque_mean


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(p * len(ordered)) - 1))
    return ordered[index]


def parse_csv_files(data_dir: Path, glob_pattern: str) -> tuple[list[dict], list[float], list[str]]:
    files = sorted(data_dir.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files matched {data_dir / glob_pattern}")

    rows: list[dict] = []
    frame_dt: list[float] = []
    modes: set[str] = set()
    for path in files:
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            file_rows = list(reader)
        if not file_rows:
            continue

        frame_times = sorted({float(r["time"]) for r in file_rows if r.get("time")})
        frame_dt.extend(b - a for a, b in zip(frame_times, frame_times[1:]) if b > a)
        modes.update(r.get("motion_mode", "") for r in file_rows if r.get("motion_mode", ""))
        for r in file_rows:
            r["_source_file"] = path.name
        rows.extend(file_rows)

    return rows, frame_dt, sorted(modes)


def build_samples(rows: list[dict], input_idx: list[int], min_elapsed: float) -> tuple[torch.Tensor, torch.Tensor, DatasetStats]:
    max_hist = max(input_idx)
    sequences: dict[tuple[str, str], list[tuple[float, list[float], float]]] = {}
    qerr_abs: list[float] = []
    torque_abs: list[float] = []
    joint_names: set[str] = set()

    for r in rows:
        try:
            elapsed = float(r["elapsed"])
            if elapsed < min_elapsed:
                continue
            joint_name = r["joint_name"]
            sign = JOINT_SIGNS[joint_name]
            q_target_policy = float(r["q_target_policy_abs"])
            q_current_policy = sign * float(r["q_current_real"])
            dq_policy = sign * float(r["dq_current_real"])
            torque_policy = sign * float(r["torque_measured"])
            online = int(r.get("online", "1"))
            error_code = int(r.get("error_code", "0"))
            sent = int(r.get("sent", "1"))
        except (KeyError, TypeError, ValueError):
            continue
        if not online or error_code != 0 or not sent:
            continue
        if not all(math.isfinite(v) for v in (q_target_policy, q_current_policy, dq_policy, torque_policy)):
            continue

        qerr = q_target_policy - q_current_policy
        qerr_abs.append(abs(qerr))
        torque_abs.append(abs(torque_policy))
        joint_names.add(joint_name)
        key = (r.get("_source_file", ""), joint_name)
        sequences.setdefault(key, []).append((float(r["time"]), [qerr, dq_policy], torque_policy))

    x_samples: list[list[float]] = []
    y_samples: list[float] = []
    for sequence in sequences.values():
        sequence.sort(key=lambda item: item[0])
        for i in range(max_hist, len(sequence)):
            pos_hist = [sequence[i - h][1][0] for h in input_idx]
            vel_hist = [sequence[i - h][1][1] for h in input_idx]
            x_samples.append(pos_hist + vel_hist)
            y_samples.append(sequence[i][2])

    if not x_samples:
        raise RuntimeError("No valid actuator samples were built from CSV files.")

    stats = DatasetStats(
        num_files=len({r.get("_source_file", "") for r in rows}),
        num_samples=len(x_samples),
        input_dim=len(x_samples[0]),
        input_idx=list(input_idx),
        input_order="pos_vel",
        csv_hz_mean=0.0,
        q_error_abs_mean=sum(qerr_abs) / max(1, len(qerr_abs)),
        q_error_abs_p95=percentile(qerr_abs, 0.95),
        torque_abs_mean=sum(torque_abs) / max(1, len(torque_abs)),
        torque_abs_p95=percentile(torque_abs, 0.95),
        modes=[],
        joint_names=sorted(joint_names),
    )
    return torch.tensor(x_samples, dtype=torch.float32), torch.tensor(y_samples, dtype=torch.float32).unsqueeze(1), stats


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows, frame_dt, modes = parse_csv_files(args.data_dir, args.glob)
    x, y, stats = build_samples(rows, args.input_idx, args.min_elapsed)
    stats.csv_hz_mean = 1.0 / (sum(frame_dt) / len(frame_dt)) if frame_dt else 0.0
    stats.modes = modes

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    n = x.shape[0]
    perm = torch.randperm(n)
    val_count = max(1, int(n * args.val_ratio))
    val_idx = perm[:val_count]
    train_idx = perm[val_count:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    input_mean = x_train.mean(dim=0)
    input_std = x_train.std(dim=0).clamp_min(1.0e-6)
    torque_mean = y_train.mean(dim=0)
    torque_std = y_train.std(dim=0).clamp_min(1.0e-6)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = NormalizedActuatorMLP(input_mean, input_std, torque_mean, torque_std).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=args.huber_beta)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_x = x_val.to(device)
    val_y = y_val.to(device)

    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total += loss.item() * batch_x.shape[0]
            count += batch_x.shape[0]

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = loss_fn(val_pred, val_y).item()
            val_mae = torch.mean(torch.abs(val_pred - val_y)).item()
            val_rmse = torch.sqrt(torch.mean(torch.square(val_pred - val_y))).item()
        train_loss = total / max(1, count)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:04d} train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} val_mae={val_mae:.4f}Nm val_rmse={val_rmse:.4f}Nm"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    scripted = torch.jit.script(model.cpu())
    model_path = output_dir / "fanfan_actuator_mlp.pt"
    scripted.save(str(model_path))

    metadata = {
        "stats": asdict(stats),
        "model_file": str(model_path),
        "actuator_net_cfg": {
            "network_file": str(model_path),
            "pos_scale": 1.0,
            "vel_scale": 1.0,
            "torque_scale": 1.0,
            "input_order": "pos_vel",
            "input_idx": args.input_idx,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "val_ratio": args.val_ratio,
            "best_val_smooth_l1": best_val,
        },
    }
    metadata_path = output_dir / "fanfan_actuator_mlp_meta.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nSaved TorchScript model: {model_path}")
    print(f"Saved metadata:         {metadata_path}")
    print("\nUse with ActuatorNetMLPCfg(pos_scale=1.0, vel_scale=1.0, torque_scale=1.0, input_order='pos_vel').")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--glob", type=str, default="*.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("logs/fanfan_actuator_mlp"))
    parser.add_argument("--input-idx", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--min-elapsed", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--huber-beta", type=float, default=0.05)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
