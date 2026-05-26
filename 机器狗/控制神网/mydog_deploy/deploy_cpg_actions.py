#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DeployCPGCfg:
    enable: bool = False
    mode: str = "cpg_residual"
    gait: str = "trot"
    dt: float = 0.02
    leg_order: tuple[str, str, str, str] = ("FR", "FL", "RR", "RL")
    phase_offsets: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "trot": {"FR": 0.0, "RL": 0.0, "FL": 0.5, "RR": 0.5},
            "pace": {"FR": 0.0, "RR": 0.0, "FL": 0.5, "RL": 0.5},
            "bound": {"FR": 0.0, "FL": 0.0, "RR": 0.5, "RL": 0.5},
            "walk": {"FR": 0.0, "RL": 0.25, "FL": 0.5, "RR": 0.75},
        }
    )
    freq_min: float = 0.8
    freq_max: float = 1.8
    k_freq: float = 3.0
    standing_cmd_threshold: float = 0.03
    hip_amp: float = 0.0
    thigh_amp: float = 0.18
    calf_lift_amp: float = 0.60
    stance_calf_amp: float = 0.08
    stride_sign: float = -1.0
    duty_factor: float = 0.60
    residual_limit_hip: float = 0.04
    residual_limit_thigh: float = 0.06
    residual_limit_calf: float = 0.06
    lowpass_alpha: float = 0.35
    max_delta_per_step: float = 0.03


class DeployCPGAction:
    def __init__(self, default_joint_angle: np.ndarray, cfg: DeployCPGCfg | None = None):
        self.cfg = cfg or DeployCPGCfg()
        self.default = np.asarray(default_joint_angle, dtype=np.float32).reshape(12)
        self.phase = 0.0
        self.last_q_cmd = self.default.copy()
        self.residual_limits = np.asarray([0.04, 0.06, 0.06] * 4, dtype=np.float32)
        self.last_debug = {}

    def reset(self):
        self.phase = 0.0
        self.last_q_cmd = self.default.copy()

    def _frequency(self, cmd_x: float) -> float:
        if abs(cmd_x) < self.cfg.standing_cmd_threshold:
            return 0.0
        return float(np.clip(self.cfg.freq_min + self.cfg.k_freq * abs(cmd_x), self.cfg.freq_min, self.cfg.freq_max))

    def q_cpg(self, cmd: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        cmd_x = float(cmd[0])
        freq = self._frequency(cmd_x)
        self.phase = (self.phase + 2.0 * math.pi * freq * self.cfg.dt) % (2.0 * math.pi)
        offsets = self.cfg.phase_offsets.get(self.cfg.gait, self.cfg.phase_offsets["trot"])
        q = self.default.copy()
        phases = []
        moving = 0.0 if freq <= 0 else 1.0
        for leg_i, leg in enumerate(self.cfg.leg_order):
            p = (self.phase + 2.0 * math.pi * offsets[leg]) % (2.0 * math.pi)
            phases.append(p)
            phase01 = p / (2.0 * math.pi)
            swing_fraction = max(0.05, 1.0 - float(self.cfg.duty_factor))
            if phase01 < swing_fraction:
                s = phase01 / swing_fraction
                swing = math.sin(math.pi * s)
                stance = 0.0
                stride = -1.0 + 2.0 * s
            else:
                s = (phase01 - swing_fraction) / max(1.0 - swing_fraction, 1e-3)
                swing = 0.0
                stance = math.sin(math.pi * s)
                stride = 1.0 - 2.0 * s
            base = leg_i * 3
            q[base + 0] += moving * self.cfg.hip_amp * stride
            q[base + 1] += moving * self.cfg.stride_sign * self.cfg.thigh_amp * stride
            q[base + 2] += moving * (-self.cfg.calf_lift_amp * swing + self.cfg.stance_calf_amp * stance)
        return q.astype(np.float32), np.asarray(phases, dtype=np.float32), freq

    def target_policy(self, action_raw: np.ndarray, cmd: np.ndarray, action_mode: str, action_scale: np.ndarray) -> np.ndarray:
        action_raw = np.asarray(action_raw, dtype=np.float32).reshape(12)
        action_clip = np.clip(action_raw, -1.0, 1.0)
        q_cpg, phases, freq = self.q_cpg(np.asarray(cmd, dtype=np.float32).reshape(3))
        if action_mode == "pure_rl":
            q_raw = self.default + np.asarray(action_scale, dtype=np.float32).reshape(12) * action_clip
            delta = np.zeros(12, dtype=np.float32)
        elif action_mode == "cpg_only":
            q_raw = q_cpg
            delta = np.zeros(12, dtype=np.float32)
        elif action_mode == "cpg_residual":
            delta = np.clip(action_clip * self.residual_limits, -self.residual_limits, self.residual_limits)
            q_raw = q_cpg + delta
        else:
            raise ValueError(f"unsupported action_mode: {action_mode}")
        step = np.clip(q_raw - self.last_q_cmd, -self.cfg.max_delta_per_step, self.cfg.max_delta_per_step)
        q_limited = self.last_q_cmd + step
        q_cmd = self.cfg.lowpass_alpha * q_limited + (1.0 - self.cfg.lowpass_alpha) * self.last_q_cmd
        self.last_q_cmd = q_cmd.astype(np.float32)
        self.last_debug = {
            "q_cpg": q_cpg,
            "delta_q_rl": delta.astype(np.float32),
            "q_raw": q_raw.astype(np.float32),
            "q_cmd": self.last_q_cmd.copy(),
            "cpg_phase": phases,
            "cpg_frequency": freq,
            "raw_gt_1_count": int(np.sum(np.abs(action_raw) > 1.0)),
        }
        return self.last_q_cmd.copy()
