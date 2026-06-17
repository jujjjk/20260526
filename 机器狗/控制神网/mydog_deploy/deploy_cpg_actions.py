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
    hip_amp: float = 0.025
    thigh_amp: float = 0.18
    calf_lift_amp: float = 0.60
    stance_calf_amp: float = 0.08
    stride_sign: float = -1.0
    duty_factor: float = 0.60
    enable_hip_balance: bool = True
    hip_stance_widen_amp: float = 0.020
    hip_swing_relax_amp: float = 0.008
    hip_balance_signs: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0)
    hip_balance_use_stance_mask: bool = True
    hip_balance_smooth_shape: str = "sin"
    hip_balance_max_abs: float = 0.06
    residual_limit_hip: float = 0.03
    residual_limit_thigh: float = 0.06
    residual_limit_calf: float = 0.06
    enable_phase_aware_hip_gate: bool = True
    hip_gate_stance_min_outward: float = 0.008
    hip_gate_swing_max_outward: float = 0.035
    hip_gate_side_signs: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0)
    lowpass_alpha: float = 0.35
    max_delta_per_step: float = 0.03


class DeployCPGAction:
    def __init__(self, default_joint_angle: np.ndarray, cfg: DeployCPGCfg | None = None):
        self.cfg = cfg or DeployCPGCfg()
        self.default = np.asarray(default_joint_angle, dtype=np.float32).reshape(12)
        self.phase = 0.0
        self.last_q_cmd = self.default.copy()
        self.hip_balance_signs = np.asarray(self.cfg.hip_balance_signs, dtype=np.float32).reshape(4)
        self.hip_gate_side_signs = np.asarray(self.cfg.hip_gate_side_signs, dtype=np.float32).reshape(4)
        self.residual_limits = np.asarray(
            [self.cfg.residual_limit_hip, self.cfg.residual_limit_thigh, self.cfg.residual_limit_calf] * 4,
            dtype=np.float32,
        )
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
        hip_stride_delta = np.zeros(4, dtype=np.float32)
        hip_balance_delta = np.zeros(4, dtype=np.float32)
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
            hip_stride = self.cfg.hip_amp * stride
            hip_balance = 0.0
            if self.cfg.enable_hip_balance:
                if str(self.cfg.hip_balance_smooth_shape).lower() == "sin":
                    stance_weight = stance
                    swing_weight = swing
                else:
                    stance_weight = 0.0 if phase01 < swing_fraction else 1.0
                    swing_weight = 1.0 if phase01 < swing_fraction else 0.0
                if not self.cfg.hip_balance_use_stance_mask:
                    stance_weight = 1.0
                hip_sign = float(self.hip_balance_signs[leg_i])
                hip_balance = (
                    hip_sign * self.cfg.hip_stance_widen_amp * stance_weight
                    - hip_sign * self.cfg.hip_swing_relax_amp * swing_weight
                )
                hip_balance = float(np.clip(hip_balance, -self.cfg.hip_balance_max_abs, self.cfg.hip_balance_max_abs))
            hip_stride_delta[leg_i] = moving * hip_stride
            hip_balance_delta[leg_i] = moving * hip_balance
            q[base + 0] += moving * (hip_stride + hip_balance)
            q[base + 1] += moving * self.cfg.stride_sign * self.cfg.thigh_amp * stride
            q[base + 2] += moving * (-self.cfg.calf_lift_amp * swing + self.cfg.stance_calf_amp * stance)
        return q.astype(np.float32), np.asarray(phases, dtype=np.float32), freq, hip_stride_delta, hip_balance_delta

    def target_policy(self, action_raw: np.ndarray, cmd: np.ndarray, action_mode: str, action_scale: np.ndarray) -> np.ndarray:
        action_raw = np.asarray(action_raw, dtype=np.float32).reshape(12)
        action_clip = np.clip(action_raw, -1.0, 1.0)
        q_cpg, phases, freq, hip_stride_delta, hip_balance_delta = self.q_cpg(np.asarray(cmd, dtype=np.float32).reshape(3))
        if action_mode == "pure_rl":
            q_raw = self.default + np.asarray(action_scale, dtype=np.float32).reshape(12) * action_clip
            delta = np.zeros(12, dtype=np.float32)
        elif action_mode == "cpg_only":
            q_raw = q_cpg
            delta = np.zeros(12, dtype=np.float32)
        elif action_mode == "cpg_residual":
            delta = np.clip(action_clip * self.residual_limits, -self.residual_limits, self.residual_limits)
            q_raw = q_cpg + delta
            if self.cfg.enable_phase_aware_hip_gate and freq > 1.0e-6:
                hip_ids = np.asarray([0, 3, 6, 9], dtype=np.int64)
                phase01 = np.remainder(phases / (2.0 * math.pi), 1.0)
                swing_fraction = max(0.05, 1.0 - float(self.cfg.duty_factor))
                swing = phase01 < swing_fraction
                stance = ~swing
                outward_before = self.hip_gate_side_signs * (q_raw[hip_ids] - self.default[hip_ids])
                outward_after = outward_before.copy()
                outward_after[stance & (outward_after < self.cfg.hip_gate_stance_min_outward)] = (
                    self.cfg.hip_gate_stance_min_outward
                )
                outward_after[swing & (outward_after > self.cfg.hip_gate_swing_max_outward)] = (
                    self.cfg.hip_gate_swing_max_outward
                )
                q_raw[hip_ids] = self.default[hip_ids] + self.hip_gate_side_signs * outward_after
                delta[hip_ids] = q_raw[hip_ids] - q_cpg[hip_ids]
                hip_gate_clamp_count = int(np.count_nonzero(np.abs(outward_after - outward_before) > 1.0e-7))
            else:
                outward_before = np.zeros(4, dtype=np.float32)
                outward_after = np.zeros(4, dtype=np.float32)
                hip_gate_clamp_count = 0
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
            "hip_stride_delta": hip_stride_delta,
            "hip_balance_delta": hip_balance_delta,
            "hip_gate_clamp_count": hip_gate_clamp_count if action_mode == "cpg_residual" else 0,
            "hip_outward_before_gate": outward_before.astype(np.float32)
            if action_mode == "cpg_residual"
            else np.zeros(4, dtype=np.float32),
            "hip_outward_after_gate": outward_after.astype(np.float32)
            if action_mode == "cpg_residual"
            else np.zeros(4, dtype=np.float32),
            "raw_gt_1_count": int(np.sum(np.abs(action_raw) > 1.0)),
        }
        return self.last_q_cmd.copy()
