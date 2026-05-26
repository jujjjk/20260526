#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import numpy as np


class DeployJointCPG:
    """Deployment-side joint-space CPG matching FanfanRlCpg training."""

    LEG_ORDER = ("FR", "FL", "RR", "RL")

    PHASE_OFFSETS = {
        "trot": {"FR": 0.0, "RL": 0.0, "FL": 0.5, "RR": 0.5},
        "pace": {"FR": 0.0, "RR": 0.0, "FL": 0.5, "RL": 0.5},
        "bound": {"FR": 0.0, "FL": 0.0, "RR": 0.5, "RL": 0.5},
        "walk": {"FR": 0.0, "RL": 0.25, "FL": 0.5, "RR": 0.75},
    }

    def __init__(
        self,
        default_joint_angle,
        lower_limit,
        upper_limit,
        policy_hz,
        gait="trot",
        freq_min=0.8,
        freq_max=1.8,
        k_freq=3.0,
        standing_cmd_threshold=0.03,
        duty_factor=0.60,
        hip_amp=0.0,
        thigh_amp=0.18,
        calf_lift_amp=0.60,
        stance_calf_amp=0.08,
        stride_sign=-1.0,
        residual_limit_hip=0.04,
        residual_limit_thigh=0.06,
        residual_limit_calf=0.06,
    ):
        self.default_joint_angle = np.asarray(default_joint_angle, dtype=np.float32).reshape(12)
        self.lower_limit = np.asarray(lower_limit, dtype=np.float32).reshape(12)
        self.upper_limit = np.asarray(upper_limit, dtype=np.float32).reshape(12)
        self.policy_hz = max(float(policy_hz), 1.0e-3)
        self.gait = str(gait).strip().lower()
        if self.gait not in self.PHASE_OFFSETS:
            self.gait = "trot"

        self.freq_min = float(freq_min)
        self.freq_max = float(freq_max)
        self.k_freq = float(k_freq)
        self.standing_cmd_threshold = float(standing_cmd_threshold)
        self.duty_factor = float(duty_factor)
        self.hip_amp = float(hip_amp)
        self.thigh_amp = float(thigh_amp)
        self.calf_lift_amp = float(calf_lift_amp)
        self.stance_calf_amp = float(stance_calf_amp)
        self.stride_sign = float(stride_sign)
        self.phase = 0.0
        self.last_frequency = 0.0
        self.last_leg_phase = np.zeros(4, dtype=np.float32)

        self.residual_limits = np.asarray(
            [
                residual_limit_hip,
                residual_limit_thigh,
                residual_limit_calf,
                residual_limit_hip,
                residual_limit_thigh,
                residual_limit_calf,
                residual_limit_hip,
                residual_limit_thigh,
                residual_limit_calf,
                residual_limit_hip,
                residual_limit_thigh,
                residual_limit_calf,
            ],
            dtype=np.float32,
        )

    def update(self, command, dt=None):
        command = np.asarray(command, dtype=np.float32).reshape(3)
        dt = (1.0 / self.policy_hz) if dt is None else max(float(dt), 1.0e-4)
        freq = self.compute_frequency(command)
        self.phase = (self.phase + freq * dt) % 1.0

        offsets = self.PHASE_OFFSETS.get(self.gait, self.PHASE_OFFSETS["trot"])
        q = self.default_joint_angle.copy()
        duty = min(max(float(self.duty_factor), 0.50), 0.90)
        swing_fraction = max(1.0 - duty, 0.05)
        moving = 1.0 if freq > 0.0 else 0.0

        leg_phases = []
        for leg_i, leg in enumerate(self.LEG_ORDER):
            p = (self.phase + float(offsets[leg])) % 1.0
            leg_phases.append(p)
            if p < swing_fraction:
                s = p / swing_fraction
                swing = math.sin(math.pi * s)
                stance = 0.0
                stride = -1.0 + 2.0 * s
            else:
                s = (p - swing_fraction) / max(1.0 - swing_fraction, 1.0e-5)
                swing = 0.0
                stance = math.sin(math.pi * s)
                stride = 1.0 - 2.0 * s

            base = leg_i * 3
            q[base + 0] += moving * self.hip_amp * stride
            q[base + 1] += moving * self.stride_sign * self.thigh_amp * stride
            q[base + 2] += moving * (
                -self.calf_lift_amp * swing
                + self.stance_calf_amp * stance
            )

        self.last_leg_phase[:] = np.asarray(leg_phases, dtype=np.float32)
        q = np.clip(q, self.lower_limit, self.upper_limit)
        return q.astype(np.float32)

    def compute_frequency(self, command):
        cmd_x = abs(float(command[0]))
        if cmd_x < self.standing_cmd_threshold:
            self.last_frequency = 0.0
            return 0.0
        freq = self.freq_min + self.k_freq * cmd_x
        freq = min(max(freq, self.freq_min), self.freq_max)
        self.last_frequency = float(freq)
        return float(freq)

    def info(self):
        return {
            "frequency": float(self.last_frequency),
            "phase": float(self.phase),
            "leg_phase": self.last_leg_phase.astype(np.float32).copy(),
            "residual_limits": self.residual_limits.astype(np.float32).copy(),
        }
