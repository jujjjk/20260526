#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import numpy as np
import onnxruntime as ort

from deploy_cpg_actions import DeployCPGAction
from obs_builder import ObsBuilder36


ACTION_SCALE = np.array([0.12, 0.22, 0.30, 0.12, 0.22, 0.30, 0.12, 0.26, 0.42, 0.12, 0.26, 0.42], dtype=np.float32)


class OnnxPolicyRunner:
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.obs_dim = int(self.session.get_inputs()[0].shape[1])

    def infer(self, obs_in: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs_in, dtype=np.float32).reshape(1, self.obs_dim)
        out = self.session.run([self.output_name], {self.input_name: obs})[0]
        return np.asarray(out, dtype=np.float32).reshape(-1)[:12]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--motor-url", default="http://127.0.0.1:8000")
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--action-mode", choices=["pure_rl", "cpg_only", "cpg_residual"], default="pure_rl")
    args = parser.parse_args()

    policy = OnnxPolicyRunner(args.onnx)
    builder = ObsBuilder36(motor_base_url=args.motor_url)
    builder.start()
    builder.set_command(args.vx, args.vy, args.wz)
    cpg_action = DeployCPGAction(builder.mapper.default_joint_angle)
    period = 1.0 / args.hz
    try:
        while True:
            obs, info = builder.build_obs()
            action = policy.infer(obs)
            target_policy_abs = cpg_action.target_policy(action, builder.cmd, args.action_mode, ACTION_SCALE)
            target_real = builder.mapper.policy_target_to_real_target(target_policy_abs, clamp=True)
            print("=" * 100)
            print("action_mode:", args.action_mode)
            print("raw_action:", action)
            print("q_cpg:", cpg_action.last_debug.get("q_cpg"))
            print("delta_q_rl:", cpg_action.last_debug.get("delta_q_rl"))
            print("q_cmd_policy:", target_policy_abs)
            print("target_real:", target_real)
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        builder.stop()


if __name__ == "__main__":
    main()
