#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import numpy as np
import onnxruntime as ort

from deploy_cpg_actions import DeployCPGAction
from obs_builder import ObsBuilder36


ACTION_SCALE = np.array([0.12, 0.22, 0.30, 0.12, 0.22, 0.30, 0.12, 0.26, 0.42, 0.12, 0.26, 0.42], dtype=np.float32)


class OnnxPolicyRunner:
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.obs_dim = int(input_shape[1])

        print("[ONNX] input :", self.input_name, self.session.get_inputs()[0].shape)
        print("[ONNX] output:", self.output_name, self.session.get_outputs()[0].shape)

    def infer(self, obs_in: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs_in, dtype=np.float32).reshape(1, self.obs_dim)

        out = self.session.run(
            [self.output_name],
            {self.input_name: obs},
        )[0]

        action = np.asarray(out, dtype=np.float32).reshape(-1)

        if action.shape[0] < 12:
            raise RuntimeError(f"ONNX output size < 12, got {action.shape}")

        action = action[:12]
        return action


def print_action_and_target(builder: ObsBuilder36, action_raw: np.ndarray, cpg_action: DeployCPGAction, action_mode: str):
    mapper = builder.mapper

    # 先打印原始 action
    print("\n[Policy raw action]")
    for i, name in enumerate(mapper.get_policy_joint_names()):
        print(f"  action[{i:02d}] {name:16s}: {action_raw[i]:+.5f}")

    # 为了安全，先限幅到 [-1, 1]
    action_safe = np.clip(action_raw, -1.0, 1.0)

    print("\n[Policy clipped action]")
    for i, name in enumerate(mapper.get_policy_joint_names()):
        print(f"  action_clip[{i:02d}] {name:16s}: {action_safe[i]:+.5f}")

    cmd = getattr(builder, "cmd", np.zeros(3, dtype=np.float32))
    target_policy_abs = cpg_action.target_policy(
        action_raw=action_raw,
        cmd=cmd,
        action_mode=action_mode,
        action_scale=ACTION_SCALE,
    )
    target_real = mapper.policy_target_to_real_target(target_policy_abs, clamp=True)
    dbg = cpg_action.last_debug
    print(f"\n[Action mode] {action_mode}")
    print("cpg_frequency:", dbg.get("cpg_frequency"))
    print("cpg_phase:", dbg.get("cpg_phase"))
    print("q_cpg:", dbg.get("q_cpg"))
    print("delta_q_rl:", dbg.get("delta_q_rl"))
    print("q_raw:", dbg.get("q_raw"))
    print("q_cmd:", dbg.get("q_cmd"))

    print("\n[Target real motor position, motor order = FR, FL, RL, RR]")
    motor_ids = mapper.get_real_motor_ids()
    real_names = mapper.real_joint_names

    for i, mid in enumerate(motor_ids):
        print(
            f"  motor_id=0x{mid:02X} {real_names[i]:16s} "
            f"target={target_real[i]:+.5f} rad"
        )

    return target_real


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="path to policy.onnx")
    parser.add_argument("--motor-url", default="http://127.0.0.1:8000")
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--action-mode", choices=["pure_rl", "cpg_only", "cpg_residual"], default="pure_rl")
    args = parser.parse_args()

    policy = OnnxPolicyRunner(args.onnx)
    builder = ObsBuilder36(motor_base_url=args.motor_url, obs_dim=policy.obs_dim)
    builder.start()
    builder.set_command(args.vx, args.vy, args.wz)
    cpg_action = DeployCPGAction(builder.mapper.default_joint_angle)

    period = 1.0 / args.hz

    try:
        while True:
            obs, info = builder.build_obs()

            # 安全检查：现在只是打印，不发电机，但也先检查
            max_age = float(np.max(info["age_ms"]))
            if max_age > 100.0:
                print(f"[WARN] motor feedback age too large: {max_age:.1f} ms")

            print("=" * 100)
            print("obs.shape:", obs.shape)
            print("base_ang_vel:", obs[3:6])
            print("projected_gravity:", obs[6:9])
            print("cmd:", obs[9:12])
            print("q_policy:", obs[12:24])
            print("dq_policy:", obs[24:36])
            print("max_age_ms:", max_age)

            action = policy.infer(obs)
            _ = print_action_and_target(builder, action, cpg_action, args.action_mode)
            builder.set_last_action(np.clip(action, -1.0, 1.0))

            time.sleep(period)

    except KeyboardInterrupt:
        print("\n[ONNX TEST] stopped by user.")
    finally:
        builder.stop()


if __name__ == "__main__":
    main()
