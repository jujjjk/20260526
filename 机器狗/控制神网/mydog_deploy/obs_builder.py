#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import math
import numpy as np

# 让当前文件能找到你前面写好的 IMU 接口
sys.path.append("/home/jetson/mydog_imu/interface")

from imu_serial_interface import ImuSerialInterface
from motor_state_interface import MotorStateHttpInterface
from semantic_mapper import JointSemanticMapper


class ObsBuilder36:
    """
    将 IMU + 12 路电机状态 + 速度命令 拼成 IsaacLab 平地策略需要的 36 维 obs。

    obs 结构：
        obs[0:3]    base_lin_vel
        obs[3:6]    base_ang_vel
        obs[6:9]    projected_gravity
        obs[9:12]   velocity_commands
        obs[12:24]  joint_pos
        obs[24:36]  joint_vel
    """

    def __init__(self, motor_base_url="http://127.0.0.1:8000", base_lin_vel_source="command", obs_dim=36, gait_phase_period=0.55):
        self.obs_dim = int(obs_dim)
        if self.obs_dim not in (36, 48, 50):
            raise ValueError("obs_dim must be 36, 48, or 50")
        self.gait_phase_period = float(gait_phase_period)
        if self.gait_phase_period <= 1e-6:
            raise ValueError("gait_phase_period must be positive")
        self.gait_phase_start_time = time.time()

        # IMU 串口接口
        self.imu = ImuSerialInterface(
            port="/dev/myimu",
            read_hz=100.0,
        )

        # 电机状态 HTTP 接口
        # 如果 FastAPI 和本程序都在 NX 上，使用 127.0.0.1 即可
        self.motor = MotorStateHttpInterface(
            base_url=motor_base_url,
            timeout=0.08,
        )

        # 真机语义 <-> 训练语义映射器
        self.mapper = JointSemanticMapper()

        # 速度命令 cmd_vx, cmd_vy, cmd_wz
        # 初期先手动设置，后面可以来自 ROS2 /cmd_vel
        self.cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # 目前没有状态估计器，所以 base_lin_vel 先置 0
        # 注意：这是临时方案，先用于跑通 36 维输入链路
        self.base_lin_vel_source = str(base_lin_vel_source).lower()
        if self.base_lin_vel_source not in ("command", "zero"):
            raise ValueError("base_lin_vel_source must be 'command' or 'zero'")
        self.base_lin_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.last_action = np.zeros(12, dtype=np.float32)

    def start(self):
        print("[ObsBuilder] starting IMU...")
        self.imu.start()

        # 等待 IMU 第一帧有效数据
        ok = self._wait_imu_ready(timeout=3.0)
        if not ok:
            raise RuntimeError("IMU not ready. Please check /dev/myimu and imu_serial_interface.py")

        print("[ObsBuilder] IMU ready.")

    def stop(self):
        self.imu.stop()

    def set_command(self, vx: float, vy: float, wz: float):
        """
        设置速度命令。
        对应 obs[9:12] = [cmd_vx, cmd_vy, cmd_wz]
        """
        self.cmd[:] = [vx, vy, wz]
        if self.base_lin_vel_source == "command":
            self.base_lin_vel[:] = [vx, vy, 0.0]
        else:
            self.base_lin_vel[:] = [0.0, 0.0, 0.0]

    def _wait_imu_ready(self, timeout=3.0) -> bool:
        start = time.time()

        while time.time() - start < timeout:
            s = self.imu.get_latest()
            if s.valid:
                return True
            time.sleep(0.01)

        return False

    def set_last_action(self, action_12):
        action = np.asarray(action_12, dtype=np.float32).reshape(-1)
        if action.shape[0] < 12:
            raise ValueError(f"last_action must have at least 12 floats, got {action.shape[0]}")
        self.last_action[:] = action[:12]

    def get_gait_phase_obs(self):
        elapsed = time.time() - self.gait_phase_start_time
        phase = (elapsed / self.gait_phase_period) % 1.0
        angle = 2.0 * math.pi * phase
        return np.array([math.sin(angle), math.cos(angle)], dtype=np.float32)

    def build_obs(self):
        """
        返回：
            obs: np.ndarray shape=(36,)
            info: 调试信息
        """

        # ==============================
        # 1. IMU 数据
        # ==============================
        base_ang_vel, projected_gravity, imu_valid = self.imu.get_policy_imu_obs()

        if not imu_valid:
            raise RuntimeError("IMU data invalid")

        # ==============================
        # 2. 电机数据
        # ==============================
        motor_snapshot = self.motor.get_latest()

        if not motor_snapshot.valid:
            raise RuntimeError("Motor state invalid")

        q_real = motor_snapshot.q_real
        dq_real = motor_snapshot.dq_real

        # ==============================
        # 3. 真机电机语义 -> 策略训练语义
        # ==============================
        q_policy, dq_policy = self.mapper.real_to_policy_q_dq(
            q_real=q_real,
            dq_real=dq_real,
        )

        # ==============================
        # 4. 拼 36 维 obs
        # ==============================
        obs = np.zeros(self.obs_dim, dtype=np.float32)

        obs[0:3] = self.base_lin_vel
        obs[3:6] = base_ang_vel
        obs[6:9] = projected_gravity
        obs[9:12] = self.cmd
        obs[12:24] = q_policy
        obs[24:36] = dq_policy
        if self.obs_dim >= 48:
            obs[36:48] = self.last_action
        if self.obs_dim >= 50:
            obs[48:50] = self.get_gait_phase_obs()

        info = {
            "imu_valid": imu_valid,
            "motor_valid": motor_snapshot.valid,

            "base_lin_vel": self.base_lin_vel.copy(),
            "base_ang_vel": np.asarray(base_ang_vel, dtype=np.float32).copy(),
            "projected_gravity": np.asarray(projected_gravity, dtype=np.float32).copy(),
            "cmd": self.cmd.copy(),

            "q_real": q_real.copy(),
            "dq_real": dq_real.copy(),
            "q_policy": q_policy.copy(),
            "dq_policy": dq_policy.copy(),
            "last_action": self.last_action.copy(),
            "gait_phase": self.get_gait_phase_obs(),

            "online": motor_snapshot.online.copy(),
            "age_ms": motor_snapshot.age_ms.copy(),
        }

        return obs, info

    def print_debug(self, obs, info):
        print("=" * 90)
        print("obs.shape:", obs.shape)

        print("\n[0:3] base_lin_vel:")
        print(obs[0:3])

        print("\n[3:6] base_ang_vel:")
        print(obs[3:6])

        print("\n[6:9] projected_gravity:")
        print(obs[6:9])

        print("\n[9:12] velocity_commands:")
        print(obs[9:12])

        print("\n[12:24] q_policy / joint_pos:")
        self._print_policy_array(info["q_policy"])

        print("\n[24:36] dq_policy / joint_vel:")
        self._print_policy_array(info["dq_policy"])

        if obs.shape[0] >= 48:
            print("\n[36:48] last_action:")
            self._print_policy_array(info["last_action"])

        if obs.shape[0] >= 50:
            print("\n[48:50] gait_phase [sin, cos]:")
            print(info["gait_phase"])

        print("\nq_real, motor order = FR, FL, RL, RR:")
        self._print_real_array(info["q_real"])

        print("\ndq_real, motor order = FR, FL, RL, RR:")
        self._print_real_array(info["dq_real"])

        print("\nage_ms:")
        print(info["age_ms"])

        print("\nonline:")
        print(info["online"])

    def _print_policy_array(self, arr):
        names = self.mapper.get_policy_joint_names()
        for i, name in enumerate(names):
            print(f"  policy[{i:02d}] {name:16s}: {arr[i]:+.5f}")

    def _print_real_array(self, arr):
        ids = self.mapper.get_real_motor_ids()
        real_names = self.mapper.real_joint_names
        for i, mid in enumerate(ids):
            print(f"  real[{i:02d}] motor_id=0x{mid:02X} {real_names[i]:16s}: {arr[i]:+.5f}")


def main():
    # 如果 app.py/FastAPI 就在本机 NX 上运行，用 127.0.0.1
    # 如果你要从另一台机器访问 NX，才用 http://172.19.xx.xx:8000
    builder = ObsBuilder36(motor_base_url="http://127.0.0.1:8000")

    builder.start()

    # 这里只是测试拼 obs，不会发电机命令
    # 先给一个小的前进命令，后面可以改成 0
    builder.set_command(0.10, 0.0, 0.0)

    try:
        while True:
            obs, info = builder.build_obs()
            builder.print_debug(obs, info)
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[ObsBuilder] stopped by user.")

    finally:
        builder.stop()


if __name__ == "__main__":
    main()
