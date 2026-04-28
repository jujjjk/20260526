from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.rough_env_cfg import (
    UnitreeA1RoughEnvCfg,
)

from isaaclab_tasks.manager_based.locomotion.velocity.config.mydog_clean_base.mydog_robot_cfg import (
    MYDOG_CFG,
)

from . import mdp_obs


@configclass
class MyDogCleanRoughEnvCfg(UnitreeA1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 1) 用 MyDog 本体
        self.scene.robot = MYDOG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 2) rough 时若启用 height scanner，就绑到 base_link
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

        # 3) MyDog 的接触体用 calf
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = (
            "rf_calf|lf_calf|rh_calf|lh_calf"
        )

        # 4) base_contact 用 base_link
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"

        # 5) trunk 随机化对象改成 base_link
        if self.events.add_base_mass is not None:
            self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"

        if self.events.base_external_force_torque is not None:
            self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"

        # 6) 输出语义映射：按关节方向做 sign remap
        self.actions.joint_pos.scale = {
            "rf_hip":   0.18,
            "rf_thigh": -0.18,
            "rf_calf":  -0.18,

            "lf_hip":   0.18,
            "lf_thigh": 0.18,
            "lf_calf":  0.18,

            "rh_hip":   -0.18,
            "rh_thigh": -0.18,
            "rh_calf":  -0.18,

            "lh_hip":   0.18,
            "lh_thigh": 0.18,
            "lh_calf":  0.18,
        }

        # 7) 输入语义映射
        self.observations.policy.joint_pos.func = mdp_obs.semantic_joint_pos_rel
        self.observations.policy.joint_vel.func = mdp_obs.semantic_joint_vel_rel

        # 8) 先去掉 last_action 观测
        self.observations.policy.actions = None

        # 9) 前期先关扰动
        self.events.add_base_mass = None
        self.events.base_external_force_torque = None
        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None