from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.rough_env_cfg import (
    UnitreeA1RoughEnvCfg,
)

from isaaclab_tasks.manager_based.locomotion.velocity.config.mydog.mydog_robot_cfg import (
    MYDOG_CFG,
)
from . import mdp_obs

@configclass
class MyDogA1BaseRoughEnvCfg(UnitreeA1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 1) 机器人本体：用你的 MyDog，而不是官方 A1
        self.scene.robot = MYDOG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 2) 如果以后切 rough，要把高度扫描绑定到你的 base_link
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

        # 3) MyDog 的可解析接触体不是 *_foot，而是 *_calf
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = (
            "rf_calf|lf_calf|rh_calf|lh_calf"
        )

        # 4) base_contact 也不是 trunk，而是 base_link
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"

        # 5) A1 里对 trunk 做的质量扰动/外力扰动，换成你的 base_link
        if self.events.add_base_mass is not None:
            self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"

        if self.events.base_external_force_torque is not None:
            self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"

     
        # 6) 语义统一：按每个关节的真实方向做 sign remap
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

        # 7) Phase A：先关扰动，先学“静稳”
        self.events.add_base_mass = None 
        self.events.base_external_force_torque = None
        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None
        
        # 8) 输入语义映射：给 policy 的 joint_pos / joint_vel 统一到同一语义空间
        self.observations.policy.joint_pos.func = mdp_obs.semantic_joint_pos_rel
        self.observations.policy.joint_vel.func = mdp_obs.semantic_joint_vel_rel

        # actions 这一项先别动，先保留原样
        # self.observations.policy.actions.func = mdp_obs.semantic_last_action