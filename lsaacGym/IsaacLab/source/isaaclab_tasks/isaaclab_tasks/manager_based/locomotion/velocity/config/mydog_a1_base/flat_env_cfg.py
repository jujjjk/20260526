from isaaclab.utils import configclass

from .rough_env_cfg import MyDogA1BaseRoughEnvCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from . import mdp_rewards


@configclass
class MyDogA1BaseFlatEnvCfg(MyDogA1BaseRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # -----------------------------
        # 平地环境
        # -----------------------------
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        self.curriculum.terrain_levels = None

        # -----------------------------
        # 先以“稳站 + 低速 walking”为主，不直接放太快
        # -----------------------------
        self.commands.base_velocity.rel_standing_envs = 0.20
        self.commands.base_velocity.ranges.lin_vel_x = [0.0, 0.25]
        self.commands.base_velocity.ranges.lin_vel_y = [0.0, 0.0]
        self.commands.base_velocity.ranges.ang_vel_z = [0.0, 0.0]

        # -----------------------------
        # 稳定性：继续保留
        # -----------------------------
        self.rewards.flat_orientation_l2.weight = -4.5
        self.rewards.ang_vel_xy_l2.weight = -0.22
        self.rewards.action_rate_l2.weight = -0.03

        # 不要太鼓励抬脚，先让四足更均衡
        self.rewards.feet_air_time.weight = 0.035

        if hasattr(self.rewards, "feet_slide"):
            self.rewards.feet_slide.weight = -0.35

        if hasattr(self.rewards, "feet_stumble"):
            self.rewards.feet_stumble.weight = -0.22

        # -----------------------------
        # 站立高度：保留，但不要单独主导
        # -----------------------------
        self.rewards.stand_height_reward = RewTerm(
            func=mdp_rewards.stand_height_reward,
            weight=0.35,
            params={
                "target_height": 0.36,
                "sigma": 0.03,
                "cmd_threshold": 0.06,
                "command_name": "base_velocity",
                "asset_name": "robot",
                "offset_body": (-0.10, 0.0, 0.0),
            },
        )

        self.rewards.stand_bad_height_penalty = RewTerm(
            func=mdp_rewards.stand_bad_height_penalty,
            weight=-0.35,
            params={
                "min_height": 0.33,
                "max_height": 0.39,
                "cmd_threshold": 0.06,
                "command_name": "base_velocity",
                "asset_name": "robot",
                "offset_body": (-0.10, 0.0, 0.0),
            },
        )

        # -----------------------------
        # 默认站姿回归：这是你真正想要的“回到稳站姿”
        # -----------------------------
        self.rewards.stand_pose_reward = RewTerm(
            func=mdp_rewards.stand_pose_reward,
            weight=1.2,
            params={
                "cmd_threshold": 0.08,
                "command_name": "base_velocity",
            },
        )

        self.rewards.stand_joint_vel_penalty = RewTerm(
            func=mdp_rewards.stand_joint_vel_penalty,
            weight=-0.05,
            params={
                "cmd_threshold": 0.08,
                "command_name": "base_velocity",
            },
        )

        # -----------------------------
        # 总接触：保留，但降到辅助项
        # -----------------------------
        self.rewards.stand_feet_contact_reward = RewTerm(
            func=mdp_rewards.stand_feet_contact_reward,
            weight=0.15,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names="rf_calf|lf_calf|rh_calf|lh_calf",
                ),
                "desired_contacts": 4.0,
                "threshold": 1.0,
                "cmd_threshold": 0.06,
                "command_name": "base_velocity",
            },
        )

        self.rewards.stand_few_contacts_penalty = RewTerm(
            func=mdp_rewards.stand_few_contacts_penalty,
            weight=-0.15,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names="rf_calf|lf_calf|rh_calf|lh_calf",
                ),
                "min_contacts": 2.5,
                "threshold": 1.0,
                "cmd_threshold": 0.06,
                "command_name": "base_velocity",
            },
        )

        # -----------------------------
        # 前后腿分开约束：这是这次最核心的新东西
        # -----------------------------
        self.rewards.front_feet_contact_reward = RewTerm(
            func=mdp_rewards.front_feet_contact_reward,
            weight=0.45,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names="rf_calf|lf_calf",
                ),
                "desired_contacts": 2.0,
                "threshold": 1.0,
                "cmd_threshold": 0.08,
                "command_name": "base_velocity",
            },
        )

        self.rewards.rear_feet_contact_reward = RewTerm(
            func=mdp_rewards.rear_feet_contact_reward,
            weight=0.75,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names="rh_calf|lh_calf",
                ),
                "desired_contacts": 2.0,
                "threshold": 1.0,
                "cmd_threshold": 0.08,
                "command_name": "base_velocity",
            },
        )

        self.rewards.rear_few_contacts_penalty = RewTerm(
            func=mdp_rewards.rear_few_contacts_penalty,
            weight=-0.60,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names="rh_calf|lh_calf",
                ),
                "min_contacts": 1.5,
                "threshold": 1.0,
                "cmd_threshold": 0.08,
                "command_name": "base_velocity",
            },
        )


@configclass
class MyDogA1BaseFlatEnvCfg_PLAY(MyDogA1BaseFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None