from isaaclab.utils import configclass

from .rough_env_cfg import MyDogCleanRoughEnvCfg


@configclass
class MyDogCleanFlatEnvCfg(MyDogCleanRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 平地
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # flat 不用 height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # flat 不用 terrain curriculum
        self.curriculum.terrain_levels = None

        # 接近官方 flat 风格，但对 MyDog 保守一点
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25

        # 先只练直行
        self.commands.base_velocity.ranges.lin_vel_x = [0.0, 0.25]
        self.commands.base_velocity.ranges.lin_vel_y = [0.0, 0.0]
        self.commands.base_velocity.ranges.ang_vel_z = [0.0, 0.0]

        # 少量 standing env
        self.commands.base_velocity.rel_standing_envs = 0.10


@configclass
class MyDogCleanFlatEnvCfg_PLAY(MyDogCleanFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None