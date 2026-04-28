from isaaclab.utils import configclass

from .rough_env_cfg import MyDogRoughEnvCfg


@configclass
class MyDogFlatEnvCfg(MyDogRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -1.5
        self.rewards.feet_air_time.weight = 0.15

        # flat terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # no terrain curriculum
        self.curriculum.terrain_levels = None


@configclass
class MyDogFlatEnvCfg_PLAY(MyDogFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None