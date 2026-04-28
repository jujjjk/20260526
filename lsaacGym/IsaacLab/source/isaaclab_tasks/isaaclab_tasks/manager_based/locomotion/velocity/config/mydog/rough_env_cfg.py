from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

from .mydog_robot_cfg import MYDOG_CFG


@configclass
class MyDogRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # robot
        self.scene.robot = MYDOG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # root link
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

        # smaller action scale for first debugging
        self.actions.joint_pos.scale = 0.50

        # events
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.2, 0.5)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"

        # keep reset close to default standing pose
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = "rf_calf|lf_calf|rh_calf|lh_calf"
        self.rewards.feet_air_time.weight = 0.125
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -1e-5
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"


@configclass
class MyDogRoughEnvCfg_PLAY(MyDogRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None