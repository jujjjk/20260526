from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from .fanfan_robot_cfg import FANFAN_CFG


@configclass
class FanfanA1CleanRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = FANFAN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Trunk"

        # Keep the A1 structure and only scale terrain difficulty to the smaller frame.
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.015, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.005, 0.04)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.005

        # Real-robot conservative policy: the 6 Nm motors cannot reliably track
        # the aggressive 0.22 rad target jumps learned in simulation.
        self.actions.joint_pos.scale = 0.10

        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.2, 0.6)
        self.events.add_base_mass.params["asset_cfg"].body_names = "Trunk"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "Trunk"
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

        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -2.0e-4
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.action_rate_l2.weight = -0.06
        self.rewards.dof_acc_l2.weight = -5.0e-7

        self.terminations.base_contact.params["sensor_cfg"].body_names = "Trunk"


@configclass
class FanfanA1CleanRoughEnvCfg_PLAY(FanfanA1CleanRoughEnvCfg):
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
