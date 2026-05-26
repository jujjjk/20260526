from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import FanfanA1CleanRoughEnvCfg


@configclass
class FanfanA1CleanFlatEnvCfg(FanfanA1CleanRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Make the flat task explicitly reject the common "crouch and crawl" solution:
        # keep the trunk up, reward real swing, penalize foot sliding, and discourage
        # any support from thigh/calf segments.
        self.rewards.flat_orientation_l2.weight = -4.0
        self.rewards.feet_air_time.weight = 0.06
        self.rewards.feet_air_time.params["threshold"] = 0.22
        self.rewards.lin_vel_z_l2.weight = -8.0
        self.rewards.ang_vel_xy_l2.weight = -0.35
        self.rewards.action_rate_l2.weight = -0.06
        self.rewards.dof_acc_l2.weight = -8.0e-7
        self.rewards.base_height = RewTerm(
            func=base_mdp.base_height_l2,
            weight=-15.0,
            params={
                "target_height": 0.293,
                "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            },
        )
        self.rewards.joint_deviation = RewTerm(func=base_mdp.joint_deviation_l1, weight=-0.05)
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.35,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            },
        )
        self.rewards.undesired_contacts = RewTerm(
            func=base_mdp.undesired_contacts,
            weight=-2.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_hip|.*_thigh|.*_calf"),
                "threshold": 1.0,
            },
        )

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None
        self.events.add_base_mass = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # Narrow flat training to forward locomotion first so the policy learns a clean
        # stepping pattern before we ask it to handle lateral motion and turning.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.2
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.30)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)


class FanfanA1CleanFlatEnvCfg_PLAY(FanfanA1CleanFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
