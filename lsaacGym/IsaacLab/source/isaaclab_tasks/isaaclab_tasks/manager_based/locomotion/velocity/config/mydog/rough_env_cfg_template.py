"""Rough environment configuration for MyDog using semantic naming."""

import math
import numpy as np
import torch
from omni.isaac.orbit_assets.mydog import MYDOG_CFG

from isaaclab.managers import SceneEntityCfg
from isaaclab.tasks.manager_based.locomotion.velocity.config.rough_env_cfg import LocomotionVelocityRoughEnvCfg

class LocomotionVelocityRoughEnvCfg(LocomotionVelocityRoughEnvCfg):

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Set the terrain type
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = None  # Will be set by task below

        # Spawn the robot
        self.scene.robot = MYDOG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # Override some defaults
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.scene.robot.init_state.pos = [0.0, 0.0, 0.6]  # Adjust height as needed
        
        # Update observations
        self.observations.policy.enable_corruption = True
        self.observations.policy.concatenate_commands = True


class LocomotionVelocityRoughEnvCfg_PLAY(LocomotionVelocityRoughEnvCfg):

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        
        # Set the commands
        self.commands.base_velocity.ranges.lin_vel_x = [0.8, 1.0]
        self.commands.base_velocity.ranges.lin_vel_y = [-0.0, 0.0]
        self.commands.base_velocity.ranges.ang_vel_z = [-1.0, 1.0]
        self.commands.base_velocity.ranges.heading = [-math.pi / 3, math.pi / 3]

        # Set the heights of the terrains
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_slope"].difficulty = 0.35
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_slope_inv"].difficulty = 0.35
        self.scene.terrain.terrain_generator.sub_terrains["box_room"].difficulty = 0.35
        self.scene.terrain.terrain_generator.sub_terrains["multiple_pillars"].difficulty = 0.35
        self.scene.terrain.terrain_generator.sub_terrains["slope"].difficulty = 0.35
        self.scene.terrain.terrain_generator.sub_terrains["wave"].difficulty = 0.35

        # Set the robot settings
        self.scene.robot.spawn.actuator_props.stiffness = 35.0
        self.scene.robot.spawn.actuator_props.damping = 1.0
        self.scene.robot.cfg.init_state.joint_pos = {
            "rf_hip_joint": 0.0,
            "rf_thigh_joint": 0.9,
            "rf_calf_joint": -1.8,

            "lf_hip_joint": 0.0,
            "lf_thigh_joint": -0.9,
            "lf_calf_joint": 1.8,

            "rh_hip_joint": 0.0,
            "rh_thigh_joint": 0.9,
            "rh_calf_joint": -1.8,

            "lh_hip_joint": 0.0,
            "lh_thigh_joint": -0.9,
            "lh_calf_joint": 1.8,
        }

        # Turn off terrain generator
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Set the command velocities
        self.commands.base_velocity.ranges.lin_vel_x = [1.0, 1.0]
        self.commands.base_velocity.ranges.lin_vel_y = [0.0, 0.0]
        self.commands.base_velocity.ranges.ang_vel_z = [0.0, 0.0]