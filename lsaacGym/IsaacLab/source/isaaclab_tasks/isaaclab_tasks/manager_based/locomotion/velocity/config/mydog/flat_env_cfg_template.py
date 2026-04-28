"""Flat environment configuration for MyDog using semantic naming."""

import math
import numpy as np
import torch
from omni.isaac.orbit_assets.mydog import MYDOG_CFG

from isaaclab.managers import SceneEntityCfg
from isaaclab.tasks.manager_based.locomotion.velocity.config.flat_env_cfg import LocomotionVelocityFlatEnvCfg

class LocomotionVelocityFlatEnvCfg(LocomotionVelocityFlatEnvCfg):

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Spawn the robot
        self.scene.robot = MYDOG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # Update the scene
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.scene.robot.init_state.pos = [0.0, 0.0, 0.6]  # Adjust height as needed


class LocomotionVelocityFlatEnvCfg_PLAY(LocomotionVelocityFlatEnvCfg):

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Set the commands
        self.commands.base_velocity.ranges.lin_vel_x = [0.8, 1.0]
        self.commands.base_velocity.ranges.lin_vel_y = [-0.0, 0.0]
        self.commands.base_velocity.ranges.ang_vel_z = [-1.0, 1.0]
        self.commands.base_velocity.ranges.heading = [-math.pi / 3, math.pi / 3]

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

        # Set the command velocities
        self.commands.base_velocity.ranges.lin_vel_x = [1.0, 1.0]
        self.commands.base_velocity.ranges.lin_vel_y = [0.0, 0.0]
        self.commands.base_velocity.ranges.ang_vel_z = [0.0, 0.0]