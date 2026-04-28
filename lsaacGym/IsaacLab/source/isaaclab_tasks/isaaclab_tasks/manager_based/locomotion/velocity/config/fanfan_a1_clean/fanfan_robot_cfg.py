import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


def _resolve_fanfan_urdf_path() -> str:
    env_path = os.environ.get("FANFAN_URDF_PATH")
    if env_path:
        return env_path

    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        candidate = parent / "fanfan" / "urdf" / "fanfan.urdf"
        if candidate.exists():
            return str(candidate)

    return str(current_file.parents[10] / "fanfan" / "urdf" / "fanfan.urdf")


def _resolve_fanfan_usd_dir() -> str:
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        candidate = parent / "fanfan" / "USD"
        if candidate.exists():
            return str(candidate)

    return str(current_file.parents[10] / "fanfan" / "USD")


FANFAN_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_resolve_fanfan_urdf_path(),
        usd_dir=_resolve_fanfan_usd_dir(),
        usd_file_name="fanfan_no_merge.usd",
        fix_base=False,
        # Fixed-joint links such as *_foot must stay as real runtime articulation bodies.
        merge_fixed_joints=False,
        # Instanceable USDs can hide fixed-joint helper links from runtime sensors/views.
        make_instanceable=False,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=3.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=8,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=40.0,
                damping=4.0,
            ),
            target_type="position",
            drive_type="force",
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Second-pass fanfan tuning: keep the A1-like stance template, but
        # open the legs a bit more so the robot starts from a taller, more
        # step-friendly posture instead of a squat that encourages dragging.
        pos=(0.0, 0.0, 0.372),
        joint_pos={
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": 0.62,
            "FR_calf_joint": -1.58,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.62,
            "FL_calf_joint": -1.58,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 0.95,
            "RR_calf_joint": -1.62,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.95,
            "RL_calf_joint": -1.62,
        },
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "FR_hip_joint",
                "FR_thigh_joint",
                "FR_calf_joint",
                "FL_hip_joint",
                "FL_thigh_joint",
                "FL_calf_joint",
                "RR_hip_joint",
                "RR_thigh_joint",
                "RR_calf_joint",
                "RL_hip_joint",
                "RL_thigh_joint",
                "RL_calf_joint",
            ],
            # Match A1-level authority so the more open default posture can be
            # held without immediately sagging back into a low crouch.
            effort_limit_sim=40.0,
            velocity_limit_sim=40.0,
            stiffness=80.0,
            damping=4,
        ),
    },
)
