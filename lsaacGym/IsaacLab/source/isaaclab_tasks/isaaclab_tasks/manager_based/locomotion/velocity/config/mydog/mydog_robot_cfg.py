import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg


def _resolve_mydog_usd_path() -> str:
    """Resolve the robot USD path across local Linux/Windows workspaces."""
    candidates: list[Path] = []

    env_path = os.getenv("MYDOG_USD_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    workspace_root = Path(__file__).resolve().parents[10]
    candidates.extend(
        [
            workspace_root / "fanfan" / "USD" / "fanfan_no_merge.usd",
            workspace_root / "fanfan" / "USD" / "fanfan.usd",
            Path("/home/nszb/python_text/fanfan/USD/fanfan_no_merge.usd"),
            Path("/home/nszb/python_text/fanfan/USD/fanfan.usd"),
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Unable to locate MyDog USD file. Set MYDOG_USD_PATH or place the asset at one of:\n"
        f"{searched}"
    )


MYDOG_USD_PATH = _resolve_mydog_usd_path()

MYDOG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=MYDOG_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=0.3,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=6,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        rot=(1.0, 0.0, 0.0, 0.0),
        pos=(0.0, 0.0, 0.36),
        joint_pos={
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.35,
            "FL_calf_joint": -0.95,

            "FR_hip_joint": 0.0,
            "FR_thigh_joint": 0.35,
            "FR_calf_joint": -0.95,

            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.55,
            "RL_calf_joint": -1.05,

            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 0.55,
            "RR_calf_joint": -1.05,
        },
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            ],
            stiffness=15.0,
            damping=3.0,
        ),
    },
)
