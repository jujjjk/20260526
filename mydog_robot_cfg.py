import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

MYDOG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/nszb/python_text/robot_assets/mydog_description/usd/mydog_rl_ready.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.30),
        joint_pos={
            "rf_hip_joint": 0.0,
            "rf_thigh_joint": 0.85,
            "rf_calf_joint": -1.45,

            "lf_hip_joint": 0.0,
            "lf_thigh_joint": -0.85,
            "lf_calf_joint": 1.45,

            "rh_hip_joint": 0.0,
            "rh_thigh_joint": 0.85,
            "rh_calf_joint": -1.45,

            "lh_hip_joint": 0.0,
            "lh_thigh_joint": -0.85,
            "lh_calf_joint": 1.45,
        },
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "rf_hip_joint", "rf_thigh_joint", "rf_calf_joint",
                "lf_hip_joint", "lf_thigh_joint", "lf_calf_joint",
                "rh_hip_joint", "rh_thigh_joint", "rh_calf_joint",
                "lh_hip_joint", "lh_thigh_joint", "lh_calf_joint",
            ],
            stiffness=35.0,
            damping=1.0,
        ),
    },
)