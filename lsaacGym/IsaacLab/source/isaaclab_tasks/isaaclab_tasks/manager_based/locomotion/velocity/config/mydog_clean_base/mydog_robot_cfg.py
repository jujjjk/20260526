import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

MYDOG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/nszb/python_text/robot_assets/mydog_description/usd/mydog/mydog_rl_ready.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=2.0,
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
            "rf_hip": 0.10,
            "rf_thigh": 0.50,
            "rf_calf": -1.00,

            "lf_hip":  0.10,
            "lf_thigh": -0.50,
            "lf_calf":  1.00,

            "rh_hip":  -0.12,
            "rh_thigh": 0.68,
            "rh_calf": -1.20,

            "lh_hip":   0.12,
            "lh_thigh": -0.68,
            "lh_calf":  1.20,
        },
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "rf_hip", "rf_thigh", "rf_calf",
                "lf_hip", "lf_thigh", "lf_calf",
                "rh_hip", "rh_thigh", "rh_calf",
                "lh_hip", "lh_thigh", "lh_calf",
            ],
            stiffness=100.0,
            damping=5.0,
        ),
    },
)