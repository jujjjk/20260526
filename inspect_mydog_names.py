import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# 1) 基本场景
sim_cfg = sim_utils.SimulationCfg(dt=0.005)
sim = SimulationContext(sim_cfg)

ground_cfg = sim_utils.GroundPlaneCfg()
ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
light_cfg.func("/World/Light", light_cfg)

# 2) 机器人配置
robot_cfg = ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/nszb/python_text/robot_assets/mydog_description/usd/mydog.usd",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.30),
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        )
    },
)

robot = Articulation(cfg=robot_cfg)

# 3) 初始化模拟
sim.reset()

print("\n=== JOINT NAMES ===")
for i, name in enumerate(robot.joint_names):
    print(i, name)

print("\n=== BODY NAMES ===")
for i, name in enumerate(robot.body_names):
    print(i, name)

simulation_app.close()
