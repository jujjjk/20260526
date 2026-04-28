# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 导入必要的库和模块
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import CapsuleCfg, ConeCfg, CuboidCfg, RigidBodyMaterialCfg, SphereCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .adr_curriculum import CurriculumCfg


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Dexsuite Scene for multi-objects Lifting"""
    
    # 机器人配置，具体机器人类型将在子类中指定
    robot: ArticulationCfg = MISSING

    # 物体配置：定义了可操作物体的属性
    object: RigidObjectCfg = RigidObjectCfg(
        # 物体在场景中的路径模式
        prim_path="{ENV_REGEX_NS}/Object",
        # 多资产生成器配置，随机选择不同形状的物体
        spawn=sim_utils.MultiAssetSpawnerCfg(
            # 包含多种形状的物体：长方体、球体、圆柱体、锥体等
            assets_cfg=[
                CuboidCfg(size=(0.05, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),  # 长方体1
                CuboidCfg(size=(0.05, 0.05, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),  # 长方体2
                CuboidCfg(size=(0.025, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),  # 长方体3
                CuboidCfg(size=(0.025, 0.05, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),  # 长方体4
                CuboidCfg(size=(0.025, 0.025, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),  # 长方体5
                CuboidCfg(size=(0.01, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),  # 长方体6
                SphereCfg(radius=0.05, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),  # 大球体
                SphereCfg(radius=0.025, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),  # 小球体
                CapsuleCfg(radius=0.04, height=0.025, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),  # 圆柱体1
                CapsuleCfg(radius=0.04, height=0.01, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),  # 圆柱体2
                CapsuleCfg(radius=0.04, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),  # 圆柱体3
                CapsuleCfg(radius=0.025, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),  # 圆柱体4
                CapsuleCfg(radius=0.025, height=0.2, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),  # 圆柱体5
                CapsuleCfg(radius=0.01, height=0.2, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),  # 圆柱体6
                ConeCfg(radius=0.05, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),  # 锥体1
                ConeCfg(radius=0.025, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),  # 锥体2
            ],
            # 刚体物理属性配置
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,  # 位置求解迭代次数
                solver_velocity_iteration_count=0,   # 速度求解迭代次数
                disable_gravity=False,               # 启用重力
            ),
            # 碰撞属性配置
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # 质量属性配置
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),  # 物体质量
        ),
        # 初始状态配置
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.1, 0.35)),  # 初始位置
    )

    # 桌子配置
    table: RigidObjectCfg = RigidObjectCfg(
        # 桌子在场景中的路径模式
        prim_path="/World/envs/env_.*/table",
        # 桌子的几何形状（长方体）
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.5, 0.04),  # 桌子尺寸
            # 刚体属性配置（运动学启用表示桌子是固定的）
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            # 碰撞属性
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # 视觉属性：设置为不可见（利用桌子的颜色作为成功指示器）
            visible=False,
        ),
        # 桌子初始状态
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.0, 0.235), rot=(1.0, 0.0, 0.0, 0.0)),  # 位置和旋转
    )

    # 地面平面配置
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",      # 平面路径
        init_state=AssetBaseCfg.InitialStateCfg(),  # 初始状态
        spawn=sim_utils.GroundPlaneCfg(),           # 生成地面
        collision_group=-1,                         # 碰撞组
    )

    # 天空光配置
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",  # 光源路径
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,  # 光照强度
            # HDR纹理文件路径，用于环境光照
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # 定义物体姿态命令配置，用于控制任务目标
    object_pose = mdp.ObjectUniformPoseCommandCfg(
        asset_name="robot",              # 机器人名称
        object_name="object",            # 目标物体名称
        resampling_time_range=(3.0, 5.0), # 重新采样时间范围
        debug_vis=False,                 # 是否显示调试可视化
        # 命令范围配置
        ranges=mdp.ObjectUniformPoseCommandCfg.Ranges(
            pos_x=(-0.7, -0.3),         # x轴位置范围
            pos_y=(-0.25, 0.25),        # y轴位置范围
            pos_z=(0.55, 0.95),         # z轴位置范围
            roll=(-3.14, 3.14),         # 滚转角范围
            pitch=(-3.14, 3.14),        # 俯仰角范围
            yaw=(0.0, 0.0),             # 偏航角范围（固定为0）
        ),
        success_vis_asset_name="table",  # 成功可视化使用的资产名称
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # 物体相对于机器人的四元数姿态
        object_quat_b = ObsTerm(func=mdp.object_quat_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        # 目标物体姿态命令
        target_object_pose_b = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        # 上一动作
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            # 启用观测损坏（增加噪声）
            self.enable_corruption = True
            # 连接所有观测项
            self.concatenate_terms = True
            # 历史长度为5个时间步
            self.history_length = 5

    @configclass
    class ProprioObsCfg(ObsGroup):
        """Observations for proprioception group."""

        # 关节位置
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.0, n_max=0.0))
        # 关节速度
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.0, n_max=0.0))
        # 手指提示状态（末端执行器状态）
        hand_tips_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            # 位置（米）、速度（米/秒）、角速度（弧度/秒）和四元数不太可能超过-2到2的范围
            clip=(-2.0, 2.0),
            params={
                "body_asset_cfg": SceneEntityCfg("robot"),  # 机器人身体配置
                "base_asset_cfg": SceneEntityCfg("robot"),  # 机器人基座配置
            },
        )
        # 接触信息（在子类中定义）
        contact: ObsTerm = MISSING

        def __post_init__(self):
            # 启用观测损坏
            self.enable_corruption = True
            # 连接所有观测项
            self.concatenate_terms = True
            # 历史长度为5个时间步
            self.history_length = 5

    @configclass
    class PerceptionObsCfg(ObsGroup):
        """Observations for perception group."""

        # 物体点云观测
        object_point_cloud = ObsTerm(
            func=mdp.object_point_cloud_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-2.0, 2.0),  # 限制在-2米到2米之间
            params={"num_points": 64, "flatten": True},  # 64个点，展平
        )

        def __post_init__(self):
            # 启用观测损坏
            self.enable_corruption = True
            # 连接维度
            self.concatenate_dim = 0
            # 连接所有观测项
            self.concatenate_terms = True
            # 展平历史维度
            self.flatten_history_dim = True
            # 历史长度为5个时间步
            self.history_length = 5

    # 观测组
    policy: PolicyCfg = PolicyCfg()       # 策略观测组
    proprio: ProprioObsCfg = ProprioObsCfg()  # 本体感知观测组
    perception: PerceptionObsCfg = PerceptionObsCfg()  # 感知观测组


@configclass
class EventCfg:
    """Configuration for randomization."""

    # -- 预启动阶段
    # 随机化物体尺寸
    randomize_object_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,  # 随机化刚体尺寸函数
        mode="prestartup",                    # 在预启动阶段执行
        params={"scale_range": (0.75, 1.5), "asset_cfg": SceneEntityCfg("object")},  # 缩放范围和目标物体
    )

    # 机器人物理材质随机化
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # 随机化刚体材质
        mode="startup",                          # 在启动阶段执行
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),  # 机器人所有身体部件
            "static_friction_range": [0.5, 1.0],     # 静摩擦系数范围
            "dynamic_friction_range": [0.5, 1.0],    # 动摩擦系数范围
            "restitution_range": [0.0, 0.0],         # 恢复系数范围
            "num_buckets": 250,                      # 分桶数量
        },
    )

    # 物体物理材质随机化
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # 随机化刚体材质
        mode="startup",                          # 在启动阶段执行
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),  # 物体所有身体部件
            "static_friction_range": [0.5, 1.0],     # 静摩擦系数范围
            "dynamic_friction_range": [0.5, 1.0],    # 动摩擦系数范围
            "restitution_range": [0.0, 0.0],         # 恢复系数范围
            "num_buckets": 250,                      # 分桶数量
        },
    )

    # 关节刚度和阻尼随机化
    joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,  # 随机化执行器增益
        mode="startup",                     # 在启动阶段执行
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),  # 机器人所有关节
            "stiffness_distribution_params": [0.5, 2.0],  # 刚度分布参数
            "damping_distribution_params": [0.5, 2.0],    # 阻尼分布参数
            "operation": "scale",                         # 缩放操作
        },
    )

    # 关节摩擦随机化
    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,  # 随机化关节参数
        mode="startup",                       # 在启动阶段执行
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),  # 机器人所有关节
            "friction_distribution_params": [0.0, 5.0],  # 摩擦分布参数
            "operation": "scale",                        # 缩放操作
        },
    )

    # 物体质量和尺寸随机化
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,  # 随机化刚体质量
        mode="startup",                      # 在启动阶段执行
        params={
            "asset_cfg": SceneEntityCfg("object"),        # 目标物体
            "mass_distribution_params": [0.2, 2.0],       # 质量分布参数
            "operation": "scale",                         # 缩放操作
        },
    )

    # 重置桌子位置
    reset_table = EventTerm(
        func=mdp.reset_root_state_uniform,  # 统一重置根状态
        mode="reset",                       # 在重置阶段执行
        params={
            "pose_range": {"x": [-0.05, 0.05], "y": [-0.05, 0.05], "z": [0.0, 0.0]},  # 位姿范围
            "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},  # 速度范围
            "asset_cfg": SceneEntityCfg("table"),  # 目标桌子
        },
    )

    # 重置物体位置
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,  # 统一重置根状态
        mode="reset",                       # 在重置阶段执行
        params={
            "pose_range": {
                "x": [-0.2, 0.2],          # x位置范围
                "y": [-0.2, 0.2],          # y位置范围
                "z": [0.0, 0.4],           # z位置范围
                "roll": [-3.14, 3.14],     # 滚转角范围
                "pitch": [-3.14, 3.14],    # 俯仰角范围
                "yaw": [-3.14, 3.14],      # 偏航角范围
            },
            "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},  # 速度范围
            "asset_cfg": SceneEntityCfg("object"),  # 目标物体
        },
    )

    # 重置机器人根状态
    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,  # 统一重置根状态
        mode="reset",                       # 在重置阶段执行
        params={
            "pose_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "yaw": [-0.0, 0.0]},  # 位姿范围
            "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},  # 速度范围
            "asset_cfg": SceneEntityCfg("robot"),  # 目标机器人
        },
    )

    # 重置机器人关节
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,  # 通过偏移重置关节
        mode="reset",                     # 在重置阶段执行
        params={
            "position_range": [-0.50, 0.50],  # 位置范围
            "velocity_range": [0.0, 0.0],     # 速度范围
        },
    )

    # 重置机器人腕部关节
    reset_robot_wrist_joint = EventTerm(
        func=mdp.reset_joints_by_offset,  # 通过偏移重置关节
        mode="reset",                     # 在重置阶段执行
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="iiwa7_joint_7"),  # 特定关节
            "position_range": [-3, 3],    # 位置范围
            "velocity_range": [0.0, 0.0], # 速度范围
        },
    )

    # 注意（Octi）：这是Remake中的一个有意技巧，通过课程学习加速训练
    # 通过将重力作为课程安排——从无重力（简单）开始，逐渐引入完整重力（困难）——代理可以更平稳地学习
    # 这消除了特殊"Lift"奖励的需要（通常用于推动代理对抗重力），这简化了奖励组合的整体效果
    variable_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,  # 随机化物理场景重力
        mode="reset",                               # 在重置阶段执行
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),  # 重力分布参数
            "operation": "abs",                   # 绝对值操作
        },
    )


@configclass
class ActionsCfg:
    pass  # 动作配置（在此环境中为空）


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # L2范数惩罚项，鼓励较小的动作幅度
    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-0.005)

    # 动作变化率的L2范数惩罚，鼓励平滑的动作
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.005)

    # 手指到物体的距离奖励，鼓励接近物体
    fingers_to_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.4}, weight=1.0)

    # 位置跟踪奖励，鼓励机器人末端执行器跟踪目标位置
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,  # 使用tanh函数的位置误差
        weight=2.0,                           # 权重
        params={
            "asset_cfg": SceneEntityCfg("robot"),      # 机器人配置
            "std": 0.2,                              # 标准差
            "command_name": "object_pose",             # 命令名称
            "align_asset_cfg": SceneEntityCfg("object"), # 对齐的资产配置
        },
    )

    # 方向跟踪奖励，鼓励机器人末端执行器匹配目标方向
    orientation_tracking = RewTerm(
        func=mdp.orientation_command_error_tanh,  # 使用tanh函数的方向误差
        weight=4.0,                              # 权重
        params={
            "asset_cfg": SceneEntityCfg("robot"),      # 机器人配置
            "std": 1.5,                              # 标准差
            "command_name": "object_pose",             # 命令名称
            "align_asset_cfg": SceneEntityCfg("object"), # 对齐的资产配置
        },
    )

    # 成功奖励，当机器人成功完成任务时给予
    success = RewTerm(
        func=mdp.success_reward,                    # 成功奖励函数
        weight=10,                                  # 权重
        params={
            "asset_cfg": SceneEntityCfg("robot"),      # 机器人配置
            "pos_std": 0.1,                          # 位置标准差
            "rot_std": 0.5,                          # 旋转标准差
            "command_name": "object_pose",             # 命令名称
            "align_asset_cfg": SceneEntityCfg("object"), # 对齐的资产配置
        },
    )

    # 提前终止惩罚，当环境提前终止时给予负奖励
    early_termination = RewTerm(func=mdp.is_terminated_term, weight=-1, params={"term_keys": "abnormal_robot"})


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # 时间到终止条件
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 物体超出边界终止条件
    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,  # 边界检查函数
        params={
            "in_bound_range": {"x": (-1.5, 0.5), "y": (-2.0, 2.0), "z": (0.0, 2.0)},  # 有效边界范围
            "asset_cfg": SceneEntityCfg("object"),  # 检查的资产配置
        },
    )

    # 异常机器人状态终止条件
    abnormal_robot = DoneTerm(func=mdp.abnormal_robot_state)


@configclass
class DexsuiteReorientEnvCfg(ManagerBasedEnvCfg):
    """Dexsuite reorientation task definition, also the base definition for derivative Lift task and evaluation task"""

    # 场景设置
    viewer: ViewerCfg = ViewerCfg(eye=(-2.25, 0.0, 0.75), lookat=(0.0, 0.0, 0.45), origin_type="env")  # 查看器配置
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)  # 场景配置：4096个环境，环境间距为3，不复制物理
    # 基本设置
    observations: ObservationsCfg = ObservationsCfg()  # 观测配置
    actions: ActionsCfg = ActionsCfg()                # 动作配置
    commands: CommandsCfg = CommandsCfg()             # 命令配置
    # MDP设置
    rewards: RewardsCfg = RewardsCfg()                # 奖励配置
    terminations: TerminationsCfg = TerminationsCfg()  # 终止配置
    events: EventCfg = EventCfg()                     # 事件配置
    curriculum: CurriculumCfg | None = CurriculumCfg()  # 课程学习配置

    def __post_init__(self):
        """Post initialization."""
        # 通用设置
        self.decimation = 2  # 决策频率为50Hz（仿真频率120Hz / decimation 2 = 60Hz，这里设置为2应该是50Hz）

        # *单目标设置
        # 设置重新采样时间范围为固定值
        self.commands.object_pose.resampling_time_range = (10.0, 10.0)
        # 不仅考虑位置，也考虑方向
        self.commands.object_pose.position_only = False
        # 设置成功可视化器配置
        self.commands.object_pose.success_visualizer_cfg.markers["failure"] = self.scene.table.spawn.replace(
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.15, 0.15), roughness=0.25), visible=True  # 失败时的红色标记
        )
        self.commands.object_pose.success_visualizer_cfg.markers["success"] = self.scene.table.spawn.replace(
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.25, 0.15), roughness=0.25), visible=True  # 成功时的绿色标记
        )

        # 每个episode长度为4秒
        self.episode_length_s = 4.0
        # 是否为有限时间范围（True表示是）
        self.is_finite_horizon = True

        # 仿真设置
        self.sim.dt = 1 / 120  # 仿真时间步长（约8.33毫秒）
        self.sim.render_interval = self.decimation  # 渲染间隔
        self.sim.physx.bounce_threshold_velocity = 0.2  # PhysX反弹阈值速度
        self.sim.physx.bounce_threshold_velocity = 0.01  # PhysX反弹阈值速度（重复设置，第二个覆盖第一个）
        self.sim.physx.gpu_max_rigid_patch_count = 4 * 5 * 2**15  # GPU最大刚体补丁数量

        # 如果有课程学习配置，则更新相关参数
        if self.curriculum is not None:
            # 设置位置容差为成功奖励中位置标准差的一半
            self.curriculum.adr.params["pos_tol"] = self.rewards.success.params["pos_std"] / 2
            # 设置旋转容差为成功奖励中旋转标准差的一半
            self.curriculum.adr.params["rot_tol"] = self.rewards.success.params["rot_std"] / 2


class DexsuiteLiftEnvCfg(DexsuiteReorientEnvCfg):
    """Dexsuite lift task definition"""

    def __post_init__(self):
        super().__post_init__()  # 调用父类的__post_init__方法
        # 移除方向跟踪奖励（只关注位置，不关注方向）
        self.rewards.orientation_tracking = None  # no orientation reward
        # 只考虑位置，不考虑方向
        self.commands.object_pose.position_only = True
        if self.curriculum is not None:
            # 让成功奖励不考虑方向
            self.rewards.success.params["rot_std"] = None  # make success reward not consider orientation
            # 让ADR（自适应域随机化）不跟踪方向
            self.curriculum.adr.params["rot_tol"] = None  # make adr not tracking orientation


class DexsuiteReorientEnvCfg_PLAY(DexsuiteReorientEnvCfg):
    """Dexsuite reorientation task evaluation environment definition"""

    def __post_init__(self):
        super().__post_init__()  # 调用父类的__post_init__方法
        # 更频繁的目标重采样，用于评估
        self.commands.object_pose.resampling_time_range = (2.0, 3.0)
        # 启用调试可视化
        self.commands.object_pose.debug_vis = True
        # 设置ADR的初始难度为最大难度
        self.curriculum.adr.params["init_difficulty"] = self.curriculum.adr.params["max_difficulty"]


class DexsuiteLiftEnvCfg_PLAY(DexsuiteLiftEnvCfg):
    """Dexsuite lift task evaluation environment definition"""

    def __post_init__(self):
        super().__post_init__()  # 调用父类的__post_init__方法
        # 更频繁的目标重采样，用于评估
        self.commands.object_pose.resampling_time_range = (2.0, 3.0)
        # 启用调试可视化
        self.commands.object_pose.debug_vis = True
        # 只考虑位置，不考虑方向（再次明确设置）
        self.commands.object_pose.position_only = True
        # 设置ADR的初始难度为最大难度
        self.curriculum.adr.params["init_difficulty"] = self.curriculum.adr.params["max_difficulty"]