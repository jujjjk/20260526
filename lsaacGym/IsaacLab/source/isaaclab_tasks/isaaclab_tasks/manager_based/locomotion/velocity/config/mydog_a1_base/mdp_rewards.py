import torch
from isaaclab.managers import SceneEntityCfg


def _get_command_norm(env, command_name: str = "base_velocity") -> torch.Tensor:
    cmds = env.command_manager.get_command(command_name)
    # velocity task 一般前 3 维是 vx, vy, wz
    return torch.norm(cmds[:, :3], dim=1)


def _quat_apply_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    q: [N, 4] in (w, x, y, z)
    v: [N, 3] local/body-frame vector
    """
    qw = q[:, 0:1]
    qxyz = q[:, 1:4]
    t = 2.0 * torch.cross(qxyz, v, dim=1)
    return v + qw * t + torch.cross(qxyz, t, dim=1)


def _get_base_height(
    env,
    asset_name: str = "robot",
    offset_body=(-0.10, 0.0, 0.0),
) -> torch.Tensor:
    """
    在机体坐标系下，把高度参考点沿 body x 方向后移 10 cm。
    你的机体是 +X 朝前，所以后移就是 x = -0.10。
    """
    robot = env.scene[asset_name]

    root_pos_w = robot.data.root_pos_w       # [N, 3]
    root_quat_w = robot.data.root_quat_w     # [N, 4], (w, x, y, z)

    offset = torch.tensor(offset_body, dtype=root_pos_w.dtype, device=root_pos_w.device)
    offset = offset.unsqueeze(0).repeat(root_pos_w.shape[0], 1)

    offset_w = _quat_apply_wxyz(root_quat_w, offset)
    point_w = root_pos_w + offset_w

    return point_w[:, 2]

def _get_feet_contact_mask(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """
    返回 shape = [num_envs, num_feet] 的 bool mask
    只统计 sensor_cfg.body_ids 指定的那些 body（这里就是四个 *_calf）
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    # IsaacLab velocity 任务里常见做法：用 history，取最近几帧中的最大接触
    # 形状通常是 [num_envs, history, num_bodies, 3]
    net_contact_forces = contact_sensor.data.net_forces_w_history

    # 只取目标脚体对应的 body ids
    # -> [num_envs, history, num_feet, 3]
    foot_forces = net_contact_forces[:, :, sensor_cfg.body_ids, :]

    # 每帧每只脚的力范数 -> [num_envs, history, num_feet]
    foot_force_norm = torch.norm(foot_forces, dim=-1)

    # 在 history 维上取最大，判断最近几帧是否接触
    # -> [num_envs, num_feet]
    is_contact = torch.max(foot_force_norm, dim=1)[0] > threshold
    return is_contact


def _get_feet_contact_count(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    is_contact = _get_feet_contact_mask(env, sensor_cfg=sensor_cfg, threshold=threshold)
    return is_contact.float().sum(dim=1)



def stand_height_reward(
    env,
    target_height: float = 0.36,
    sigma: float = 0.02,
    cmd_threshold: float = 0.08,
    command_name: str = "base_velocity",
    asset_name: str = "robot",
    offset_body=(-0.10, 0.0, 0.0),
):
    cmd_norm = _get_command_norm(env, command_name=command_name)
    z = _get_base_height(env, asset_name=asset_name, offset_body=offset_body)

    reward = torch.exp(-((z - target_height) ** 2) / (2 * sigma * sigma))

    mask = (cmd_norm < cmd_threshold).float()
    return reward * mask

def stand_feet_contact_reward(
    env,
    sensor_cfg: SceneEntityCfg,
    desired_contacts: float = 4.0,
    threshold: float = 1.0,
    cmd_threshold: float = 0.08,
    command_name: str = "base_velocity",
):
    cmd_norm = _get_command_norm(env, command_name=command_name)
    n_contact = _get_feet_contact_count(env, sensor_cfg=sensor_cfg, threshold=threshold)

    # 4 足接触最好，少脚则变差
    reward = n_contact / desired_contacts
    reward = torch.clamp(reward, 0.0, 1.0)

    mask = (cmd_norm < cmd_threshold).float()
    return reward * mask



def stand_bad_height_penalty(
    env,
    min_height: float = 0.33,
    max_height: float = 0.39,
    cmd_threshold: float = 0.08,
    command_name: str = "base_velocity",
    asset_name: str = "robot",
    offset_body=(-0.10, 0.0, 0.0),
):
    cmd_norm = _get_command_norm(env, command_name=command_name)
    z = _get_base_height(env, asset_name=asset_name, offset_body=offset_body)

    too_low = (z < min_height).float()
    too_high = (z > max_height).float()
    penalty = too_low + too_high

    mask = (cmd_norm < cmd_threshold).float()
    return penalty * mask

def stand_few_contacts_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
    min_contacts: float = 3.5,
    threshold: float = 1.0,
    cmd_threshold: float = 0.08,
    command_name: str = "base_velocity",
):
    cmd_norm = _get_command_norm(env, command_name=command_name)
    n_contact = _get_feet_contact_count(env, sensor_cfg=sensor_cfg, threshold=threshold)

    penalty = (n_contact < min_contacts).float()

    mask = (cmd_norm < cmd_threshold).float()
    return penalty * mask



def stand_pose_reward(
    env,
    cmd_threshold: float = 0.08,
    command_name: str = "base_velocity",
):
    """
    当命令很小时，奖励关节回到默认站姿附近。
    joint_pos_rel 本身就是相对默认站姿的偏差。
    偏差越小，reward 越高。
    """
    cmd_norm = _get_command_norm(env, command_name=command_name)

    # 这里拿已经“相对默认站姿”的关节偏差
    # 如果你当前 rough_env_cfg 已经把 observation 切成 semantic_joint_pos_rel，
    # 这里仍建议直接用 base_mdp.joint_pos_rel 的原始接口逻辑；
    # 因为本质是“离默认站姿多远”，符号翻不翻都不影响平方范数
    from isaaclab_tasks.manager_based.locomotion.velocity import mdp as base_mdp
    q_rel = base_mdp.joint_pos_rel(env)   # [num_envs, 12]

    # 二范数越小越好，这里用 exp 做平滑奖励
    pose_err = torch.sum(q_rel * q_rel, dim=1)
    reward = torch.exp(-2.0 * pose_err)

    mask = (cmd_norm < cmd_threshold).float()
    return reward * mask


def stand_joint_vel_penalty(
    env,
    cmd_threshold: float = 0.08,
    command_name: str = "base_velocity",
):
    """
    当命令很小时，惩罚关节速度过大，避免站姿附近还在抖。
    """
    cmd_norm = _get_command_norm(env, command_name=command_name)

    from isaaclab_tasks.manager_based.locomotion.velocity import mdp as base_mdp
    dq = base_mdp.joint_vel_rel(env)   # [num_envs, 12]

    penalty = torch.sum(dq * dq, dim=1)

    mask = (cmd_norm < cmd_threshold).float()
    return penalty * mask

def stand_pose_reward(
    env,
    cmd_threshold: float = 0.08,
    command_name: str = "base_velocity",
):
    from isaaclab_tasks.manager_based.locomotion.velocity import mdp as base_mdp

    cmd_norm = _get_command_norm(env, command_name=command_name)
    q_rel = base_mdp.joint_pos_rel(env)

    pose_err = torch.sum(q_rel * q_rel, dim=1)
    reward = torch.exp(-2.0 * pose_err)

    mask = (cmd_norm < cmd_threshold).float()
    return reward * mask


def stand_joint_vel_penalty(
    env,
    cmd_threshold: float = 0.08,
    command_name: str = "base_velocity",
):
    from isaaclab_tasks.manager_based.locomotion.velocity import mdp as base_mdp

    cmd_norm = _get_command_norm(env, command_name=command_name)
    dq = base_mdp.joint_vel_rel(env)

    penalty = torch.sum(dq * dq, dim=1)

    mask = (cmd_norm < cmd_threshold).float()
    return penalty * mask


def front_feet_contact_reward(
    env,
    sensor_cfg: SceneEntityCfg,
    desired_contacts: float = 2.0,
    threshold: float = 1.0,
    cmd_threshold: float = 0.08,
    command_name: str = "base_velocity",
):
    cmd_norm = _get_command_norm(env, command_name=command_name)
    n_contact = _get_feet_contact_count(env, sensor_cfg=sensor_cfg, threshold=threshold)

    reward = n_contact / desired_contacts
    reward = torch.clamp(reward, 0.0, 1.0)

    mask = (cmd_norm < cmd_threshold).float()
    return reward * mask


def rear_feet_contact_reward(
    env,
    sensor_cfg: SceneEntityCfg,
    desired_contacts: float = 2.0,
    threshold: float = 1.0,
    cmd_threshold: float = 0.08,
    command_name: str = "base_velocity",
):
    cmd_norm = _get_command_norm(env, command_name=command_name)
    n_contact = _get_feet_contact_count(env, sensor_cfg=sensor_cfg, threshold=threshold)

    reward = n_contact / desired_contacts
    reward = torch.clamp(reward, 0.0, 1.0)

    mask = (cmd_norm < cmd_threshold).float()
    return reward * mask


def rear_few_contacts_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
    min_contacts: float = 1.5,
    threshold: float = 1.0,
    cmd_threshold: float = 0.08,
    command_name: str = "base_velocity",
):
    cmd_norm = _get_command_norm(env, command_name=command_name)
    n_contact = _get_feet_contact_count(env, sensor_cfg=sensor_cfg, threshold=threshold)

    penalty = (n_contact < min_contacts).float()

    mask = (cmd_norm < cmd_threshold).float()
    return penalty * mask