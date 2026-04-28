from isaaclab_tasks.manager_based.locomotion.velocity import mdp as base_mdp
from .joint_semantics import get_sign_tensor


def semantic_joint_pos_rel(env):
    # 这是已经减过默认站姿的相对关节角
    x = base_mdp.joint_pos_rel(env)
    sign = get_sign_tensor(env.device)
    return x * sign


def semantic_joint_vel_rel(env):
    x = base_mdp.joint_vel_rel(env)
    sign = get_sign_tensor(env.device)
    return x * sign


def semantic_last_action(env):
    x = base_mdp.last_action(env)
    sign = get_sign_tensor(env.device)
    return x * sign