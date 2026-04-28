from isaaclab_tasks.manager_based.locomotion.velocity import mdp as base_mdp
from .joint_semantics import get_sign_tensor


def semantic_joint_pos_rel(env):
    x = base_mdp.joint_pos_rel(env)
    sign = get_sign_tensor(env.device)
    return x * sign


def semantic_joint_vel_rel(env):
    x = base_mdp.joint_vel_rel(env)
    sign = get_sign_tensor(env.device)
    return x * sign