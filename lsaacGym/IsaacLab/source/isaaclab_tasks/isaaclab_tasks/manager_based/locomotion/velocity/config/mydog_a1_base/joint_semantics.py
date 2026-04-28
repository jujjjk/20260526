import torch

# 顺序必须和 MYDOG_CFG.actuators["legs"].joint_names_expr 一致
ACTION_SIGN = [
    +1, -1, -1,   # rf_hip, rf_thigh, rf_calf
    +1, +1, +1,   # lf_hip, lf_thigh, lf_calf
    -1, -1, -1,   # rh_hip, rh_thigh, rh_calf
    +1, +1, +1,   # lh_hip, lh_thigh, lh_calf
]

def get_sign_tensor(device):
    return torch.tensor(ACTION_SIGN, dtype=torch.float32, device=device)