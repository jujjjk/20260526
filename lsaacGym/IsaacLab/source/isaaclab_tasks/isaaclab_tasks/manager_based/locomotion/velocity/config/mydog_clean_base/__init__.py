import gymnasium as gym

from .flat_env_cfg import MyDogCleanFlatEnvCfg, MyDogCleanFlatEnvCfg_PLAY


gym.register(
    id="Isaac-Velocity-Flat-MyDogClean-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MyDogCleanFlatEnvCfg,
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.manager_based.locomotion.velocity.config.mydog.agents.rsl_rl_ppo_cfg:"
            "UnitreeA1FlatPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Velocity-Flat-MyDogClean-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MyDogCleanFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.manager_based.locomotion.velocity.config.mydog.agents.rsl_rl_ppo_cfg:"
            "UnitreeA1FlatPPORunnerCfg"
        ),
    },
)