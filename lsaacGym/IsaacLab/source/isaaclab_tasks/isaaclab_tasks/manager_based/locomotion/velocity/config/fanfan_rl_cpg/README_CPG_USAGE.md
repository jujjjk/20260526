# Fanfan RL+CPG Usage

This branch keeps `fanfan_a1_clean` untouched and registers new tasks under `FanfanRlCpg`.

## Generate Profiles

Run from the IsaacLab workspace root:

```bash
python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg/tools/analyze_urdf_for_cpg.py
python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg/tools/extract_motor_specs.py
python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg/tools/fit_motor_dynamics_from_real_data.py --csv /path/to/real_log.csv
python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg/tools/generate_randomization_from_urdf.py
```

Generated files:

- `config/motor_profile.yaml`
- `config/motor_profile_fitted.yaml`
- `config/randomization_profile.yaml`
- `logs/cpg_urdf_report.*`
- `logs/motor_dynamics_fit_report.*`
- `logs/randomization_from_urdf_report.*`

## CPG Checks

```bash
python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg/tools/test_cpg_phase.py
python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg/tools/test_cpg_joint_output.py
python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg/tools/test_cpg_residual_action.py
```

These tests require the IsaacLab Python environment with `torch`.

The default simulation CPG basis is `joint_sine`: it keeps the validated stand
pose, uses the URDF joint order, and adds a small swing-phase lift envelope.
`foot_ik` remains experimental and is not the default path.


## Play And Train

```bash
./isaaclab.sh -p scripts/environments/zero_agent.py \
  --task Isaac-Velocity-Flat-FanfanRlCpg-CPGOnly-v0 \
  --num_envs 16

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-FanfanRlCpg-v0 \
  --num_envs 1024 \
  --max_iterations 90000 \
  --headless
```

## Log Gate

```bash
python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg/tools/analyze_cpg_policy_log.py /path/to/policy_log.csv --skip-sec 1.0
python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg/tools/pre_real_deploy_check.py --csv /path/to/policy_log.csv
```

## Deploy-Side ONNX Test

Default mode is pure RL. Enable CPG only explicitly:

```bash
python policy_onnx_test.py --onnx policy.onnx --vx 0.05 --action-mode cpg_only
python policy_onnx_test.py --onnx policy.onnx --vx 0.05 --action-mode cpg_residual
```

Do not walk the robot if `pre_real_deploy_check.py` reports `FAIL`.
