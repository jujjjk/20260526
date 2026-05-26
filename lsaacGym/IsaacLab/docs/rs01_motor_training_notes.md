# RS01 Motor Training Notes for FanfanA1Clean

This note records the parameterized motor setup used for the FanfanA1Clean Isaac Lab task.
The goal of this pass is to stop using the learned actuator network by default and train with
a conservative RS01-based implicit PD actuator.

## RS01 manual parameters

| Item | Value |
| --- | ---: |
| Peak torque | 17 N*m |
| Rated continuous load torque | 6 N*m |
| No-load speed | 315 rpm, about 33 rad/s |
| Rated-load speed | 100 rpm, about 10.47 rad/s |
| Rated voltage | 36 VDC |
| Voltage range | 24-50 VDC |
| Peak phase current | 23 Apk |
| Rated-load phase current | 7 Apk |
| Torque constant | 1.22 N*m/Arms |
| Reduction ratio | 7.75:1 |
| Motor mass | about 0.38 kg |
| CAN bus | 1 Mbps |
| Motion-control torque range | -17 to 17 N*m |
| Motion-control position range | -12.57 to 12.57 rad |
| Motion-control velocity range | -44 to 44 rad/s |
| Motion-control Kp range | 0 to 500 |
| Motion-control Kd range | 0 to 5 |

The current real robot uses `Kp=40` and `Kd=5`, so the first training actuator also uses
these gains.

## Training interpretation

- Simulator effort limit is `17 N*m`, matching RS01 peak torque.
- Reward penalties use `6 N*m` as the continuous-torque reference.
- Velocity limit is `33 rad/s`, matching the manual no-load speed approximation.
- Action scale is set to the safe value `0.15 rad` for each joint.  The normal value
  `0.20 rad` is about 11.5 degrees and can be tried after the safe version is stable.
- Armature `0.01` and friction `0.08` are engineering initial values, not manual values.
  They must be identified from real joint step response and free-swing tests.

## Domain randomization

The task uses existing Isaac Lab events only:

- stiffness scale: `0.8` to `1.2`
- damping scale: `0.8` to `1.1`
- joint friction absolute range: `0.03` to `0.15`
- armature absolute range: `0.005` to `0.02`

Future work:

- Add motor strength randomization around `0.75` to `1.05`.
- Add action delay randomization, for example `0` to `2` control steps.

These are intentionally not added in this pass because doing them cleanly would require a
small custom event or action wrapper.

## URDF/USD mass warning

Each RS01 motor is about `0.38 kg`.  A 12-DOF robot has about `4.56 kg` of motor mass.
If the URDF/USD only contains link shell mass and omits the motors, the simulated robot will
be much lighter than the real robot.  A policy trained on that model may walk in simulation
but fail to stand, lift legs, or avoid motor protection on hardware.

Before serious sim-to-real deployment, check:

- whether motor mass is included in each hip/thigh/calf link,
- whether COM locations match the installed motors,
- whether link inertias include motor rotor and gearbox effects,
- whether foot contact geometry/friction matches the real feet.

## Deployment reminder

The real controller should still limit and smooth target joint positions.  A good first
hardware-side limit is:

```text
max_delta_q_target = 0.02 to 0.03 rad/control step
```

Torque clipping alone is not enough; a large one-step target jump can still cause motor
protection or unstable contact behavior.
