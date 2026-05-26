# Fanfan Phase-Gait Training Notes

This note records the current stable training direction for `FanfanA1Clean`.

## Why symmetry is disabled

The policy now observes a gait clock through `gait_phase_obs`, represented as:

```text
sin(2*pi*phase), cos(2*pi*phase)
```

Mirror augmentation was useful before the phase clock existed, but it is not
currently phase-aware. A left/right mirror of a trot phase should shift the
clock by half a cycle:

```text
sin -> -sin
cos -> -cos
```

Until `agents/symmetry.py` applies this transform to the `gait_phase` entries,
symmetry data augmentation is disabled for learning.

## Why the old GaitReward is disabled

The old `GaitReward` rewards diagonal synchronization in contact and air-time
signals. It does not know which diagonal pair should swing at a given time.
That made it possible to learn a symmetric but low-amplitude dragging gait.

The current line keeps the class and config available, but sets its weight to
zero by default.

## Current reward structure

The main gait-shaping terms are phase-based:

- `phase_trot_foot_clearance`: asks the swing foot to follow a small vertical
  arc even before the foot has become airborne.
- `phase_trot_swing_contact`: penalizes a foot that remains in contact when
  the phase clock says it should be swinging.
- `phase_trot_calf_flexion`: asks the calf joint to fold during the phase
  swing window, without relying on an airborne mask.

The ordinary foot-quality terms remain active but are not increased further:

- `swing_foot_clearance`
- `swing_calf_flexion`
- `contact_foot_drag`
- `long_contact`
- `feet_slide`

Stage 1 prioritizes visible foot lift and natural swing over high speed or
extremely smooth actions.  The RS01 parameterized motor settings remain:

- `Kp = 40`
- `Kd = 5`
- peak torque `17 N*m`
- continuous torque reference `6 N*m`
- velocity limit `33 rad/s`
- action scale: hip `0.12`, thigh `0.22`, calf `0.30`

## Phase relationship

The phase gait uses a fixed low-speed trot clock:

```text
gait_period = 0.55 s
swing_ratio = 0.45
FR and RL: phase offset 0.0
FL and RR: phase offset 0.5
```

So FR/RL swing together, and FL/RR swing half a cycle later.

## Foot height caveat

Current clearance rewards read the height of `.*_foot` body origins. If videos
show foot dragging while the debug height metric claims the foot is above the
clearance target, the body origin is probably not the real toe/sole point.

In that case, add fixed `toe_link` or `sole_link` markers to the URDF/USD:

```text
FR_foot -> FR_toe_link
FL_foot -> FL_toe_link
RR_foot -> RR_toe_link
RL_foot -> RL_toe_link
```

Then point clearance/contact diagnostics at `.*_toe` or `.*_sole` instead of
`.*_foot`.
