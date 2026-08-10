# Distal rollout-margin target-authority result

## Verdict

**PASS for the scoped L5--L7 margin target.**

Clean H100 job `37808` completed on `worker-1` in `00:00:03`; its independent
validator accepted the artifact. Across all 13,025 immutable two-action
cloned-OSC rollouts, zero actions had seven nonnegative ellipsoid margins while
raw MuJoCo reported protected L5--L7 contact.

| population | actions | dangerous target false-safes |
|---|---:|---:|
| 85-state fit grids | 10,625 | 0 |
| 25-state validation/test off-grid | 2,400 | 0 |
| total | 13,025 | 0 |

This supports using the seven quantitative minimum-substep ellipsoid margins
as learning targets for the current moka-pot L5--L7 experiment. It does not
establish whole-arm safety, EE safety, CAR, task success, or model calibration.
No new simulation, training, QP, or closed-loop execution occurred.

The result authorizes a separately preregistered leave-one-episode-out local
residual bound. Closed-loop E05 remains blocked.

## Immutable evidence

- run root: `/mnt/data/quanth/experiments/vlsa-distal-margin-target-authority/margin-target-authority-20260810a`
- result file SHA-256: `c29a99383f50012b3f9b0b4fadcbbb320bcd42a7932d8d5079144ecb460ccce2`
- result payload SHA-256: `9e881dca46c06d63acee6bd20e7313016679f6fd4ab2dc91ebc0acfb95394e3d`
- validation file SHA-256: `05b8a517dcd1286cc30b486d1cc0bc2f1fdcfd223a919b28ab9d892877480fd2`
- source commit: `d78c7c3a6d3e92998f22914775e15a4866221eb1`

