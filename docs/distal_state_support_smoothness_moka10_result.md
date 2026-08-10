# Distal state-support and oracle-smoothness result

## Verdict

**Collect additional grouped boundary episodes before changing the model.**
Clean H100 job `37733` completed normally and its independent recomputation
validator accepted the artifact. None of the 15 unseen E05/E10/E15 states lies
inside the frozen training-support envelope. The action-conditioned model is
not authorized, and closed-loop E05 remains blocked.

## Feature shift

Eight of the frozen 62 inputs exceed five training standard deviations:

| Feature | Test maximum absolute z | Fraction of test rows above 5 z |
|---|---:|---:|
| `qdot_rad_s_3` | 10.182 | 0.600 |
| `nominal_first_xyz_0` | 9.342 | 0.467 |
| `nominal_second_xyz_0` | 9.010 | 0.333 |
| `qdot_rad_s_4` | 8.862 | 0.333 |
| `qdot_rad_s_2` | 8.258 | 0.400 |
| `qdot_rad_s_1` | 6.091 | 0.333 |
| `q_rad_2` | 6.079 | 0.267 |
| `q_rad_4` | 5.729 | 0.200 |

The same variables dominate nearest-state distance. Joint velocities 3/4/1,
nominal first-action x, nominal second-action x, and joint velocity 2 account
for `20.35%`, `12.39%`, `8.74%`, `11.87%`, `7.19%`, and `5.21%` of aggregate
test-neighbor squared distance, respectively.

## State support

The p95 nearest-cross-episode training distance is `1.264877` RMS z. Test
nearest-neighbor distances range from `1.410731` to `3.212344`, with median
`2.161661`. Therefore, supported test states are `0/15`. All nearest test
neighbors came from spatial-I-task-1 episode 18 rather than the held-out
goal-II-task-0 population.

Some E15 context vectors stay below five z featurewise, but their combined RMS
distance still exceeds the training p95. This prevents a single extreme
feature rule from hiding multivariate distribution shift.

## Oracle smoothness

The cross-episode training thresholds are `24.387 mm` union-margin RMSE and
`0.152` accepted-set Jaccard. Eleven of 15 test-neighbor pairs pass both.
Failures are E05 steps 181--183 and E10 step 152; test union-margin RMSE spans
`2.234--60.694 mm` and Jaccard spans `0.288--0.888`.

This is useful but not sufficient evidence for learning. Most matched oracle
decisions are locally similar, yet every test state is outside registered
state support. The earlier MLP could therefore fail by extrapolation even when
the nearby oracle field is relatively smooth.

## Decision

The frozen next action is
`collect_additional_grouped_boundary_states_from_new_episodes`. All candidates
derived from one episode must remain in one split. E05/E10/E15 remain test-only
and cannot influence collection thresholds, model fitting, or calibration.
Only after the expanded training population places every test state inside a
newly preregistered support envelope should an action-conditioned conservative
safety-value model be tested.

No training, simulation, QP, or closed-loop execution occurred in this audit.

## Provenance

- Source commit: `6a3327d2a85f5766507a4751b08e72c9e173411d`.
- Slurm: `37733`, `worker-2`, one H100, four CPUs, 32 GB, four seconds.
- Result file SHA-256:
  `114ba547b16f88f1e18c59e0fa245c6fd47782f6a15c9775cbff148070469237`.
- Result payload SHA-256:
  `adcfea06ed6de05a8311e2a4f60a963b2f38c2afa15e5a225c5818eecca122fc`.
- Validation file SHA-256:
  `4de0cd4997dfda6495416cb75a7b21d582f5ede04924d0a06a583289f53b87bc`.
- Full local artifact:
  `output/vlsa_distal_state_support_smoothness/state-support-smoothness-20260810a`.
