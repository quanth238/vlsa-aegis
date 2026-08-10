# Same-task boundary coverage expansion result

## Verdict

**NO-GO for retraining.** The additional same-task episodes materially reduce
the feature shift and recover support for 7/15 immutable test states, but the
registered gate requires 15/15. No MLP was trained and closed-loop E05 did not
run.

## H100 evidence

- Slurm job: `37742`, `COMPLETED` on `worker-2` in `00:09:02`.
- Clean source: `67b67aa7fb8f44af6aa1219df762170fbb4d3550`.
- New episodes: E00/E20 training and E30 validation; E05/E10/E15 remained
  test-only.
- New exact labels: 15 states, 1,875 two-step cloned-OSC grid rollouts.
- Expanded split: 40 training, 10 validation, and 15 test states.
- Independent recomputation status: `valid`.

The inherited result field `new_simulation_executed=false` describes the pure
support-analysis subroutine. The same result explicitly records
`boundary_collection_simulation_executed=true`; the 1,875 H100 rollouts above
were executed. The post-result source clarifies this metadata for future runs.

## Coverage result

| Measure | Before | After |
|---|---:|---:|
| Supported test states | 0/15 | 7/15 |
| Smooth test-state oracle pairs | 11/15 | 15/15 |
| Maximum test feature shift | 10.182 z | 5.601 z |
| Features above 5 z | 8 | 2 |

All five E05 states are now supported. E15 steps 149--150 are supported, while
steps 146--148 remain outside the frozen distance threshold. All five E10
states remain unsupported. The remaining shifted features are the nominal
second-action x coordinate (`5.601 z`) and nominal first-action x coordinate
(`5.035 z`).

The expanded training-derived support threshold is `0.972458` RMS z. E10
distances are `1.260280--1.679309`; E15 steps 146--148 are
`1.011380--1.188877`. The dominant nearest-neighbor distance contributions are
the nominal first/second Cartesian actions followed by joint velocities 4 and
2. This confirms partial but incomplete coverage of precisely the features
identified in job `37733`.

The regional oracle itself is not the blocker: all 15 test comparisons pass
the frozen induced-value/accepted-set smoothness gate.

## Separate collector compatibility gate

The additional raw label population is complete, but the older single-affine
collector reports `dataset_gate_pass=false`. Thirteen of 15 states have a
valid single-affine certificate and exact-safe QP. The exact crossing states
E00 step 167 and E30 step 137 have zero safe actions in the sampled trust
region, so their single-affine certificate/QP is undefined. These states are
retained as negative regional safety data; they are not silently dropped.

This compatibility failure is not the reason retraining is blocked. The
decisive blocker is the preregistered 7/15 state-support result.

## Decision

- Do not retrain the current MLP.
- Do not switch to the action-conditioned model yet.
- Do not run closed-loop E05.
- Collect more complete same-task episodes targeted at the E10 trajectory and
  early E15 action/joint-velocity ranges, while E05/E10/E15 remain test-only.

## Artifact identity

- Result file/payload:
  `89da1c0018db80ba35460368acddc5ae69e8400a1eb5ee0b99edc91f4b58e858` /
  `1d2c30931c356d10aed6c38fb6f93f2c3de86b245f531ac86aa1f98886926163`.
- Validation:
  `ee4c3144da62387dbd07da6c597d34800ab2e489aa822f6d9a8fde7ff2f72ee3`.
- Additional dataset/collection result:
  `dcbc724b09cbde5631e8be431fcf54b008dd01e6a295ec9123551cc8a8a1bfe8` /
  `1164c11a1d5957e2ca3f9017c11216d507ab499dfef2e17f4df0146f65de5ca7`.
- Expanded dataset/oracle:
  `b4590b5aa74bcbd626e3dbe19fe816899d89aaa106ddb7699c1be666a0b54fb6` /
  `b6e8d64a39ed91f1aef55b351269a23c8256de76b987414e4040642d94ab53cb`.

Remote immutable root:
`/mnt/data/quanth/experiments/vlsa-distal-same-task-boundary-expansion/same-task-boundary-expansion-20260810a`.
