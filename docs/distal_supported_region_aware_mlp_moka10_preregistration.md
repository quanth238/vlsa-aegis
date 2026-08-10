# Supported-state region-aware MLP preregistration

## Question

After job `37771` establishes training support and smooth oracle behavior for
all 15 E05/E10/E15 states, can the unchanged state-conditioned, region-aware
coefficient-output MLP reproduce the validated regional safe set on those
unseen states?

This gate tests the model, regional QP, and fresh exact selected-action
rollouts. It does not run closed-loop E05.

## Immutable population

- Dataset/oracle: job `37771`, 85 states and 27 regions per state.
- Grouped split: 60 train, 10 validation, 15 test.
- Test-only episodes: E05, E10, E15.
- Test actions and margins never influence training, early stopping,
  calibration, or uncertainty tightening.
- All failures and states remain in their registered splits.

The five E30 validation states added after the original learned experiment do
not yet have off-grid calibration labels. Before training, collect exactly 96
deterministic uniform actions per E30 state using seed `20260810`, the fixed
action box, immutable second action, and exact two-step cloned OSC. These 480
rollouts are validation-only. They do not inspect or alter E05/E10/E15.

## Unchanged learned method

The following sections must byte-equivalently match the original region-aware
preregistration after removal of its closed-loop authorization flag:

- 62-dimensional state/constraint/region input;
- five-member MLP ensemble with widths 256/256/128 and SiLU;
- 1,800-epoch, patience-180 deterministic full-batch CPU training;
- coefficient, regional-value, and one-sided overestimate losses;
- two-standard-deviation vertex guard;
- validation-only per-region/per-row calibration plus 1 mm padding;
- fixed 27-region partition and seven-row regional QP;
- all learned acceptance, QP, exact-rollout, EE-compatibility, and
  selected-action-shift thresholds.

## Decision gates

On the unchanged 1,440 unseen off-grid test actions, require:

- zero false-safe actions;
- global oracle/learned accepted-set Jaccard at least 0.9;
- minimum per-state Jaccard at least 0.8;
- accepted true-safe support in 15/15 test states.

Only if these preliminary gates pass, solve a learned seven-row regional QP at
every test state and require:

- 15/15 valid selected QPs;
- 15/15 fresh exact-safe two-step rollouts;
- 15/15 compatibility with the separate released AEGIS EE proxy;
- selected-action difference from the exact regional oracle no more than 0.1
  L2 at p95 and 0.25 maximum.

A complete pass authorizes a separate receding-QP closed-loop E05
preregistration. A failure after validated 15/15 input support authorizes
replacing coefficient output with an action-conditioned conservative
safety-value model. Neither continuation runs in this gate.

## Identity

- Config SHA-256:
  `9253f93faaf9d86a08e3fcddcfeb10d632b446ff8dffccb223683d89cbeeb609`.
- Expanded dataset file/payload SHA-256:
  `98158b3ea85993945ddc5170747af8a14a929597becddcce92d484f343d95c80` /
  `629142dd9b7710b7f26f00d001fd5ee3781cc1a000fef82c9d9fca9aa79ab41b`.
- Expanded oracle file/payload SHA-256:
  `5ac8efe88fed934ade3ec36512aa2a0d870142bd91ae09923e7cae577aea77b7` /
  `0716aae9cba8ef1bed5ce3b0800733593bfe8c550c599da68e5106729a72aebe`.
