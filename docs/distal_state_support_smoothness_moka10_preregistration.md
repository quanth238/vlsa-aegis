# Distal state-support and oracle-smoothness preregistration

## Question

Why did the state-conditioned regional MLP see a `10.18 sigma` test shift and
reject every unseen action? Before changing the model or collecting data, this
audit asks:

1. Which frozen input features create the shift?
2. Does every test state have training support comparable with support observed
   between different training episodes?
3. At the nearest training state, does the induced multi-region safety oracle
   preserve quantitative margins and zero-margin action decisions?

This gate performs no training, simulation, QP solve, or closed-loop execution.
It consumes only immutable job-`37580`, job-`37690`, and job-`37722` artifacts.

## Feature-shift audit

The exact 62-dimensional job-`37722` feature rows are standardized using only
the 30 training states. Features are ranked by test maximum and test p95
absolute z-score. A feature is reported as shifted when its maximum exceeds
`5.0`. Test values cannot set normalization or thresholds.

## State support

State context retains the frozen 53-dimensional pair representation but
removes the constraint one-hot and candidate-action slots, leaving 43 features
per protected constraint. Training mean and population standard deviation,
with a `1e-6` floor, define the metric.

For each training state, the reference neighbor is its nearest training state
from a different complete episode. The frozen support threshold is the p95 of
those 30 RMS standardized distances. Each validation/test state is compared
with its nearest training state. A test state is supported only when its RMS
distance is below the reference p95 and no context feature exceeds `5.0`
training standard deviations. Sufficient support requires all 15 test states.

## Oracle smoothness

Raw regional coefficient equality is intentionally not used because job
`37701` showed coefficient non-uniqueness can be decision-harmless. Each
27-region oracle is evaluated instead on the matched normalized 125-action
grid. For each action, its decision-level union margin is

\[
M(u)=\max_{r\ni u}\min_{i=1,\ldots,7}
\left[c_{ri}+g_{ri}^{\top}(u-u_r)\right].
\]

Two states are compared by the RMSE of `M` and the Jaccard overlap of
`M(u)>=0`. The p95 margin RMSE and p05 Jaccard among the cross-episode nearest
training pairs are frozen thresholds. All 15 test-to-nearest-training pairs
must pass both for the oracle to be classified as smooth over supported state
variation. Full regional-row RMSE is diagnostic.

## Decision

- Insufficient support: collect boundary states from additional complete
  episodes while preserving grouped splits.
- Supported but nonsmooth: collect denser local boundary states or reduce the
  regional trust region before learning.
- Supported and smooth, given the prior MLP failure: authorize a separate
  preregistration for an action-conditioned conservative safety-value model.

No outcome authorizes closed-loop E05. That remains blocked until a later
learned model recovers unseen safe support with zero false-safes and fresh exact
QP verification.

Config SHA-256:
`5dacdc8e5eb843db3dcb27ecd6747059fcd3cd785c10a54404d162a46fc340ec`.
