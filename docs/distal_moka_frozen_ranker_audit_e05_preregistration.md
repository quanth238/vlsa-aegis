# E05 Moka frozen ranker audit

## Question

Was the `7/8` safer-branch result from job `39312` evidence of a transferable
paired-ranking signal, or a small-sample accident?

## Frozen protocol

At the identical E05 action-182 state, load the immutable direction-
conditioned scalar checkpoint without training. Generate 64 new directions
from seed `2026081410`, disjoint from all model-development directions. At
radius `0.0125`, replay both branches through cloned OSC and compare the
model-selected sign with exact twenty-action L5--L7 margins.

The primary gate requires:

- branch accuracy at least `0.85`;
- exact one-sided binomial significance against 50% at `p < 0.05`;
- positive mean exact gain on the six nominally near-active L5 witnesses.

Report global and near-active L5 branch accuracy, wrong-direction rate,
selected exact clearance gain, and random-sign baselines.

## Best-of-N diagnostic

After selecting one sign per direction, rank the 64 branches by predicted
worst L5--L7 margin. For prefixes `N={4,8,16,32,64}`, report the selected
candidate's exact rank, percentile, and regret relative to the best exact
branch. At `N=64`, also report top-1 agreement and regret relative to the best
of all 128 physical branches. These ranking metrics are diagnostic, not an
unregistered extra gate.

No model changes, QP, corrected execution, or closed-loop control are
permitted. A pass supports only a one-state Best-of-N ranking mechanism; it
does not validate a continuous safety potential or generalization.
