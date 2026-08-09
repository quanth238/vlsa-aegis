# Action-188 boundary-capacity diagnostic

## Question

Can the existing seven-output execution-loss MLP represent the local L5--L7
safety boundary when it receives exact boundary labels and simulator-gradient
supervision, and can its seven-row QP produce an exactly verified safe action?

This is a post-oracle-selected, same-state capacity diagnostic. It does not
test unseen-state generalization, closed-loop task success, or formal safety.
Completed Table 1 artifacts remain read-only.

## Frozen experiment

The immutable E05 action ledger is replayed to action 188. The nominal action
starts proxy-safe but crosses the exact-box L5_part_1 boundary during one OSC
`env.step`. A deterministic 10-by-10-by-10 Cartesian grid provides 1,000
candidate actions inside the existing `L_inf <= 0.5` trust region and action
bounds.

The critical boundary band is `[-5, +5] mm`. The 32 closest safe and 32
closest unsafe interior grid actions become gradient anchors. Each anchor has
six cloned central-difference probes with epsilon `0.02 action`, giving 384
additional exact transitions and 1,384 records total. A gradient row is valid
only when the minimizing MuJoCo substep and exact obstacle primitive agree at
the anchor and both perturbations in all three dimensions.

Each base action and all its probes stay in one split. Boundary-safe and
boundary-unsafe groups use a deterministic 70/15/15 train/validation/test
split and each receives 50% of the margin-loss weight. Actions outside the
frozen +/-5 mm band remain in the dataset as diagnostic evidence but are
excluded from learning and calibration. This is the apparatus repair after
job 37205 showed that the sampled trust region contains no separate far-safe
stratum; no action, label, band, trust region, model, or gate was changed.

## Paired arms

Both arms use the frozen 33 inputs, two 128-unit SiLU layers, seven Softplus
execution-loss outputs, identical initialization, calibration, and bounded
iterative seven-row QP:

1. `boundary_margin`: weighted Huber margin supervision only.
2. `boundary_margin_gradient`: the same loss plus normalized simulator-gradient
   supervision.

The earlier validated job 37198 nominal-trajectory result is immutable
context, not retrained or reinterpreted.

## Ordered gates

Collection and validation run first with no training. Training is authorized
only if the grid contains at least 32 safe and 32 unsafe boundary candidates
and every split contains a witness-stable critical gradient anchor.

The same-state model gate requires:

- critical-row near-boundary test RMSE strictly below the current-clearance
  baseline;
- zero conservative false-safe test candidates across all seven rows;
- at least one valid critical test gradient and mean cosine similarity at
  least 0.8.

The projection gate additionally requires an activated, valid seven-row QP
and one exact cloned OSC transition with all seven exact proxy gaps
nonnegative, no raw L5--L7 contact, and negligible obstacle motion.

The research direction receives a local `GO` only if the
`boundary_margin_gradient` arm passes both gates. Only then may the next work
collect recoverable boundary states across tasks/episodes, use complete
episode splits, and finally test closed-loop E05 task completion.
