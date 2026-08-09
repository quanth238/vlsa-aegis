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

## Frozen outcome

Clean H100 dataset job `37209` completed on `worker-1` and its independent
validator passed. The 1,000 grid actions contained 365 proxy-safe and 635
proxy-unsafe critical-row labels; every action was raw-contact-safe for this
one cloned step. All 64 finite-difference anchors were witness-stable, split
41/11/12 across train/validation/test. The dataset/result/validation file
SHA-256 values are `ddc8699ee59b4278fdbd39a0b302f7100b7801dc5d5e277a7bf4860cf8f3f73f`,
`0226abaa4035a13a62a9ce00cf78e7dba579e080824e4b7a4542102324fe75f7`,
and `bb30ee30799dd08b0f6266129f4a663be5436d0e0fcb11b1c1f312bb22e76f51`.

Clean paired H100 job `37210` then passed independent validation. On 224
held-out boundary records, critical RMSE was `0.01396 mm` for margin-only and
`0.01440 mm` for margin-plus-gradient, versus the frozen current-clearance
baseline at `3.70047 mm`. Both had zero conservative false-safe candidates.
Critical gradient cosine was `0.99987` and `0.99997`, respectively.

Both seven-row QPs solved. The gradient arm changed XYZ from
`[0.026957, 0.309338, -0.712885]` to
`[-0.090338, 0.408238, -0.217085]`. Its exact cloned transition had minimum
L5_part_1 clearance `1.391 mm`, all seven distal rows nonnegative, zero raw
distal contacts, and zero obstacle displacement. Its QP wall time was
`1.344 ms` and complete two-linearization projection time was `5.549 ms`.
The margin-only arm also passed exact verification with L5_part_1 at
`1.330 mm`.

The result is a local capacity `GO`: boundary-focused exact physics labels
repair the flat-gradient failure of job `37198`, and the learned surrogate can
drive the existing seven-row QP to a verified safe action. It does not show
unseen-state or task generalization. Margin-only already performs almost
identically, so this experiment does not establish that explicit gradient
supervision is necessary or better. Result/validation SHA-256 values are
`51809a270d68c3268655f2e8fbcbc7e06bd515d73e34746181c5ac3a2fba154b`
and `2681c64c5575fe2b83df0fe8439eeaa844bde7eb8e6b260d038f65bf62f810ef`.
