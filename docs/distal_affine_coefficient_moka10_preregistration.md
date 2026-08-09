# Multi-state affine-coefficient feasibility diagnostic

## Question

Can a state-conditioned neural model predict the quantities required by the
seven-row two-step safety QP—nominal margin, action coefficient, and a
nonnegative one-sided error—on unseen complete episode groups?

This replaces the failed absolute-margin field from job `37508`; it does not
reinterpret that immutable result. It is not a closed-loop or formal-safety
claim.

## No-training target gate

The same ten immutable Table-1 episodes and complete task-group split are
used. E05/E10/E15 remain test-only. For each episode, register the first
two-step exact-proxy crossing and the four preceding states. At every state,
execute a fixed `5 x 5 x 5` first-action grid inside the existing
`L_inf <= 0.5` trust region while retaining the immutable second AEGIS action.
This produces 50 distinct states and 6,250 exact cloned-OSC grid labels.

For every state and each of the seven L5--L7 rows, fit the existing
candidate-conditioned minimum-L1 affine lower envelope with 1 micrometre
one-sided padding. Store `b = exact nominal margin`, `a = affine gradient`,
and `e = b - affine intercept`.

The target gate passes only if every registered state has an exact
proxy/raw-safe grid candidate, seven valid lower envelopes, a valid seven-row
QP, and a fresh exact two-step QP rollout with all seven distal rows and the
unchanged AEGIS EE proxy safe, zero protected raw contact, and at most 0.1 mm
obstacle motion. Failed and infeasible states remain in the artifact. No
training occurs unless an independent validator passes all 50 states.

## Conditional learned gate

After authorization, a shared geometry-factorized MLP predicts `b`, the three
components of `a`, and nonnegative `e`. Calibration uses only the complete
validation episode group and subtracts the maximum validation affine
overprediction per row plus 1 mm. Test gates are:

- zero sampled false-safe candidate rows;
- mean held-out gradient cosine at least 0.8;
- a valid seven-row QP and fresh exact two-step all-eight proxy/raw-safe
  verification for every test state.

Closed-loop E05 is explicitly not authorized by this diagnostic.
