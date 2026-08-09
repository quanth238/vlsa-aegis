# Multi-state affine-coefficient result

## Verdict

**NO-GO for the current state-to-`(b,a,e)` MLP.** This is not an OSQP or
target-existence failure. All 50 exact coefficient targets exist, but the
learned mapping does not generalize to the held-out task group.

## Evidence

- H100 dataset job `37580`: 50 states, 6,250 exact cloned-OSC grid rollouts.
- Scoped target audit `37621`: all 50 seven-row L5--L7 targets valid; complete
  split remains 30 train / 5 validation / 15 test states.
- H100 learned job `37630`: 24,069-parameter shared MLP, best epoch 8,
  deterministic CPU training inside the H100 allocation.
- Held-out gradient cosine mean: `0.5818`, below the frozen `0.8` gate.
- Held-out false-safe candidates: `48`, all on E15, versus required zero.
- E05/E10 gradient cosine means: `0.4231/0.4231`; E15: `0.7404`.
- Nominal-margin RMSE on E05/E10/E15: `27.09/28.99/34.51 mm`.
- State-conditioned-error RMSE: `28.12/31.90/34.99 mm`.

Because the model gates failed, no learned QP or fresh learned-action
simulation was run. Therefore there is no collision-free or task-success
claim.

## Interpretation

The exact per-state affine representation remains feasible, and every held-out
state passed the privileged coefficient/QP oracle. The failure is learning a
stable mapping from state to those coefficients. The candidate-conditioned
minimum-L1 certificate is not a unique physical Jacobian: inactive rows often
receive zero coefficients, and the chosen affine region can switch when the
closest safe candidate changes. A single smooth MLP is therefore being asked
to regress a sparse, region-dependent optimizer selection.

Do not train this same model longer or run closed-loop E05. A defensible next
test must first make the target identifiable—for example, explicitly label
the active safe-set region and learn region-specific affine rows, or define a
canonical local physical derivative plus a separately calibrated one-sided
residual. It must repeat the zero-false-safe and held-out gradient gates before
simulation.
