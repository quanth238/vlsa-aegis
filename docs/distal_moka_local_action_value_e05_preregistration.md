# E05 Moka local nonlinear prediction gate

## Question

At the identical E05 action-182 state, can the nonlinear dependence of
twenty-action L5--L7 safety on a five-action XYZ perturbation be predicted on
fresh directions after the full affine-row representation failed?

## Matched arms

Both arms use the compact 25D witness input and add exactly one 15D query,
giving 40 inputs and the same two-layer 128-unit scalar MLP.

1. **Direction-conditioned scalar response:** input a unit perturbation
   direction and predict its central-secant safety response for one link/time
   witness.
2. **Nonlinear local action value:** input the signed action delta and predict
   the absolute safety margin for one link/time witness. Its response is the
   difference between the predicted positive and negative values.

The exact 32 directions and radius `0.0125` from validated job `39305` are
replayed. Directions 0--19 train, 20--23 select checkpoints, and untouched
directions 24--31 test both models. The state, nominal actions, geometry,
twenty-action horizon, seed, optimizer, and architecture are identical.

## Gate

A model passes only if untouched directions achieve:

- response cosine at least `0.8`;
- response sign accuracy at least `0.85`;
- the same cosine/sign thresholds on the six nominally near-active rows;
- at least `7/8` correct choices between the positive and negative branch;
- worst-margin RMSE at most `1 mm`.

This is a one-state representation/capacity test. It cannot demonstrate state
generalization, collision prevention, task preservation, or formal safety.
No QP is solved and no corrected action executes.
