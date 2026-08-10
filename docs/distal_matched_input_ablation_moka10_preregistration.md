# Matched old-56D versus complete-OSC input ablation

## Question

Did omitted OSC/controller inputs cause the action-conditioned rollout-margin
MLP to fail, or does the failure remain when the exact same fresh target rows
are paired with complete inputs?

## Frozen comparison

The immutable complete-OSC dataset contains 85 states, 10,625 two-action
candidates, and 74,375 seven-constraint rows. E05, E10, and E15 remain test
only. No simulator is run and no label is recollected.

Two arms use identical row identities, rollout-minimum ellipsoid targets,
current-margin residual targets, complete-episode splits, MLP widths, loss,
normalization rule, ensemble seeds, batches, early stopping, and schedule:

- `old56`: q/qdot, relative active proxy geometry, constraint identity,
  first-action XYZ, fixed second-action XYZ, and normalized candidate XYZ;
- `completeOSC`: the complete aligned simulator/controller snapshot, semantic
  transforms, both full 7D actions, and constraint identity.

The old features are reconstructed numerically from the fresh semantic state.
The reconstructed start margins must match the stored fresh margins to
`1e-10 m`. Both arms must have identical row/label hashes before training.

## Metrics and interpretation

Report train, validation, and test overall and near-boundary RMSE, action-level
false-safes, exact-safe recall, and state support. A fit requires at most
3.808 mm overall RMSE and 2 mm RMSE for rows within 5 mm of the boundary. The
full test gate additionally requires zero false-safe actions, at least 50%
safe recall, and safe support in 15/15 states.

- Complete passes while old56 fails: missing inputs were the root cause.
- Both fit training but both fail test: state/setup coverage is insufficient.
- Both fail training fit: the output target or plain-MLP representation is the
  problem.
- Any other outcome is mixed and remains inconclusive.

No Poisson field, calibration, QP, simulation, new data, or closed-loop E05 is
authorized by this experiment.
