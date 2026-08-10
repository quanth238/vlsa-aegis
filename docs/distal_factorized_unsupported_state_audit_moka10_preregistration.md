# One-sided factorized unsupported-state audit

The independently replayed job-38070 verdict remains NO-GO. This diagnostic
asks why the one-sided model has no accepted exact-safe random candidate in
three of the fifteen E05/E10/E15 states. Apparatus-only H100 selector
diagnostic `38074` established before geometry inspection that two states have
no exact-safe candidate in the registered region and one contains exact-safe
candidates that are all rejected. The audit retains all three so those causes
remain distinct.

For each discovered unsupported state, record:

- the number of exact-safe candidates already inside the registered 64-action
  region;
- the least-negative predicted worst margin and its exact/predicted active row
  and substep;
- baseline versus one-sided margins and joint-trajectory errors;
- exact FK/ellipsoid margins for all five one-sided ensemble members;
- distance to the nearest grouped training state that contains an exact-safe
  random action.

State distance uses only the complete 2,110D OSC state, not candidate action.
Normalization is fit on training states. The support threshold is the 95th
percentile of each exact-safe-supported training state's nearest
different-episode training neighbor, together with a 5-sigma maximum-feature
gate.

Trajectory error is called large only when the state's one-sided joint RMSE
exceeds the 95th percentile among the twelve supported diagnostic-test states.
A conditional bias correction is merely nominated—not fit—only if the best
rejected margin is within 5 mm of zero, training support passes, and trajectory
error is not large. If some but not all exact ensemble members accept an
exact-safe candidate, aggregation is implicated. Existing exact-safe actions
inside the region rule out candidate-region size as the immediate cause.

No training, new controller rollout, correction fitting, calibration, QP,
closed loop, Poisson/SDF, classifier, or verdict change is permitted. The
diagnostic E05/E10/E15 cases cannot be used for a final generalization claim.
