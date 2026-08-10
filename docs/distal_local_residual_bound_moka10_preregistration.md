# Distal local residual-bound preregistration

## Question

Can the already trained action-conditioned two-step margin MLP become
conservative on unseen E05/E10/E15 states by subtracting a state/action-local
upper bound on its dangerous error, while retaining a safe action in every
state?

This experiment does not alter the MLP, collect new rollout labels, or run
closed-loop E05. The prerequisite target-authority audit passed with zero
ellipsoid-safe/raw-contact cases over 13,025 rollouts.

## Frozen estimator

Twelve leave-one-complete-training-episode-out folds retrain the unchanged
five-member action-conditioned MLP. Each held episode supplies residuals

\[
r_\ell(z,a)=\hat m_\ell(z,a)-m_\ell^{\mathrm{OSC}}(z,a).
\]

The 52,500 out-of-fold residuals form seven constraint-specific nearest-
neighbor pools. Distance uses the 43 state/pair context features plus the three
normalized candidate coordinates. For a query, the bound is the maximum
dangerous residual among its fixed 256 nearest neighbors, clipped below at
zero, plus 1 mm. Neither validation nor test labels adapt this estimator.

\[
\underline m_\ell=\hat m_\ell-q_\ell-0.001\ \mathrm m.
\]

The existing ridge-Huber fitter converts these lower values on the fixed
125-action grid into seven rows for each of the 27 fixed overlapping regions.

## Immutable split and gates

The split remains 60 train / 10 validation / 15 test states by complete
episode. E05/E10/E15 are test-only and cannot enter training, residual pools,
normalization, thresholds, or model selection.

The gate requires on the 1,440 untouched off-grid test actions and 15 states:

- zero false-safe actions;
- exact-safe accepted support in 15/15 states;
- a valid selected regional QP in 15/15 states;
- fresh two-action cloned-OSC verification safe in 15/15 states.

The released AEGIS EE proxy is diagnostic only. If any gate fails, stop before
closed-loop. If all pass, preregister receding-QP E05 separately.

Config SHA-256:
`484f727ee642f7a138c7eb9bcef589d0ef0d13a5e22f4079c4c7be4e742ad33b`.
