# Factorized terminal-bias audit preregistration

## Question

Before another training run, determine whether the job-37980 factorized model's
L5/substep-50 optimistic clearance error is already present in grouped training
or validation episodes, or appears only in the repeatedly inspected
E05/E10/E15 diagnostic episodes.

The audit uses the immutable five-member model and the 64 out-of-fit random
actions at every one of the 85 states. It recomputes, with fixed-k0 obstacle
geometry, all seven exact-q and predicted-q ellipsoid clearances at all 51
substeps. The residual is

\[
r_{r,k}=\hat h_{r,k}-h^{\mathrm{exact}}_{r,k}.
\]

Positive residual is dangerous optimism. Results are reported for every link,
substep and split. A link's terminal bias is material only with at least 30
near-boundary samples, median optimism of at least 0.5 mm, and at least 60%
positive residuals.

## Geometry-signal gate

The audit computes central finite differences of the known FK plus ellipsoid
geometry with respect to all seven joints. On independent random candidates,
near-boundary clearance linearization and prediction-error linearization must
both have at most 0.5 mm RMSE. Signed clearance-delta cosine and sign agreement
must both be at least 0.9. Action-space gradients composed through the measured
joint motion must remain stable over 32 candidate resamples: median cosine at
least 0.95 and fifth percentile at least 0.8.

This exact numerical geometry signal supersedes the earlier 2 mm fitted
Jacobian permission. Failure blocks the proposed loss regardless of the bias
localization.

## Ensemble audit

Each of the five immutable ensemble members is evaluated through exact FK and
ellipsoid geometry on all false-safe random actions and same-state,
nearest-margin correctly classified controls. High disagreement requires both
a false-safe/control median standard-deviation ratio of at least 2 and AUROC
of at least 0.8. Otherwise the error is treated as systematic rather than
readily detectable by ensemble calibration.

## Decision

- Material train or validation terminal bias plus a valid geometry signal may
  authorize one matched one-sided-geometry loss ablation.
- Bias appearing only in diagnostic E05/E10/E15 indicates an unseen-state
  coverage/generalization problem; changing the loss is not authorized.
- Failure of the geometry signal blocks the ablation.

No training, new controller rollout labels, calibration, QP, closed loop,
Poisson/SDF, generic surface loss, or binary classifier is permitted in this
audit. E05/E10/E15 are diagnostic cases only. Any later generalization claim
requires newly reserved, complete unseen episodes grouped at episode level.
