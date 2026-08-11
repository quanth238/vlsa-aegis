# Frozen Explicit-J Ensemble-Member Audit

Validated job `38673` recovered action-response magnitude but learned poor
directions. This read-only audit determines whether that failure occurs inside
each ensemble member or is created by averaging heterogeneous members.

The audit reloads the immutable five-member model and existing 7,905-action
dataset. It performs no training and no simulator rollout. For every member it
reports joint sensitivity on train, validation, and the already-used
diagnostic split:

- aggregate and terminal direction/magnitude;
- translation, rotation, and gripper families;
- all 14 action coordinates at all 51 horizons;
- joint-trajectory RMSE;
- frozen checkpoint epoch.

It also reports all ten pairwise member cosines and

\[
c=\frac{\|\frac{1}{M}\sum_m J_m\|}
        {\frac{1}{M}\sum_m\|J_m\|},
\]

which measures cancellation introduced by ensemble averaging.

The diagnosis is fixed before outcome inspection:

- at most one member passes train and validation: member-level optimization
  or checkpoint failure;
- at least four members pass but validation \(c<0.5\): ensemble cancellation;
- otherwise: heterogeneous member and ensemble failure.

Checkpoint epoch is descriptive only and cannot establish causality. Training,
simulation, new labels, unopened episodes, calibration, QP, closed loop,
Poisson/SDF, and classifiers are forbidden. The output may authorize only a
separately preregistered next experiment.

Config SHA-256:
`40a77cb25e1ef18ae0a80dc2eaec2f19dd4ea8bb8128132ef148ad4090b8733e`.

## Completed result

Independently validated H100 job `38696` classified the frozen model as
`member_level_optimization_or_checkpoint_failure`: zero of five members passed
the registered train-and-validation gates. See
`docs/distal_factorized_explicit_jacobian_member_audit_moka10_result.md`.
