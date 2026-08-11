# Fixed obstacle-orientation joint-mask confirmation

## Purpose

H100 job `38092` found that only validation episode
`vlsa-t1-goal-ii-t3-e35` explodes. Obstacle rotation-matrix entries that are
constant in training flip sign in this setup and become `2e6` z-scores. The
same physical orientation is encoded both at the obstacle root and in every
exact obstacle primitive. Neither single-group mask passed the registered 90%
causal threshold because the other duplicate remained.

This post-audit confirmation freezes that observation before execution. It is
not a model comparison or a proposed deployed input representation.

## Fixed masks and gates

The immutable dataset and both five-member ensembles are replayed exactly.
Four masks set selected normalized inputs to zero, equivalent to the training
mean:

1. exact-primitive rotation matrices only;
2. obstacle-root rotation matrix only;
3. their fixed union;
4. all inputs whose raw training standard deviation is below `1e-6`.

The duplicated-orientation explanation passes only if the union removes at
least 90% of E35 terminal-joint RMSE from both frozen models, leaves the other
validation episode below 0.1 rad, and increases test terminal RMSE by no more
than 10%. The fourth mask is diagnostic context and does not decide the gate.

A pass authorizes only a separately preregistered matched representation
ablation. It does not authorize direct training, simulation, calibration,
candidate expansion, QP, or closed loop. Configuration SHA-256:
`bc11384a0aeb35ab8ad609d47e460d0b3eea0efec991581e04f8ddf5bc616fce`.
