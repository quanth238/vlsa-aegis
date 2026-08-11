# Structured bounded-orientation ablation preregistration

## Question

Did the duplicated, independently standardized obstacle rotation matrices
prevent the flat and time-conditioned factorized OSC models from learning a
stable execution map?

Jobs `38092` and `38093` established a causal representation failure. A valid
180-degree obstacle rotation in one validation episode changed scalar entries
that had zero training variance. The `1e-6` standard-deviation floor mapped
them to magnitudes of `2e6`, and the same orientation was repeated through the
root transform and 15 primitive transforms. Jointly masking those 288 scalars
removed more than 99.999% of the impossible terminal error from both frozen
models. Masking is diagnostic, however, because it discards orientation.

## Frozen representation change

This experiment removes exactly these two scalar groups from the learned
execution-model input:

- root obstacle rotation matrix entries;
- repeated exact-primitive world rotation matrix entries.

It appends one bounded 6D representation formed by the first two columns of
the obstacle root rotation. The third column is determined by their cross
product. Known primitive-local transforms remain in the explicit geometry
backend and are not redundantly repeated in the execution-model input.

Presence bits remain raw `0/1`. The 6D orientation remains raw in `[-1,1]`.
Other nonconstant features use train-only mean and raw standard deviation;
train-constant features are centered and divided by one. Validation and test
rows never determine normalization.

The input changes from 2,124 to 1,842 dimensions: remove 288, append 6.

## Matched arms

Both registered architectures are retrained:

1. the flat `51 x 7` one-sided geometry model from job `38070`;
2. the shared time-conditioned decoder from job `38076`.

The 7,905 immutable cloned-OSC rollouts, 85 states, complete two-action inputs,
60/10/15 episode-grouped split, joint trajectories, ellipsoid traces,
finite-difference sensitivities, losses, five seeds, optimizers, schedules,
and early-stopping rules remain fixed. Only the representation and its
train-only normalization change. No new simulation labels are collected.

## Decision gate

The representation itself must keep the maximum validation magnitude at or
below 500 and both validation terminal joint RMSE values below 0.1 rad.

The flat candidate must achieve zero test false-safes, at least 90% exact-safe
recall, support in at least 12 eligible test states, near-boundary RMSE no
worse than 2.9382342 mm, and terminal joint RMSE no worse than 19.9738 mrad.

The time-conditioned candidate must have zero test false-safes and strictly
improve over job `38076` in test near-boundary RMSE, terminal joint RMSE, and
supported-state count.

A pass only authorizes evaluation on newly reserved unseen complete episodes.
It does not authorize calibration, candidate expansion, QP, Poisson/SDF, or
closed-loop control.

Config SHA-256:
`adf6a42bdf01d1eb4910c1d50275065b34a6df37583865922d2a186e14df241c`.
