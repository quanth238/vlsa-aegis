# E05 exact multi-step action-chunk oracle

## Purpose

The action-186 full-bound diagnostic found no one-step action that preserved
all seven L5--L7 ellipsoid gaps. This pilot therefore starts from immutable
action 185, before the safe-set crossing, and asks whether a short correction
can keep the actual OSC trajectory safe through the crossing.

This is privileged single-state feasibility evidence. It is not neural
generalization and cannot authorize closed-loop E05 by itself.

## Paired arms

The immutable released-AEGIS chunks at horizons two and five are the paired
baselines. Four candidate families are frozen:

1. two-step one-shot: replace only action 185 translation;
2. two-step distributed: residuals `[delta, -delta]`;
3. five-step one-shot: replace only action 185 translation;
4. five-step distributed: residual weights
   `[1, 0.5, 0, -0.5, -1] * delta`.

One-shot bridge editing and distributed residual velocity remain distinct
arms. Distributed corrections are generated only inside their jointly
feasible action box, without clipping, and sum to zero in each Cartesian
dimension. Thus endpoint preservation is explicit. Normalized action
displacements use scale only; no normalization mean is subtracted.

The one-shot arm tests 729 full-bound first actions plus the exact nominal and
reverse action. The distributed arm tests a 729-point feasible-delta grid plus
the exact zero residual.

## Exact authority and gate

Each action chunk is run after one exact clone synchronization and then
advanced sequentially through the actual OSC and every internal MuJoCo
substep. Safety requires, throughout the entire chunk:

- all seven exact-box L5--L7 proxy gaps nonnegative;
- zero raw protected-link contact;
- no more than 0.1 mm obstacle motion per control step.

The smallest safe chunk over the tested families is executed once from the
same state, with every step required to match its independently cloned next
state exactly.

If no family contains a safe chunk, these registered low-dimensional chunk
parameterizations are rejected; that is not proof that every possible chunk
is infeasible. If a verified chunk exists, grouped chunk-boundary collection
is authorized. Neural training and closed-loop E05 remain later gates.
