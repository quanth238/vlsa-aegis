# Direct-horizon OSC displacement gate

## Question

Can an execution model predict every OSC joint displacement directly from the
initial complete controller state and the full two-action Cartesian chunk,
without the recursive drift observed in job `38240`, while remaining
conservative on newly reserved episode groups?

The model is

\[
\Delta\hat q_k=F_\theta(x_{\rm OSC},A,k),\qquad
\hat q_k=q_0+\Delta\hat q_k.
\]

`k=0` is an architectural zero. The decoder evaluates all other horizons
directly. It never predicts per-step increments and never uses a cumulative
sum.

## Frozen populations

- Training/validation: the existing complete 60/10 state groups only.
- Prior E05/E10/E15 groups: diagnostics only and excluded from the new
  prediction verdict.
- New reserved prediction episodes: E01/E11/E21/E31/E41. Five label-blind,
  initially nonnegative, two-action states are selected per episode by current
  clearance and a five-step spacing rule. Each state receives the unchanged 93
  candidate design, including XYZ, rotation, and gripper dimensions.
- Future intervention episodes: E02/E12/E22/E32/E42. Their immutable manifest
  is frozen before any new rollout labels and remains unopened unless both the
  prediction and residual-bound gates pass.

The reserved collector records 2,325 complete 51-state joint/ellipsoid traces,
raw contacts, complete controller snapshots, action hashes, and next-state
hashes. Five candidates per state are repeated exactly. A separate H100
validator freshly replays the nominal candidate at every state.

## Matched model

The state/action encoder has widths 256/256. A shared 128-wide decoder receives
the encoded state/action and a fixed nine-dimensional horizon encoding. It
outputs the seven joint displacements at that horizon. The input is the stable
1,842-dimensional structured complete-OSC representation and the output is
added to measured `q0`.

Training retains five fixed seeds and directly supervises:

1. all joint displacements at substeps 1--50;
2. central finite-difference action-to-joint sensitivity;
3. the signed safety-normal joint error through the validated local `dh/dq`.

The geometry term is symmetric. Both optimistic and conservative signed
normal errors receive the same Huber penalty, with fivefold weight within 5 mm
of the boundary. No one-sided optimism penalty, binary classifier, generic
surface loss, calibration, or QP is added.

## Comparisons and prediction gate

All three frozen models are evaluated on exactly the same newly reserved
states and actions through MuJoCo FK and the existing L5--L7 ellipsoid geometry:

- direct-horizon displacement (new);
- job `38070` flat one-sided factorized model (previous best conservative arm);
- job `38240` cumulative-increment model (drift comparator).

The new model must satisfy every condition on validation and the reserved
population:

- zero false-safe actions;
- exact joints through the frozen static ellipsoid evaluator must themselves
  have zero false-safes and at least 99% recall against recorded rollout
  margins, so geometry mismatch cannot be misattributed to the predictor;
- accepted exact-safe support in every recoverable state;
- reserved safe recall at least 0.90;
- reserved near-boundary RMSE at most 2.671122 mm;
- joint and safety sensitivity cosine each greater than 0.8, with magnitude
  error reported;
- random-action joint RMSE at most 12 mrad overall and 20.586673 mrad at the
  terminal step;
- RMSE horizon slope at most 0.5 mrad/substep, terminal/overall ratio at most
  2, exact `q0`, and strictly less drift than the matched cumulative model.

Failure of any condition is a strict NO-GO and stops the experiment before
calibration, QP, or closed loop.

## Conditional downstream gate

Only a complete prediction pass freezes the execution model. Out-of-fold
episode predictions may then fit a separate state/link/time/trust-radius
upper envelope for the optimistic affine residual. Separate held-out episodes
must calibrate that envelope with zero false-safes and a feasible exact-safe
action in every recoverable state. No QP is part of that calibration gate.

Only both passes can authorize trust-region sequential QP correction with
geometry/envelope recomputation and fresh cloned-OSC verification after every
iteration. Final generalization uses the still-unopened E02/E12/E22/E32/E42
episodes; E05 remains the diagnostic closed-loop collision case. Simulation
and training run only inside a verified single-H100 Slurm allocation.

## Frozen identities

- Direct config SHA-256:
  `722d9183294b2f85dbbd1f0a341585a891a7a729bb79ee535827f7a4aa4742f7`.
- Reserved prediction manifest SHA-256:
  `4ac030c4757c7a705cacf7ed227076324c23fb7d4ce8c6ae7615c6fa7f9e26e6`.
- Future untouched intervention manifest SHA-256:
  `609eddd0ac768246827a5f126d05799bf7f2e314dadcc0a5f94343bc6287739c`.
