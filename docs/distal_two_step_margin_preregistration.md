# Grouped two-step execution-margin feasibility diagnostic

## Research question

Can a learned two-step execution-margin model, trained from exact cloned OSC
rollouts, conservatively predict L5--L7 safety on unseen episode/task groups
and, when recomputed after every executed action, prevent the primary E05 L5
collision while retaining native task completion?

This is an outcome-conditioned ten-episode moka-pot diagnostic. It is not a
SafeLIBERO population estimate, a deployable perception result, whole-arm
safety, or a formal safety certificate.

## Why two steps

The accepted action-186 full-bound oracle found no one-step command that
preserved all seven L5--L7 proxy gaps. The accepted action-185 chunk oracle
found 459/731 safe two-step one-shot chunks. The new experiment therefore
varies only the first XYZ command, retains the second immutable AEGIS command,
and measures both sequential OSC transitions at every MuJoCo substep.

## Grouped data gate

The existing immutable ten-case selection is retained:

- train: six complete episodes from `goal-II-t2` and `spatial-I-t1`;
- validation/calibration: one complete `goal-II-t3` episode;
- test: the complete selected `goal-II-t0` group, including E05.

E05 is test-only and cannot enter fitting, early stopping, normalization, or
calibration. For each episode, the collector finds the first nominal two-step
proxy crossing in the registered pre-contact window. It tries that state and
at most three preceding states, selecting the latest state whose fixed
8-by-8-by-8 first-action grid contains balanced safe and unsafe examples
within 5 mm of the boundary.

Each eligible episode supplies 512 two-step chunks and eight balanced gradient
anchors. Six central finite differences per anchor provide witness-audited
gradients with respect to the first XYZ command. A separate validator must
accept every selected episode before any training begins. Failed episodes are
reported and never dropped.

## Paired learned arms

Both arms predict the nonnegative loss from current clearance to the minimum
clearance over both complete OSC transitions and use identical labels, splits,
losses, calibration, and hidden widths.

1. `global`: one monolithic feature vector and seven-output MLP, extending the
   previous execution-margin representation with the second nominal action.
2. `factorized`: one shared scalar MLP applied separately to each of the seven
   constraints. Its inputs include the closest exact obstacle box expressed in
   the robot ellipsoid frame, relative orientation and dimensions, joint and
   controller state, constraint identity, and both actions.

Training uses Huber margin loss plus finite-difference gradient supervision.
Samples are weighted equally by training episode and boundary sign. Per-row
conservative calibration uses only the validation episode.

## Ordered decision

Each held-out episode must have:

- active-boundary RMSE below the current-clearance baseline;
- zero calibrated conservative false-safe candidates;
- at least one witness-stable active gradient with mean cosine at least 0.8;
- a valid seven-row QP whose projected two-step chunk is exactly proxy-safe,
  raw-contact-free, and below 0.1 mm obstacle motion per step.

Only if the factorized arm passes every held-out episode is closed-loop E05
executed. The safety filter uses the immutable successful AEGIS action sequence
as the paired nominal plan, recomputes the learned two-step projection after
each executed action, and executes only the first correction. It does not run
an online cloned candidate oracle. Exact-box clearances, raw substep contacts,
CAR, and the native task goal are measurements only.

The research direction is validated by this diagnostic only if the grouped
factorized gate passes and the conditional E05 rollout has nonnegative exact
L5--L7 clearance, no protected contact, no CAR crossing, and native task
success.
