# State-conditioned region-aware MLP preregistration

## Question

Can a state-conditioned neural model reproduce the validated zero-margin
27-region L5--L7 safety oracle on unseen robot states and, if it passes that
gate, prevent the primary E05 distal-link collision under receding QP
replanning while preserving native SafeLIBERO completion?

Job 37690 remains strict NO-GO under its original zero/+1 mm all-gates test.
Job 37701 only established that its zero-margin controller decisions are
stable under regional resampling. This experiment cannot reinterpret either
artifact and does not train on E05, E10, or E15.

## Learned object

The model does not regress the rejected minimum-L1 target. For each current
state, protected constraint, and one of the 27 fixed overlapping regions, its
input is the validated 53-dimensional state/relative-geometry feature with the
candidate-action slot set to the region anchor, plus the region's normalized
lower, upper, and center coordinates. A shared five-member MLP ensemble predicts
the affine quantity used by the QP:

\[
    \underline h_{r\ell}(u)
    = c_{r\ell}+g_{r\ell}^{\top}(u-u_r).
\]

Training uses complete train groups only. It combines regional affine-value
regression with a one-sided penalty whenever the learned plane exceeds an exact
cloned-OSC grid margin. Raw coefficient loss has lower weight because job 37701
showed coefficient identity is not the control decision.

For deployment, ensemble disagreement is converted into a conservative anchor
guard using the worst standard deviation over the eight region vertices.
Validation groups then supply a per-region/per-constraint maximum one-sided
calibration plus a fixed 1 mm padding. Test groups never affect training,
calibration, uncertainty, early stopping, or thresholds.

## Unseen-state gate

The immutable test population contains 15 states from complete E05, E10, and
E15 episodes. On the same 1,440 off-grid actions used by the oracle comparison,
the learned union must have zero false-safes, global accepted-set Jaccard at
least 0.90, statewise Jaccard at least 0.80, and retain safe support in all 15
states.

At every test state, 27 learned seven-row QPs are solved with their regional
bounds. The valid proposal closest to the nominal action is selected without
using simulator outcome. All 15 selections must exist, pass a fresh exact
two-action cloned-OSC L5--L7 rollout, and remain compatible with the unchanged
released-AEGIS end-effector proxy. Relative to the immutable oracle selection,
the learned action shift must be at most 0.10 at p95 and 0.25 maximum.

Every condition must pass before closed-loop E05 is allowed. A failed learned
gate is a scientific NO-GO and produces no closed-loop run.

## Conditional E05 experiment

The successful released-AEGIS action ledger is replayed exactly until the first
learned intervention. After that intervention, pi0.5-LIBERO replans from the
new observation while the released AEGIS EE filter is recomputed from the live
state. Before every executed action, the MLP predicts all regional rows, the 27
QPs are solved, and only the nearest valid action is executed. The first action
is executed and the complete procedure is recomputed at the next state.

The learned QPs activate only when the directly measured current L5--L7
ellipsoid clearance is at most 50 mm. This is the edge of the registered E05
state support (state 184 starts at 45.3 mm); above it, the unchanged released
AEGIS action is executed. Exact shadow measurement remains active in both
modes. This fixed activation prevents unsupported far-field extrapolation from
changing task motion before the learned boundary model is relevant.

A cloned two-action OSC rollout records exact substep margins but is strictly
measurement-only: it cannot select, reject, repair, stop, or provide fallback.
This keeps the learned model and QP as the online safety mechanism while still
providing exact simulation evidence.

The research question passes only with zero raw L5--L7 contact, nonnegative
exact distal substep margins for every executed transition, no paper CAR, and
native task success. Collision avoidance without task completion, stopping,
or task success with collision is NO-GO.
