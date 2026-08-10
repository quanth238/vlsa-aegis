# Factorized OSC execution pilot preregistration

## Question

Can a state-conditioned model learn the short-horizon execution map of the
unchanged black-box OSC more reliably than a model that directly regresses the
nonsmooth rollout-minimum safety value?

This is a two-action, same-task mechanism pilot for the registered L5--L7
case. It is not a Poisson-field contribution, whole-body result, CBF
invariance claim, QP experiment, or closed-loop task result.

## Motivation and frozen predecessor evidence

The complete-input collection from job `37910` established that all 10,625
snapshot/action pairs were deterministic. The matched job `37925` nevertheless
gave 72 test false-safes for the compact 56D minimum-margin MLP and 271 for the
complete-input arm; both failed their training-boundary fit. Job `37933` then
proved that the old 56D input is insufficient under controlled interventions:
changing rotation while preserving the old input produced 83 proxy safety
crossings and 41 raw-contact crossings.

The prior 125-action dataset cannot test the proposed factorization. It stores
only rollout minima and varies only the first XYZ translation. This pilot
therefore recollects complete joint traces and varies translation, rotation,
and gripper commands in both actions.

## Immutable split and action population

The 85 saved states and complete 60/10/15 episode split remain unchanged.
E05, E10, and E15 are test-only. At every state, the frozen nominal two-action
chunk is expanded to 93 candidates:

- one nominal chunk;
- clipped negative and positive probes for each of the 14 coordinates of the
  two complete 7D actions;
- 32 deterministic antithetic local pairs in the same 14D action space.

Translation and rotation probe steps are `0.05` normalized action; gripper
steps are `0.25`. Random radii are `0.10`, `0.10`, and `0.25`, respectively.
All commands remain in `[-1,1]`. Five registered candidates per state are
rolled twice and must reproduce joint traces, ellipsoid traces, contacts, and
both next-state hashes bitwise. This gives 7,905 primary and 425 repeat cloned
two-action OSC rollouts.

Each action trace contains the initial state, all 25 internal MuJoCo states of
the first OSC step, and all 25 new internal states of the second step: 51
seven-joint configurations. Seven ellipsoid clearances, raw protected
contacts, obstacle motion, and next-state hashes are recorded with the trace.

For every state and all 14 action coordinates, the actual clipped pair defines

\[
G^{\mathrm{OSC}}_{k,d}
=
\frac{q_k(A_d^+)-q_k(A_d^-)}{A_d^+-A_d^-}.
\]

At an action bound this is an explicitly recorded one-sided secant using the
nominal point; no nominal denominator is assumed.

## Matched learning arms

Both arms receive exactly the same complete 2,110D simulator/controller state
and both complete 7D actions. They use the same episode split, MLP trunk
widths `[256,256,128]`, five seeds, optimizer, schedule, and candidate rows.

The direct arm predicts the seven rollout-minimum ellipsoid margins and is
supervised by the corresponding 14-coordinate margin secants.

The factorized arm predicts the complete 51-by-7 joint trace as a residual
from the initial joint state. It is supervised by both joint values and all 14
joint-trajectory secants. Safety is not predicted by this network. Predicted
joints are passed through MuJoCo forward kinematics, the unchanged L5--L7
ellipsoids, and the existing exact-box support-gap evaluator with the obstacle
fixed at its observed initial pose.

The random antithetic candidates are the primary held-out action subset. The
axis probes are used separately for sensitivity evaluation. Raw MuJoCo contact
is reported as `D_sim`; ellipsoid support gap remains `D_opt`.

## Frozen decision gate

A factorization GO requires all of the following on untouched test episodes:

- exact-joint/fixed-initial-obstacle ellipsoid recomposition has zero
  false-safes and at least 95 percent safe recall, separating geometry
  composition error from execution prediction error;
- joint-trajectory RMSE at most `0.01 rad`;
- link-ellipsoid center RMSE at most `5 mm`;
- mean joint sensitivity cosine at least `0.8`;
- mean action-space ellipsoid-margin sensitivity cosine at least `0.8`;
- at most 36 factorized false-safe random actions;
- strictly fewer factorized false-safes than both the fresh matched direct arm
  and the previous best direct result of 72;
- exact-safe recall at least 50 percent and accepted safe support in 15/15
  test states;
- near-boundary factorized margin RMSE at most `3.058 mm`.

Passing authorizes a separate geometry-pipeline pilot only. It does not
authorize calibration, a QP, or closed-loop E05. Failure localizes whether the
remaining blocker is joint execution prediction, its action sensitivity, or
the explicit geometry composition.

## Exclusions

Poisson/SDF fields, conservative calibration, QP correction, VLA inference,
closed-loop E05, and any whole-body or invariance claim are forbidden. The
existing ellipsoid/contact evidence is preserved and completed Table 1
artifacts remain read-only.

Frozen config SHA-256:
`1fde25dc94f10bb4c4af91c35782df067172cdad5a0e6b6b42ae8b72b0ac4088`.
