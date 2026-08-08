# Execution-aware distal-margin residual network

Status: v2 neural pilot frozen after the validated v1 oracle-only outcome and
before any neural training outcome.

## Question

Can a small physics-structured neural model correct the inaccurate continuous
action-to-link-motion approximation well enough for a seven-constraint QP to
find an exactly verified safe Cartesian correction near the primary E05
L5/L6 failure?

This is a single-case mechanism experiment. It is not online-policy or
population efficacy, deployable perception, a whole-arm claim, or a formal
safety certificate.

It is also deliberately narrower than the proposed action-chunk method in
`idea_draft.md`.  The pilot filters one released Cartesian translation through
one complete `env.step`; it does not yet learn per-time-step chunk clearances
or modify the VLA denoising flow.  Those extensions are justified only if the
one-step representation, learnability, and exact-projection gates pass.

## Immutable state and geometry

The experiment replays the completed job-`37109` action ledger and collects
the 13 interval-start states at actions `180--192`. At each state it evaluates
the same deterministic candidate generator used by the oracle-affine audit.
Deduplication after action-bound clipping yields 85--87 unique XYZ actions per
state and 1,125 transitions in total; the exact per-state counts are frozen in
the protocol config. Rotation remains zero and the archived gripper command is
unchanged.

The robot representation is the accepted seven-part tight distal union:
three L5 slabs, two L6 slabs, and two L7 slabs. The obstacle representation is
the exact live union of 15 compiled MuJoCo moka-pot boxes with zero inflation.
Every candidate is executed through a complete cloned `OSC_POSE` `env.step`,
capturing the interval start and every internal MuJoCo step.

The optimizer label and simulator authority remain distinct:

- `D_opt`: seven minimum exact-box support gaps across the complete transition;
- `D_sim`: raw nonpositive L5--L7 contact and maximum within-step obstacle
  displacement.

The support gap is a differentiable optimizer proxy, not an exact signed
Euclidean mesh distance and not the physical safety authority.  The seven
primitive rows remain separate in the optimizer; taking a three-body or smooth
minimum is reporting/guidance only and cannot replace simultaneous hard rows.

The initial data job is an oracle-analysis gate and performs no neural
training. Validated H100 job `37195` passed contact geometry authority but
showed that action 191 is already outside the exact proxy safe set. It found
exactly one earlier recoverable crossing, action 188: all rows start positive,
the nominal transition makes `L5_part_1` negative, and 16 registered actions
inside `L_inf <= 0.5` are both proxy-safe and raw-safe. The v2 feasibility
pilot therefore selects action 188 post-oracle and reports that selection
explicitly; it is not a preregistered efficacy test.

## Physics-structured model

The model predicts the nonnegative clearance loss over one OSC transition:

\[
\widehat m_i^{\rm exec}(s,a)
=h_i(s)-\operatorname{softplus}(r_{\phi,i}(s,a)).
\]

This guarantees that the predicted minimum cannot exceed the measured
interval-start clearance. Inputs are the seven joint positions and velocities,
OSC position goal, obstacle position, seven current gaps, nominal XYZ, and
candidate XYZ. The network has two 128-unit SiLU layers and seven Softplus
outputs in millimetres.

This structural nonnegative residual is valid because the target is the
minimum over an interval that includes its start.  It must not be reused
unchanged for a future per-time-step clearance model, where clearance can
increase relative to the initial state.

State groups, rather than actions from the same state, are separated:

- train: `180--185`;
- validation: `186--187`;
- held-out test: `188--192`.

The oracle-selected projection state 188 is held out from both parameter
training and calibration.

Per-row conservatism is the maximum validation overprediction plus a fixed
`1 mm` padding. The model gate requires zero conservative false-safe
candidates on the held-out state groups and lower test RMSE than the
no-motion/current-clearance baseline.

## Neural multi-constraint projection

At action 188, the network supplies seven action gradients. A bounded QP
minimizes deviation from the released AEGIS XYZ command while requiring every
linearized conservative margin to be nonnegative. The warning threshold is
`8 mm`, but the hard transition target is zero; the earlier fixed-hard-8-mm
experiment showed that requiring 8 mm over the whole transition can be
infeasible. At most three linearization iterations are allowed and all actions
remain in `[-1,1]^3` and the fixed trust region.

A valid neural proposal must then pass a fresh exact cloned transition with:

- all seven minimum-substep exact-box gaps nonnegative;
- zero raw L5--L7 contacts;
- at most `0.1 mm` within-step obstacle displacement.

The ordered GO rule is: oracle-analysis gate, held-out model gate, valid neural
QP, and exact projected-transition verification. A GO authorizes a separately
preregistered closed-loop task test; it does not itself claim task completion.

A future smooth soft-min may provide a stable action-space guidance direction,
but `H >= 0` is sufficient only when every constituent prediction is a valid
lower bound.  The denoising-time inequality in the draft should therefore be
described as barrier-inspired flow guidance unless safe-set invariance and
terminal feasibility are separately proved.  The present pilot makes neither
claim.

## H100 runtime separation

Live preflight job `37192` established that the simulator environment's
PyTorch `1.11.0+cu113` cannot execute H100 `sm_90` kernels. This is retained as
an apparatus result. Job `37193` verified OpenPI PyTorch `2.7.1+cu126` on the
H100. Therefore MuJoCo collection and verification use the evaluation Python,
training uses the OpenPI Python on the same allocation, and weights cross the
runtime boundary in a version-neutral NumPy archive. Inference for exact
verification runs on CPU in the simulator process and is timed separately.

Completed Table 1 artifacts remain read-only and unchanged.

## Frozen outcome

Validated H100 job `37198` returned NO-GO. The representation gate passed, but
the held-out residual model was less accurate than the no-motion baseline and
its conservative seven-row projection was infeasible at action 188. The
training split contained no unsafe row labels, exposing a boundary-coverage
failure rather than a lack of exact safe actions. This result does not support
action-chunk or denoising-flow claims in `idea_draft.md`.
