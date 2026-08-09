# Privileged receding affine-oracle E05 closed-loop preregistration

## Question

Can the exact two-step safe-set mechanism already validated at isolated E05,
E10, and E15 states prevent the primary E05 L5 collision when recomputed after
every executed action, while retaining the immutable successful released-AEGIS
nominal sequence and native SafeLIBERO task completion?

This is a simulator-oracle feasibility diagnostic. It is not a learned model,
deployable perception method, formal whole-body certificate, or population
claim. Neural training remains unauthorized.

## Frozen intervention

The run uses the primary `vlsa-t1-goal-ii-t0-e05` initial state and the 237
read-only released-AEGIS Table 1 `env.step` inputs. Before every action, a
fresh cloned MuJoCo environment executes the nominal first action and the next
immutable AEGIS action through the real OSC controller. The final action uses
a one-action horizon because no next nominal action exists.

The nominal action is executed unchanged only if all seven accepted L5--L7
slab margins and the unchanged released-AEGIS EE proxy remain nonnegative at
every internal MuJoCo substep, with zero raw protected contact and no more than
0.1 mm within-step obstacle motion.

If the nominal horizon is unsafe, the first normalized XYZ action is sampled
on the existing 8-by-8-by-8 grid inside the clipped L-infinity 0.5 trust
region. Orientation and gripper channels and the second AEGIS action stay
unchanged. Exact two-step labels generate seven candidate-conditioned affine
lower envelopes with 1 micrometre one-sided padding. The ordinary
minimum-deviation QP receives exactly seven L5--L7 rows. The released-AEGIS EE
proxy is an exact outcome veto, not an undisclosed eighth QP row.

The QP candidate receives a fresh exact cloned rollout. The controller then
executes only its first action and replans from the resulting state. The
executed transition is measured at every internal substep and its next-state
hash must exactly equal the cloned first-transition hash. No grid fallback is
executed if the certificate, QP, or fresh exact verification fails.

## Evidence and decision

The H100 run records distinct QP time and 512-candidate simulator-oracle time,
all action decisions, exact and executed all-eight margins, raw L5--L7 contact,
paper CAR, native BDDL goal progress, clone-state receipts, and an MP4/JPG of
the actual executed environment. Completed Table 1 artifacts remain read-only.

`privileged_closed_loop_e05_go=true` requires:

- every executed L5--L7 and released-AEGIS EE proxy margin nonnegative at
  every substep;
- zero raw L5--L7 contact;
- no active-obstacle displacement above the paper's 1 mm CAR threshold;
- every executed next state exactly matching the verified clone; and
- native SafeLIBERO task completion.

A GO shows that the privileged receding safe-set target can solve E05 and
therefore provides a meaningful target for later state-conditioned learning.
A NO-GO rejects only this registered local-grid, single-affine, two-step
oracle; it does not prove that longer-horizon or nonlinear interventions are
impossible.

Config SHA-256 is
`0a82ee6c6250996e10d2a4c30a0bb48927d08240b298a0fb5b21a438c855186b`.
The prerequisite is independently validated H100 representation job `37280`.
