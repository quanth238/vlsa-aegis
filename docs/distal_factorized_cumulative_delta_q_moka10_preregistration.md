# Cumulative joint-displacement execution model preregistration

## Question

Can a model of the unchanged SafeLIBERO OSC predict safety-relevant joint
motion and Cartesian-action sensitivity without predicting a clearance or
collision label directly?

The model receives the bounded job-38095 complete controller-state
representation and the complete two-action Cartesian command. It predicts 50
joint increments whose cumulative sum defines

\[
\Delta\widehat q_k=\sum_{s=1}^k\delta\widehat q_s,
\qquad
\widehat q_0=q_0,
\qquad
\widehat q_k=q_0+\Delta\widehat q_k.
\]

This differs from the previous flat 51-by-7 output because the initial state
is exact and every future configuration is constrained to be the integral of
a shared time-conditioned motion field.

## Frozen mechanism population

The first gate reuses the immutable 17-episode, 85-state, 7,905-rollout
dataset. It preserves complete OSC inputs, full 2-by-7 actions, all 51 exact
joint states, seven L5--L7 clearance traces, grouped splits, and the job-38095
bounded orientation representation. E05/E10/E15 remain diagnostic only.

Training supervises cumulative joint displacement, per-substep joint
increments, 14-dimensional finite-difference action sensitivity, and the same
near-boundary/late-horizon one-sided local geometry error used previously.
The deployed model outputs only joints. MuJoCo FK and analytic ellipsoid
geometry remain the safety evaluator.

## Mechanism gate

Validation and diagnostic test must both have zero false-safe actions and
support every exact-safe-eligible state. Diagnostic test recall must be at
least 90%, boundary RMSE at most 2.671122 mm, terminal joint RMSE at most
20.586673 mrad, and both joint and safety sensitivity cosine at least 0.8.
Substep-zero joint error must be exactly zero.

Failure blocks all new collection, QP and closed loop. Passing authorizes only
collection and prediction on the reserved episodes below.

## Untouched final population

Episodes E01, E11, E21, E31 and E41 of the primary same task are reserved
before inspecting new geometry or candidate labels. If the mechanism gate
passes, select five noncontact states per episode by the registered smallest
nonnegative L5--L7 current margin rule with five-action spacing, and collect
the same 93 full-action candidates per state.

Final prediction requires zero false-safes, support in every recoverable
reserved state and sensitivity cosine above 0.8. Only that result can
authorize a separate trust-region sequential-QP experiment. Calibration,
binary classification, Poisson/SDF, QP and closed-loop execution are forbidden
in this gate.
