# Direct controller-rollout margin diagnostic

## Question

Does replacing the binary contact classifier with a differentiable quantitative
controller-rollout margin produce a useful action gradient on the primary E05
failure, rather than merely a direction that is no better than random?

This experiment reuses the independently validated two-step dataset from H100
job `37270`. The primary case is the recoverable state at action 185. Its first
Cartesian XYZ action is variable and action 186 remains the immutable released
AEGIS command. Every label is the minimum exact-box support gap for each of the
seven accepted L5--L7 ellipsoid parts over both complete OSC transitions and
all internal MuJoCo substeps. Raw contact and obstacle motion remain separate
simulator authority.

## Model and grouping

The model is shared across the seven constraint rows. It consumes the existing
factorized state, candidate action, controller goal, and relative robot--box
geometry features. Two 128-unit Softplus hidden layers feed a linear scalar
output, so the predicted signed margin may be positive or negative.

Training uses six complete episodes from two task groups. Calibration uses one
episode from a third task group. The three `goal-II-t0` episodes are test-only;
E05 never enters training or calibration. The loss combines clipped signed
margin Huber regression, witness-stable finite-difference gradient cosine, and
normalized gradient-magnitude error. Complete simulator states never cross
train, validation, and test.

## Decisive E05 test

At action 185, evaluate the gradient of the smallest predicted raw margin at
the immutable nominal first action. Execute one exact two-step rollout at
radius `0.1` along the normalized increasing-margin direction. Execute 256
uniform random feasible directions at the identical radius from the identical
controller/simulator state. Compare the exact change in

\[
\rho_{\mathrm{roll}}(A)=\min_{\ell=1,\ldots,7}m_\ell(A).
\]

The learned direction passes only when it increases the exact margin and no
more than five percent of matched random directions improve the margin by at
least as much, using the add-one empirical value.

Separately, use the seven calibrated predicted margins and Jacobians in the
existing bounded minimum-deviation QP. A pass requires a valid seven-row QP
and a fresh exact two-step rollout with every proxy margin nonnegative, zero
protected contact, and at most `0.1 mm` obstacle motion.

## Decision

`research_direction_go=true` requires all of the following:

- E05 active-boundary RMSE beats the current-clearance baseline;
- conservative false-safe count is zero over every held-out episode;
- E05 active-gradient cosine mean is at least `0.8`;
- the learned E05 direction beats matched random directions;
- the calibrated seven-row QP passes fresh exact simulator verification.

This is a grouped, single-state mechanism diagnostic. It does not run
closed-loop E05, claim task completion, establish population efficacy, or
provide a formal safety certificate.
