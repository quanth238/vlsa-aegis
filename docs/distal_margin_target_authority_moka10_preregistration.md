# Distal rollout-margin target-authority preregistration

## Question

Before fitting a state-conditioned error bound, do the quantitative seven-row
ellipsoid rollout margins ever declare an action safe while raw MuJoCo reports
protected L5--L7 contact during the same two-action cloned-OSC rollout?

## Immutable audit population

The audit reads already completed exact artifacts and runs no new simulation:

- 85 states x 125 grid actions = 10,625 rollouts;
- 10 validation plus 15 test states x 96 off-grid actions = 2,400 rollouts;
- total: 13,025 two-action rollouts, including every internal MuJoCo substep.

No state or action can be dropped. E05/E10/E15 labels are used only to audit
geometry-target authority; they cannot select a learned model or threshold.

## Definitions and gate

An action is ellipsoid-safe only when every one of the seven L5--L7 minimum
substep margins is nonnegative. It is raw-distal-safe only when MuJoCo records
zero protected L5--L7 contacts and `D_sim_raw_safe` is true.

The target-authority gate passes only with zero actions satisfying

\[
\min_\ell m_\ell^{\mathrm{ellipsoid}}\geq0
\quad\land\quad
\mathrm{raw\ distal\ contact}.
\]

The released EE proxy, CAR obstacle displacement, and task success are separate
properties and are excluded from this geometry-target gate.

A pass authorizes only a separate preregistration of leave-one-episode-out
state-conditioned residual calibration. It does not authorize QP or closed-loop
E05.

Config SHA-256:
`9aa8f5f2e4942d79f248dbec16d899718432d931691cf47b44be098601730747`.

