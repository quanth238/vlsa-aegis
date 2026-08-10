# Complete-OSC action-conditioned margin preregistration

## Question

Did the previous action-conditioned MLP fail because its inputs omitted state
used by the operational-space controller (OSC), rather than because the
two-action ellipsoid-margin map is not learnable by this model class?

## Fixed population

The experiment preserves the existing 85 states, 125 residual actions per
state, and complete-episode 60/10/15 split. E05, E10, and E15 remain test-only.
The old margin labels are not reused. Each canonical settled state and archived
action prefix is reconstructed, and all 10,625 full two-action candidates are
replayed in cloned OSC.

## Ordered input-sufficiency gate

For each state, the artifact stores the complete simulator state, auxiliary
solver/control arrays, wrapper timing, controller goals/orientation state,
torques and gripper state, plus explicit q/qdot, current and goal EE poses,
active-obstacle pose, all seven robot ellipsoid transforms, and every exact
obstacle-primitive transform. Each candidate stores both complete 7D actions,
including translation, rotation, and gripper components.

Every identical snapshot/action pair is executed twice. Its seven float64
rollout-minimum margins must be bitwise equal, the canonical raw protected
MuJoCo contact receipt must be equal, and both next-state hashes must be equal.
One mismatch blocks all training. This tests whether the recorded input can
uniquely determine the registered target in the deterministic simulator.

## Prediction-only model gate

Only after all 10,625 paired replays pass, a five-member shared row-wise MLP
predicts the two-action rollout margin as the current margin plus a residual.
Its input is the complete state vector, both full actions, and the seven-way
constraint identity. Normalization uses training episodes only; early stopping
uses validation episodes only.

On untouched E05/E10/E15, pass requires all of:

- zero action-level ellipsoid-proxy false-safes;
- at least 50% exact-safe action recall;
- at least one accepted exact-safe action in all 15 states;
- near-boundary (absolute exact margin at most 5 mm) RMSE at most 2 mm;
- overall RMSE no worse than the prior 3.808 mm result.

No calibration, QP, pi0.5 feature, or closed-loop E05 execution is permitted.
A pass authorizes only a separately preregistered calibration and
physical-contact-verified QP experiment. A deterministic replay pass followed
by prediction failure means missing OSC state alone is not sufficient; the
model or data representation remains the blocker.

Frozen config SHA-256:
`ed3dc204b233f24ec60259c3a165e2006ceb04a68ee3d0ab2f618f595ddfa4c4`.
