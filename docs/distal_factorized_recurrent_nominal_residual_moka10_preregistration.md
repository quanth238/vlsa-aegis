# Recurrent Nominal-Plus-Residual OSC Execution Gate

## Research question

Can a learned model of the fixed OSC provide a physically useful Cartesian
action-to-joint-motion derivative before it is used to steer a frozen VLA?

The model predicts execution, not collision class or task reward. Known FK and
the existing L5--L7 ellipsoid evaluator remain responsible for geometry.

## Matched comparison

The frozen direct-horizon normalized-secant model from job `38586` is the
baseline. The experimental arm uses the identical 7,905-action dataset,
grouped train/validation/diagnostic splits, five seeds, normalized paired
secant supervision, symmetric safety-normal supervision, and L5--L7 geometry.

The experimental temporal representation is:

\[
\Delta\hat q_k
=N_\theta(x,A_0,k)
+R_\theta(x,A_0,A-A_0,k)
-R_\theta(x,A_0,0,k).
\]

Both branches use GRUs over the two causal action phases, but every output is a
direct displacement from measured \(q_0\). Predicted increments are never
cumulatively integrated. The centered residual is exactly zero for the
nominal anchor, \(\Delta\hat q_0=0\), and the second Cartesian action cannot
affect substeps 1--25.

The checkpoint rule is deliberately aligned with the previous diagnosis:
among checkpoints satisfying the frozen validation joint-error limit, choose
the smallest train-scaled validation paired-secant error, followed by geometry
and joint error. This is a feasibility comparison, not a claim that recurrence
alone caused any improvement, because checkpoint selection differs from the
frozen baseline.

## Staged gate

Stage A uses only existing train and validation episodes. It measures complete
horizon and terminal joint error, L5--L7 near-boundary error, false-safes,
recoverable-state support, joint/safety sensitivity direction and magnitude,
and horizon drift.

It requires:

- zero validation false-safes;
- at least 90% safe-action recall and support in every recoverable validation
  state;
- aggregate and terminal joint and safety sensitivity cosine above `0.8` on
  both train and validation;
- median sensitivity gain in `[0.5, 1.5]` and mean relative gain error at most
  `0.5`;
- validation boundary RMSE at most `2.671122 mm`;
- validation joint RMSE at most `21.202037 mrad`, terminal RMSE at most
  `35.067896 mrad`, slope at most `0.5 mrad/substep`, and terminal/overall
  ratio at most `2.0`.

If Stage A fails, stop. The task-3 episodes remain unopened and matched-random
rollouts, flow guidance, calibration, QP, and closed loop remain forbidden.

Only a Stage-A pass permits Stage B on the already frozen, untouched
goal-II-task-3 E00/E05/E10/E15/E20 episodes. Stage B will require zero
false-safes, support in every recoverable state, useful sensitivity magnitude,
and a learned correction whose fresh exact OSC result ranks significantly
above 64 matched random directions at the same action radius. Passing Stage B
still authorizes only a separate frozen-VLA flow-guidance experiment.

## Scope limitation

The immutable current dataset contains nominal, coordinate finite-difference,
and antithetic local candidates. It does not contain actual VLA flow-noised or
model-guided chunks. Therefore Stage A tests local execution-gradient
feasibility only. Flow-distribution data cannot be added without breaking the
matched comparison and remains downstream of this gate.

Config SHA-256:
`0124715fe1962a38f2437f8ca2a1407e7ed35b47c5505913f2ca49426c57dc10`.
