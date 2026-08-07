# Distal three-ellipsoid active multi-CBF preregistration

## Question

Test primary report case `vlsa-t1-goal-ii-t0-e05`, where released AEGIS first
contacts the active obstacle with link 5 at action 187 while its end-effector
proxy barrier remains positive. The experiment asks whether adding the three
surface-fitted L5/L6/L7 barriers prevents protected-link contact and paper CAR
failure without losing native task success.

## Paired intervention

Restore the immutable Table-1 initial simulator state and replay its 237
released-AEGIS `env.step` inputs as the nominal action sequence. Before each
step, solve one simultaneous QP over the normalized translational OSC action
`v`:

```text
minimize     0.5 * (v - v_aegis)' * I * (v - v_aegis)
subject to   a_i' * B(q) * v >= -10 * h_i,  i in {L5, L6, L7}
             -1 <= v_j <= 1
```

Here `B(q)` is the damped resolved-rate map from the executable XYZ action to
seven estimated joint velocities, and each `a_i` is the analytical rigid-link
ellipsoid gradient mapped through the live MuJoCo Jacobian. The L5/L6/L7
ellipsoids are the accepted mesh-vertex MVEEs. The obstacle remains the
released AEGIS perception MVEE. `D_opt=0.01 m`; raw MuJoCo contact and active
obstacle displacement remain the separate `D_sim` authority.

Only XYZ may change. Rotation remains zero and the archived AEGIS gripper
command is preserved exactly. A failed or infeasible QP is retained as a
method failure; it is never passed through silently.

When the nominal action already satisfies all three inequalities, its unique
minimum-intervention optimum is the nominal action itself. The implementation
still times and postchecks OSQP, then executes the nominal XYZ exactly instead
of its sub-tolerance floating-point displacement. A material intervention is
registered above `1e-6` action L2.

## Frozen success definition

The primary problem is solved only if all of the following hold within the
same 237-action nominal horizon:

- no raw active-obstacle contact with `robot0_link5`, `robot0_link6`, or
  `robot0_link7`;
- active-obstacle maximum L1 displacement is at most 1 mm, so paper CAR
  passes;
- the native SafeLIBERO goal becomes satisfied; and
- every executed action used one valid QP with exactly three CBF rows.

This exact-action counterfactual isolates the new safety filter. It is not a
closed-loop policy claim and not population efficacy evidence.
