# Distal three-ellipsoid cloned-step multi-CBF preregistration

## Question

The accepted continuous XYZ multi-CBF result for
`vlsa-t1-goal-ii-t0-e05` satisfies all three analytic L5/L6/L7 rows but still
contacts the obstacle because its resolved-rate prediction disagrees with the
actual discrete OSC transition. This follow-up changes only the constraint
transition model: can three next-step ellipsoid-clearance constraints obtained
from short cloned SafeLIBERO steps prevent the protected-link contact and paper
CAR failure while retaining the native task?

## Frozen controller

The primary environment restores the same immutable Table-1 state and receives
the same 237 archived AEGIS actions as nominal inputs. A second SafeLIBERO
environment is constructed from the same BDDL, episode state, seeds, OSC
configuration, and 20 settling actions. Before every probe step, the complete
MuJoCo state, solver warm-start arrays, controls, environment clock, OSC goal
state, update flag, robot torques, and stateful Panda gripper command are copied
from the primary environment. OSC interpolation must be disabled.

For nominal XYZ `u0`, the probe executes `u0` and central perturbations of
`+/-0.05` in each clipped XYZ dimension. The three buffered next-step
clearances form `h_plus(u0)`, and the six perturbed steps form a `3 x 3`
finite-difference matrix `G`. One simultaneous QP solves

```text
minimize     0.5 * (u - u0)' * I * (u - u0)
subject to   h_plus_i(u0) + G_i * (u - u0) >= 0, i in {L5,L6,L7}
             -1 <= u_j <= 1
```

Here zero is the boundary of the already buffered support gap, so the physical
optimizer clearance remains the preregistered `D_opt=0.01 m`. Rotation remains
zero and the archived gripper command is preserved.

The exact QP candidate is then executed in the clone. If any verified
next-step buffered clearance is below `-1e-6 m`, candidates at fixed scales
`0.5, 0.25, 0.125, 0` toward the predefined stop action are tried in that
order. The first verified-safe candidate is executed in the primary
environment. If no candidate passes, the method terminates without executing
an unverified action. After the primary `env.step`, its complete next state and
three clearances must match the accepted clone within `1e-10`; otherwise the
run is an apparatus failure. The nominal transition is repeated at step zero
as an additional determinism check.

## Frozen interpretation

The continuous multi-CBF result remains an immutable negative comparison; its
gains, ellipsoids, and QP are not retuned. The new experiment passes only if
all executed steps use exactly three discrete constraints, every action passes
the cloned-step verification, no protected-link contact occurs, paper CAR
passes, and the native task succeeds in the same 237-action horizon. A failure
or infeasible QP is retained. This is a one-case deterministic simulator-oracle
test, not a real-time method, population result, or neural-training target.
