# Distal-only 8 mm exact-box closed-loop E05 protocol

This is the user-authorized follow-up to H100 job `37183`.  It tests whether
the warning margin can address the actual L5/L6 failure when the margin is not
allowed to trigger on the released end-effector proxy.

## Frozen comparison

- Case: `vlsa-t1-goal-ii-t0-e05` from the immutable Table-1 manifest.
- Nominal controller: live `pi05_libero`, the registered policy-noise schedule,
  five executed actions per query, and the released AEGIS end-effector QP.
- Robot geometry: accepted v4 contiguous slabs: three L5, two L6, two L7,
  plus the unchanged released AEGIS EE ellipsoid.
- Obstacle oracle: the 15 exact live MuJoCo moka-pot collision boxes previously
  identified and hash-bound by job `37163`.  This is privileged simulation
  geometry and is not a deployable perception result.
- Constraint targets, in order:

  `L5_0..2, L6_0..1, L7_0..1, released_EE = [0.008 m x 7, 0.0 m]`.

  Thus the warning margin changes only L5--L7.  The released AEGIS EE filter
  remains in the nominal controller, and the exact-box verification row adds
  no EE warning margin.

## Transition and action selection

Every proposed action is executed in a synchronized cloned `OSC_POSE`
environment.  Clearance is the minimum across the interval-start state and
every internal MuJoCo state visited by the complete `env.step`, not only its
endpoint.  A candidate is acceptable only if all seven distal minima are at
least `8 mm`, the EE minimum is nonnegative, raw L5--L7 contact count is zero,
and maximum within-step obstacle displacement is at most `0.1 mm`.

The nominal action is executed unchanged when it passes those gates.  On a
predicted crossing, six central-difference probes construct the eight-row
local action model, the existing QP proposes the closest correction, and the
frozen local/global candidate set is exactly verified.  Selection first
minimizes deviation from the released AEGIS action, then maximizes the minimum
distal clearance.  If no verified candidate exists, the method reports the
failure and stops; it does not silently execute an unsafe action.

After each executed action, the main state must match the accepted clone to
`1e-10`.  Live policy inference continues in the released five-action
closed-loop schedule until the native task goal succeeds or the registered
goal-suite horizon is exhausted.

## Decision

`primary_problem_solved=true` requires all of the following in one run:

- native SafeLIBERO task success;
- no raw L5/L6/L7 contact;
- no paper CAR event (`<=1 mm` total obstacle displacement);
- every executed transition matches its accepted exact clone; and
- every accepted minimum-substep target vector satisfies
  `[8 mm x 7, 0 mm]`.

The outcome is a single-case privileged feasibility result.  It is neither a
population claim, deployable obstacle-perception method, nor formal whole-arm
safety certificate.  Completed Table-1 artifacts remain read-only.
