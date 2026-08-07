# Distal three-ellipsoid shadow preregistration

## Scope

Evaluate exactly three independent ellipsoids on primary case
`vlsa-t1-goal-ii-t0-e05`: one each for `robot0_link5`, `robot0_link6`, and
`robot0_link7`. Links 1 through 4 are excluded. The observer is read-only and
must preserve the archived AEGIS action ledger and completed Table-1 artifact.

## Geometry and constraints

For each protected rigid link, transform every vertex of every participating
compiled MuJoCo collision mesh into the world frame. Fit a Khachiyan
minimum-volume enclosing ellipsoid with tolerance `1e-4` and at most 20,000
iterations, then inflate all three semiaxes by the exact maximum Mahalanobis
radius plus `1e-9` relative numerical padding. Validation
must independently re-evaluate every recorded vertex and require maximum
normalized quadratic at most one and at least `0.999999`, proving both
containment and contact with the fitted surface.

Each link retains its own MuJoCo rigid-body point and angular Jacobians. The
released frozen obstacle MVEE is unchanged. One simultaneous OSQP receives
exactly three CBF rows plus the same seven joint-velocity bounds and metric as
E01. `D_opt=0.01 m` remains distinct from raw-contact `D_sim`.

## Required H100 evidence

- Allocation-side numerical tests must exercise the mesh fit and coupled QP.
- A direct SafeLIBERO render must visibly show only L5, L6, and L7 bounds.
- The primary rollout must preserve the archived initial state, policy noise,
  executed actions, physical link-5/link-6 contact, and native task success.
- Every action must record exactly three constraints, every infeasible QP must
  remain explicit, and setup/solve/total runtime must be reported.
- Validation must replay the immutable archived `env.step` inputs and match
  stepwise reward, done, end-effector pose, obstacle displacement, robot
  contact, native success at action 236, and first contact at action 187.

Passing this shadow gate does not authorize executing the proposed correction
or training a KKT/VI model.
