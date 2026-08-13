# E05 contact-free prefix then live VLA replanning

## Question

Can the frozen Cartesian VLA absorb a bounded L5--L7 repulsive prefix once it
receives the changed physical observation, without forcing avoidance and task
rejoining into the same five commands?

## Frozen protocol

Replay immutable Table 1 AEGIS through action 181. Select the registered
`soft_free_5d` candidate from the independently validated controllability
artifact and execute only its actions 182--186. This prefix has no endpoint
preservation or task-return penalty. At action 187, query frozen pi0.5 with
registered query seed 37 and continue querying every five executed actions.
Apply only the unchanged released AEGIS end-effector QP. No further L5--L7
repulsion, search, rejection, or stopping is allowed.

The released AEGIS direction is reinitialized once at action 187 from the
changed end-effector proxy and immutable perceived obstacle. This avoids
transplanting a stale auxiliary direction from the unexecuted archived suffix;
the QP equations and parameters remain unchanged.

Raw MuJoCo contacts involving protected L5--L7 are the physical collision
authority. Paper CAR is evaluated independently. Conservative link ellipsoid
proxy and exact robot-ellipsoid/compiled-box overlap are diagnostics.

The primary pass requires native task completion, zero protected raw contact
after activation, and CAR at most 1 mm. The run continues after collision so
task recovery and safety remain separately observable. All internal MuJoCo
substeps are measured from action 182.

This is one E05 mechanism test, not a generalization or invariance claim. A
pass supports constructing a practical policy-conditioned recoverable set from
warning states whose bounded repulsive prefix followed by the frozen policy is
physically safe and task-completing.
