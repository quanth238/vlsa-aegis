# E05 conservative obstacle-primitive oracle audit

## Question

Test whether replacing only the non-authoritative frozen AEGIS obstacle MVEE
with live conservative collision-geometry primitives repairs the action-192
false-safe mechanism.  The accepted seven L5--L7 slabs, released AEGIS
end-effector proxy, immutable state/action authority, candidate actions,
affine model, trust region, solver, raw-safety checks, and decision order remain
unchanged from the committed oracle-affine audit.

This is a single-state mechanism test.  It does not train a neural network,
run a policy population, or provide a safety certificate.

## Frozen source and pairing

- Case: `vlsa-t1-goal-ii-t0-e05`.
- State: immediately before job-`37109` action 192, reproduced by exact replay
  of its hash-bound executed actions 0--191.
- Nominal action: job-`37109` immutable executed action 192.
- Base audit config file SHA-256:
  `c7019c176e0e8b6379cdb1b83e09d7129c1237a4f28ec2f3daba49a7b57ddc95`.
- New obstacle config file SHA-256:
  `5e66f4113142ca57f4dbd070209fbb327860d0517226a38a3c31f7b66d8004ea`.
- Simulation: clean Slurm allocation with exactly one H100.

Completed Table-1 and job-`37109` artifacts remain read-only.  The released
AEGIS perception MVEE remains recorded as a diagnostic but is not used by the
new optimizer clearances.

## Conservative obstacle union

Select every MuJoCo geometry in the active obstacle body lineage whose
`contype` or `conaffinity` is nonzero.  Represent each selected compiled mesh
with one Khachiyan MVEE followed by exact farthest-vertex inflation and
`1e-9` relative numerical padding.  Because the ellipsoid is convex, containing
every compiled vertex contains the mesh's convex collision hull.  Supported
non-mesh collision primitives use their existing closed-form enclosing
ellipsoid and certificate.  Every source collision geom must appear exactly
once, and the total primitive count must not exceed 64.

Each fitted template is stored relative to its source geom and updated from
the live geom pose at the interval start and every internal MuJoCo step.  This
removes both contact-surface under-coverage and frozen-pose staleness from the
registered obstacle label.

For robot proxy `i` and obstacle primitives `j`, the retained scalar barrier is

\[
h_i(s)=\min_j h_{ij}(s).
\]

Thus the audit still has eight rows—seven accepted distal slabs and the
released end-effector proxy—but satisfying a row requires separation from
every obstacle primitive.  `D_opt` remains distinct from raw MuJoCo contact
and obstacle motion `D_sim`.

## Unchanged oracle audit

Use the same 87 deterministic actions, including the same 60 candidates in
the `L_infinity <= 0.5` trust region.  For every action, record the minimum of
each of the eight barriers over the interval start and all 25 internal MuJoCo
steps.  Fit the same nominal-anchored affine model, apply the same one-sided
registered-candidate error bound plus `1e-6 m`, solve the same bounded
eight-row OSQP, and exactly execute its proposal in the synchronized clone.

For every nominal raw contact, the geometry gate additionally requires the
contact point to be inside the primitive fitted to the exact contacted
obstacle geom, inside the corresponding robot slab union, and to have a
nonpositive minimum pair support gap.

## Decision and late-activation interpretation

The original ordered GO rule is unchanged: geometry authority, a locally
jointly raw/proxy-safe action, zero calibrated registered-candidate
false-safes, a valid QP, and an exactly raw/proxy-safe QP transition must all
pass.

A negative barrier at the immutable interval start is not an apparatus error
and must not be hidden by changing the target or omitting the start state.  It
means the conservative proxy would have needed to intervene earlier; the
registered action-192 audit then stops with no local one-step feasible action.
Any earlier-trigger follow-up must be separately preregistered after this
result.  No primitive, padding, candidate, margin, trust region, or solver
setting may be tuned after outcome inspection.
