# Native MuJoCo L5--L7 geometry inventory preregistration

## Question

Can the immutable 85-state moka-pot population be labeled with quantitative
signed distances from the actual compiled MuJoCo L5--L7 collision geoms to the
active obstacle, before replacing the existing ellipsoid-proxy targets?

This is a discovery and measurement-authority gate. It trains no model, solves
no QP, runs no closed loop, and makes no safety or task-success claim.

## Frozen population

The 60/10/15 complete-episode split and state steps come from the immutable
expanded dataset. E05/E10/E15 remain test-only. The 17 source episodes and
their archived Table-1 action ledgers are hash-bound through the three existing
selection manifests.

## Physical rows

The experiment does not force MuJoCo geometry into the seven artificial slab
rows. At compile time it discovers every collision-eligible geom attached
directly to `robot0_link5`, `robot0_link6`, or `robot0_link7`. One physical row
is defined per discovered protected geom:

\[
m_p^{\mathrm{MJ}}=\min_{k,o}
\left[d_{\mathrm{MJ}}(G_p(q_k),G_o(q_k))-d_{\mathrm{safe}}\right],
\]

where obstacle geoms are collision-eligible compiled geoms in the active
moka-pot body lineage. The frozen discovery margin is
\(d_{\mathrm{safe}}=0\). Three link minima and the global minimum are reported
only as aggregates, not substituted for physical targets.

`mujoco.mj_geomDistance` is queried with a 1 m cutoff. Every value and witness
must be finite and strictly below the cutoff. Every raw protected contact must
map to a registered pair and agree with the direct signed-distance query within
10 micrometres. Archived E05 steps 187 and 188 supply an explicit contact
witness rather than relying on a vacuous no-contact audit.

The ordinary seven ellipsoid gaps are retained only as `D_opt` comparison
values. Native distances and raw contacts are `D_sim` evidence.
For receipt fidelity, comparison-only ellipsoid features reproduce the existing
collector's cohort-local placeholder rule: use the E05 row when that cohort
contains E05, otherwise use the cohort's first immutable manifest row.

## Initial-state rule

The two-action cloned-OSC trace includes \(k=0\) only for states whose native
initial margin is nonnegative. Any initially negative state is reported as a
separate recovery cohort; it cannot be relabeled safe by any action. The gate
requires all 15 held-out test states to belong to the prevention cohort.

## Gate

The inventory passes only if:

- every protected body has at least one registered physical group and pair;
- the protected-group semantic inventory is identical across all 17 episodes;
- every signed-distance query is finite and uncensored;
- every raw contact pair is registered and distance-consistent;
- at least one primary raw-contact witness is observed;
- all 15 test states have nonnegative native initial margins.

A pass authorizes only a new candidate-rollout target collection gate. Model
training, physical-row QPs, and closed-loop E05 remain blocked.

Config SHA-256:
`7b1265e7cab6bc7362cc21a1a76101999e2dd9492bf31b3f36228c0d637b4e8a`.
