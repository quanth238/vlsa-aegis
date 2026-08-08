# E05 substep geometry and oracle-affine mechanism test

## Question

Before training a physics neural network, determine whether the proposed model
class can correct the primary AEGIS false-safe mechanism.  The test gives the
network oracle access to the local action-to-minimum-substep barrier map.  It
does not train a model and does not claim policy or population efficacy.

The test is intentionally ordered:

1. verify that the current proxy geometry represents the raw contact;
2. verify that a physically and proxy-safe Cartesian action exists locally;
3. fit the best nominal-anchored affine model available from the registered
   cloned candidates;
4. solve the ordinary eight-constraint QP with a one-sided conservative error
   offset and execute its proposal through the exact OSC transition.

Failure of an earlier item stops the research interpretation of later items.

## Frozen state and action authority

- Case: `vlsa-t1-goal-ii-t0-e05` only.
- State: the deterministic state immediately before action 192 in completed
  H100 job `37109`, after exact replay of its immutable executed SITL actions
  0--191.
- Nominal action: immutable `executed_sitl_action` at job `37109` action 192.
- False-safe witness: all eight endpoint proxy clearances are positive at that
  action (minimum `0.003071851565 m`), while raw MuJoCo records direct
  `robot0_link6_collision` contact at `-0.000523052603 m` and paper-CAR
  displacement of `0.001438559285 m`.
- Protected proxies: the accepted seven L5--L7 slab ellipsoids and unchanged
  released AEGIS end-effector proxy.
- Obstacle proxy: unchanged released AEGIS perception MVEE.
- Raw authority: direct MuJoCo L5/L6/L7 contact and maximum obstacle motion at
  every internal physics substep remain distinct from proxy clearance.

The archived Table-1 artifact remains read-only and hash-bound as the task,
initial-state, pairing, and frozen-perception authority.  The completed job
`37109` result is separately read-only and bound by file SHA-256
`d79a28585e74cece727fbc3d3e6a72eb1647a782cae4962ec3af9045f448e603`
and payload SHA-256
`2a4ecd5ff79a363141e0927d1dfbcfb26248ec9bb5e2bc123cc8f879c3395a53`.
Simulation may run only in a clean Slurm allocation containing exactly one
H100.

## Internal-substep measurement

Robosuite calls `_update_observables` after every internal `sim.step`.  The
probe temporarily wraps that method only inside its cloned transition and
records the interval start plus every resulting internal physics state.  It
records all eight proxy gaps, robot joint position and velocity, OSC goal,
obstacle pose, and raw nonpositive contact events.

For each direct L5--L7 contact point, the audit records its normalized
quadratic value in the matching robot slab union and frozen obstacle MVEE.
The geometry gate passes only if every raw contact point lies within both
registered proxy supersets and the corresponding center-direction support gap
is nonpositive.  Contact with a positive gap or an uncovered contact point
means geometry is not an authoritative target for transition-only learning.

## Candidate and affine model

The frozen candidate set contains the nominal action, clipped central
differences of `0.1` in XYZ, all local Cartesian offsets at magnitudes `0.25`
and `0.5`, the `{-1,0,1}^3` global lattice, stop, and reverse nominal.  Duplicate
actions are removed deterministically.

For candidate action `u`, ground truth for constraint `i` is

\[
c_i^\star(s,u)=\min_k h_i(s_k),
\]

where `k` ranges over the interval start and every internal MuJoCo step.  In
the `L_infinity <= 0.5` trust region, the oracle model is

\[
\widehat c_i(s,u)=c_i^\star(s,u_0)+g_i^T(u-u_0).
\]

Each `g_i` is the nominal-anchored least-squares optimum.  The conservative
offset is the maximum one-sided overprediction on the registered local
candidates plus `1e-6 m`.  The exact OSQP then minimizes squared deviation
from the nominal XYZ subject to all eight corrected affine lower bounds being
at least zero, the original normalized action bounds, and the registered
`L_infinity <= 0.5` trust region.

## Go/no-go rule

The direction receives a local structural GO only if:

- the nominal transition reproduces raw protected contact and every contact
  passes the proxy-coverage consistency gate;
- at least one candidate inside the trust region is both raw-safe and has all
  eight minimum-substep proxy gaps nonnegative;
- the calibrated affine lower bounds have zero false-safe registered local
  candidates;
- the eight-row QP is feasible; and
- exact substep execution of its proposal is raw-safe and proxy-safe.

Otherwise the artifact records one of four primary stop reasons: geometry
authority failure, no jointly raw/proxy-safe local action, affine-model
failure, or QP/exact-execution failure.  No margin, proxy, candidate set,
trust region, or solver parameter may be changed after observing the result.
