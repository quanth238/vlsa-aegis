# Static full-body Poisson-CBF feasibility review

## Decision

The proposal is feasible as a staged **simulator-ground-truth, static-obstacle
mechanism pilot**, but it is not runnable or claim-valid as originally
written.  This branch begins at the exact source commit used by the completed
Table-1 reproduction:

```text
base commit: 1592aa59361f431ba96c6ddcbebcb596f6c20853
base tree:   928409ab6fc75c2543c9f84a08fc69028852df6d
branch:      codex/full-body-poisson-cbf-feasibility
```

The method would be an independent implementation from the equations in
Wilkinson et al., not a reproduction of released Poisson code.  The paper
defines joint-velocity control, robot-surface sampling, an epsilon-buffered
free domain, a Poisson safety function, and a multi-constraint CBF-QP.  It
does not provide enough public implementation detail to copy an existing
pipeline.

## What the existing repository supports

| Requirement | Current support | Decision |
|---|---|---|
| Frozen SafeLIBERO state, seeds, horizons | Yes | Reuse through a new immutable manifest |
| Exact recorded action replay | Yes, `actions[*].executed` | Shadow/debug only |
| Active-obstacle identity and MuJoCo topology | Yes | Reuse frozen contact authority |
| Robot and selected-obstacle collision geoms | Yes | Use collision-enabled geoms, not visual meshes |
| Joint positions and velocities | Available at runtime | Add value logging; archived results contain hashes only |
| Arbitrary body-point Jacobian | MuJoCo API is available | Validate numerically before use |
| JOINT_VELOCITY controller | Robosuite supports it | New eight-element action path and calibration required |
| Full-body surface samples | No | Implement and coverage-audit |
| Conservative voxel grid and buffer | No | Implement and test against exact geoms |
| Poisson field, gradient, interpolation | No | Implement with an independent reference solver |
| Seven-joint MC-CBF-QP | No | Implement with hard limits and fail-closed behavior |
| Cartesian-to-joint nominal adapter | No | Major integration risk; matched adapter-only arm required |
| Physics-substep full-body contact / clearance | No | Add before an active safety claim |
| Dynamic field, carried-object or self-collision safety | No | Out of scope for this pilot |

The claim-bearing Table-1 runner is `main/evaluate_safelibero_aegis.py`, not
the released demo.  It intentionally accepts only the two completed Table-1
arms.  The pilot must therefore use a new opt-in runner, schema, manifest, and
result root.  It must not alter or reinterpret the completed Table-1 run.

## Corrections required before implementation

### 1. Correct the nominal adapter scale

Do not use

```text
xi_nom = [0.2 * policy_xyz, 0, 0, 0]
```

The released AEGIS factor `0.2` converts its internal optimization variable
back to the original normalized OSC action and cancels the preceding factor
of five.  It is not the VLA-to-velocity scale.  The pinned OSC controller maps
normalized XYZ input to a position-goal delta in `[-0.05, 0.05]` m.  Derive a
nominal twist from the controller-scaled delta and the 50 ms control interval,
then calibrate it against realized motion.  An extra `0.2` can create apparent
safety by slowing the robot fivefold.

### 2. Treat controller replacement as a separate intervention

The current OSC environment expects six arm controls plus one gripper action.
JOINT_VELOCITY expects seven normalized joint-velocity controls plus one
gripper action.  Its default normalized range maps to approximately
`[-0.5, 0.5]` rad/s, so a physical `qdot_safe` cannot be passed directly.

Restore the exact settled simulator state into the new controller, synchronize
its internal goal, preserve the gripper command, and record command-to-measured
velocity tracking.  Every causal comparison needs an adapter-only arm.

### 3. Separate replay from closed-loop evaluation

Replay exact `actions[*].executed` to validate geometry and shadow algebra.
Do not replay `nominal_raw`.  The archived results do not contain per-step
qpos/qvel values, so Jacobian and counterfactual stages require a fresh,
instrumented replay.  Once the controller changes the trajectory, later VLA
observations and actions may legitimately diverge; task evaluation must use
fresh closed-loop inference with the frozen checkpoint and noise schedule.

### 4. Enforce the static-field assumption

SafeLIBERO obstacles have free joints.  A field built once after settling is
valid only while the selected obstacle remains at the field pose.  Log pose
drift before every control update and preregister the admissible tolerance.
Cases that exceed it stay in the all-case denominator but cannot support the
static theorem claim.  The 47 support-contact link cases are stress tests, not
primary static evidence.

### 5. Use exact physical collision geometry

The fixed link-5/6 population contains four active obstacle types: milk, moka
pot, red coffee mug, and wine bottle.  Their physical collision models are
unions of oriented box geoms; rendered meshes are collision-disabled.  Start
with conservative box-cell intersection.  General mesh voxelization is not
needed for this first pilot and must not silently substitute visual geometry
for MuJoCo collision authority.

### 6. State the geometry guarantee precisely

A finite dense-point validation is an empirical coverage audit, not a formal
proof over a continuous surface.  Either derive certified primitive/triangle
covering bounds or use the term `coverage-audited`.  Define voxel occupancy so
that a cell is free only when every point in it is farther than epsilon from
the exact obstacle union.  Do not add a half-cell diagonal twice through both
voxelization and buffering.

### 7. Remove artificial outer-boundary behavior

The Poisson problem also sets the outer workspace boundary to zero.  A local
box with only 20 cm padding can become an artificial obstacle.  Freeze a
workspace large enough for the robot and intended motion, log the distance to
that boundary, and report any constraint activated by it.

### 8. Evaluate the CBF residual, not only minimum h

The diagnostic that predicts intervention is

```text
r_i_nom = grad(h_i)^T J_i qdot_nom + alpha_i h_i.
```

A Poisson safety function is not a metric distance, and the smallest `h` can
come from the outer boundary.  Compare the most negative nominal residual
with exact simulator clearance and the first link contact.  Keep minimum `h`
as a secondary diagnostic.

### 9. Add missing physical verification

Existing results sample active-obstacle contact only after each 50 ms
`env.step`.  They cannot rule out contact during internal MuJoCo substeps.
Before active claims, record every physics-substep contact and full-robot to
selected-obstacle simulator clearance (`D_sim`).  Keep optimizer/grid
clearance (`D_opt`) separate.

## Fixed targeted population

The 109 cases are an outcome-conditioned mechanism population, not an
unbiased SafeLIBERO benchmark.

| Stratum | Cases | Task failures | Use |
|---|---:|---:|---|
| Positive released AEGIS barrier; no settled relevant contact | 60 | 31 | Primary static feasibility population |
| Active obstacle already contacts movable support after settling | 47 | 29 | Support/dynamic stress population |
| Released AEGIS barrier becomes nonpositive before link contact | 2 | 2 | Recovery/sampled-data stress population |
| **All link-5/6 cases** | **109** | **62** | Intent-to-treat denominator |

`Clean` above means positive **released AEGIS proxy** barrier and no sampled
settled relevant contact.  It does not establish a positive buffered Poisson
field.  Every case needs a new safe-start preflight.  Report both all 109 and
the preregistered static-admissible subset; never drop infeasible cases.

## Revised experiment ladder

1. Run the allocation-backed prerequisite probe in this branch.
2. Freshly replay one exact recorded failure and log qpos, qvel, full state,
   controller state, all contacts, impulses, and selected-obstacle pose.
3. Build surface samples and exact box occupancy; audit continuous units,
   coverage, buffer, connected free domain, and outer boundary.
4. Validate a reference Poisson solve, optimized solve, interpolation,
   gradients, PDE residual, sign, and boundary values.
5. Shadow exact recorded execution.  Require byte-identical actions and
   outcomes while logging `h`, Jacobians, nominal residuals, and `D_sim`.
6. Run a synthetic static-box manual controller with substep verification.
7. Run frozen VLA plus adapter only.  Establish useful task motion before
   adding safety.
8. Under the same controller and geometry, compare adapter only,
   end-effector-only PSF, link-5/6 PSF, and all-moving-link PSF.
9. Freeze parameters after three to five bring-up cases.
10. Run all 60 primary cases, then all 47 support cases and both recovery
    cases without further tuning.

For a 100 Hz filter, hold each 20 Hz VLA action over five 10 ms inner updates,
recompute robot state/Jacobians/constraints each update, and preserve the same
50 ms high-level action duration.  A 20 Hz active filter is only a sampled-
data canary.

## Required comparison and endpoints

The minimum causal comparison is:

```text
same frozen VLA -> same joint-velocity adapter -> no PSF
same frozen VLA -> same joint-velocity adapter -> full-body PSF-CBF
```

Add end-effector-only and link-5/6-only PSF arms to isolate geometric scope.
An original OSC/AEGIS replay is a descriptive reference, not a matched
controller ablation.

Report physical link contact, minimum positive `D_sim`, CAR, task success,
goal progress, executed path length, motion retention, correction magnitude,
QP infeasibility, invalid field queries, controller tracking, and runtime.
Because `qdot=0` is feasible for a static safe-start CBF, collision avoidance
alone is insufficient: task progress and retained motion determine whether
safety came from useful correction or stopping.

## Claim boundary

If successful, the pilot can support:

> On a fixed, collision-conditioned arm-link diagnostic population, a
> simulator-ground-truth, selected-obstacle, sampled-body static PSF-CBF
> reduced physical link contact relative to an otherwise identical
> joint-velocity adapter while retaining useful task motion.

It cannot support dynamic full-body safety, a reproduction of the paper's
implementation, safety over all 1,600 cases, repaired perception, carried-
object or self-collision safety, or superiority of Poisson fields over a
matched full-body distance/SDF CBF.  The last claim requires an identical
full-body exact-distance comparator.

## Repository and cluster gates

`feature_list.json` at the base commit still marks `A04-population` active.
Under the repository rules, a verifier must close the completed reproduction
and analysis gates before a new Poisson implementation gate becomes active.
Until then, this branch contains proposal review and prerequisite evidence
only.

The first H100 check must use one Slurm allocation, no policy server, and a
new output root under `/mnt/data/quanth/experiments`.  It must record the exact
Git commit, interpreter, packages, Slurm ID, host, controller properties,
geometry authority, Jacobian result, and an atomic `results.json`.  The
completed Table-1 result root is read-only.

## Sources

- Wilkinson et al., [Full-Body Dynamic Safety for Robot Manipulators: 3D
  Poisson Safety Functions for CBF-Based Safety Filters](https://arxiv.org/abs/2604.21189),
  arXiv:2604.21189v1, 2026.
- Completed Table-1 reproduction run:
  `vlsa-table1-contact-authority-population-20260718a`.
