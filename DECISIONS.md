# Reproduction decisions

## ADR-0001: Start from the untouched author release

Accepted. The reproduction branch starts at upstream commit `57b1aef`. The
previous same-state canary code is not copied into this worktree.

## ADR-0002: Reproduce the translational protocol first

Accepted. The user's Table 1 values correspond to `pi0.5,t` and `AEGIS_t`.
Full-action AEGIS is a separate supplementary protocol and cannot be mixed into
the primary result.

## ADR-0003: Repair apparatus defects without changing the safety algorithm

Accepted. Permitted repairs are limited to:

- pass the suite name into the existing point filter;
- recognize `safelibero_long` and its 550-action horizon;
- represent empty perception and infeasible QP as explicit terminal apparatus
  states rather than undefined-name crashes;
- add structured artifacts, deterministic pairing controls, streaming videos,
  and diagnostic measurements around the released loop.

No CBF, MVEE, GroundingDINO, threshold, action scale, solver, or policy setting
may be tuned to improve the result.

## ADR-0004: Make policy-noise pairing opt-in

Accepted. The reserved `__crfs__` request envelope carries a per-request JAX
seed and is removed before normal OpenPI transforms. Requests without the
envelope retain the upstream inference path.

## ADR-0005: Freeze Codex labels before outcomes

Accepted because GLM-4.5V is unavailable and the user explicitly requested
Codex. Capture runs execute only reset, the 20 settling steps, and lossless
image export. A label is valid only when bound to the exact settled image hash
before any policy or AEGIS outcome is produced.

## ADR-0006: Report paper metrics and physical diagnostics separately

Accepted. The primary table retains:

- CAR: no active-obstacle L1 displacement greater than 1 mm;
- TSR: environment task success within the suite horizon;
- legacy ETS: the released loop counter.

Each result also records executed action count, MuJoCo contact evidence when
available, selector/perception/MVEE/QP state, intervention magnitude, and video
identity. These diagnostics must never be substituted into the paper table.

## ADR-0007: Never aggregate a partial population

Accepted. The aggregator requires all 1,600 unique cases for every requested
method. Policy failures remain scientific failures; apparatus failures block
publication and must be rerun rather than dropped.

## ADR-0008: Align the first canary with the paper's qualitative task

Accepted. The capture-only canary uses manifest ordinal 100
(`safelibero_spatial`, Level I, logical task 2, episode 0), matching the task
in the supplied failure/success figure. The episode index is not claimed to be
the unpublished episode used to make the authors' figure.

## ADR-0009: Separate capture readiness from outcome readiness

Accepted. A fresh capture-only allocation may collect pre-outcome evidence
while `A01-reproduction-apparatus` remains active. No pi0.5 or AEGIS outcome
may run until the full checkpoint identity and paired-result validator are
implemented and independently reviewed.

## ADR-0010: Use the allocation-compatible OSMesa renderer

Accepted after exact job `28460_0` failed before reset because EGL attempted
to open inaccessible host render devices. VinUni's legacy Robosuite/MuJoCo
workloads use `MUJOCO_GL=osmesa` and `PYOPENGL_PLATFORM=osmesa`. This changes
only the headless rendering backend; the H100 remains allocated for pi0.5
inference in outcome jobs.

## ADR-0011: Restore the released pre-construction NumPy seed

Accepted after exact job `28461_0` reached OSMesa but failed during the
environment constructor's temporary random placement. The released evaluator
sets NumPy seed 7 before constructing OffScreenRenderEnv. Capture now restores
that same ordering before applying the immutable full simulator state.

## ADR-0012: Retain released-method failures and preserve the frozen action space

Accepted. Once all required dependencies and inputs are valid, an OSQP
exception, infeasible/invalid QP solution, or empty perception output is a
failure of the released method on that case, not permission to exclude the
case as an apparatus failure. Hard QP failures terminate and remain in the
population; empty perception uses an explicitly reported fail-open path.

The upstream empty-perception branch forwards the raw nominal action, including
rotation. That behavior conflicts with the registered Table-1
translational-only arm. The reproduction therefore applies the already
corrected translational nominal action (XYZ and gripper preserved, rotation
zeroed), reports `method_failure_passthrough`, and records both the corrected
execution and upstream behavior in every affected artifact. This is a
protocol-preserving wrapper correction, not a claim that AEGIS intervened.

## ADR-0013: Treat deterministic geometry failure as no-execution method failure

Accepted. A valid GroundingDINO result can still fail in the released
ConvexHull/MVEE construction. Such a case is retained as an exact pre-control
`aegis_geometry` method failure with zero policy queries, zero executed
actions, one evidence frame, and `safety_by_no_execution=true`. Only this
exact state may omit the first returned-action hash; all other failures remain
fully action-bound.

## ADR-0014: Require real QP execution before the paired canary can pass

Accepted. Successful schema validation is insufficient to establish that the
AEGIS integration ran. The paired-canary receipt additionally requires the
frozen selector to match the active obstacle, perception status `ready`, and
at least one executed `aegis_qp` action with valid finite OSQP diagnostics.
Fail-open, no-action, selector-mismatch, and empty-geometry canaries are
integration-inconclusive and cannot authorize the population.

## ADR-0015: Use the released GroundingDINO CUDA path first

Accepted. The first paired canary uses `GROUNDINGDINO_DEVICE=cuda`, matching
the release default. A CPU perception run would be a separately preregistered
compatibility variant, not a silent replacement for the released baseline.

## ADR-0016: Exclude mount-local `st_dev` from cross-worker identity

Accepted after paired-canary task `28468_0` stopped before execution. The
checkpoint hash was computed on worker-2, but Linux reports a different
`st_dev` for the same shared file through another cluster mount namespace.
`st_dev` identifies a mounted filesystem view, not file content, and therefore
cannot be compared across Slurm workers. The stable identity continues to bind
the resolved checkpoint path and every file's relative path, byte size, inode,
mtime, and ctime, in addition to the allocation-backed full-content tree hash.

## ADR-0017: Keep evaluator and publisher compatible with Python 3.8

Accepted after retry-B task `28470_0` reached the evaluator but failed before
the first policy query because `str.removesuffix` is unavailable in the
released SafeLIBERO Python-3.8 environment. Equivalent prefix/suffix slicing
is used in the evaluator, manifest builder, and aggregator, and a structural
test prohibits both Python-3.9-only string helpers in these runtime modules.

## ADR-0018: Prove failure diagnostics are action-invariant

Accepted. The revised ordinal-100 integration gate runs four serial rollouts
inside one allocation and one pi0.5 server:

1. pi0.5 with diagnostics disabled;
2. pi0.5 with diagnostics enabled;
3. pi0.5+AEGIS with diagnostics disabled;
4. pi0.5+AEGIS with diagnostics enabled.

For each method, the enabled and disabled runs must preserve exact returned
action bytes, executed action bytes, query-indexed noise schedule, scientific
outcome fields, simulator state semantics, native goal progress, and decoded
terminal video content. Only observer timing, diagnostic records, diagnostic
file encodings, and their hashes may differ. This is an apparatus gate, not a
population efficacy result.

## ADR-0019: Freeze the revised gate to GroundingDINO on CPU

Accepted and supersedes ADR-0015 for the revised canary and dependent
population. The validated historical canary
`vlsa-table1-paired-canary-20260717d` used GroundingDINO on CPU and produced
the exact AEGIS action reference used by the action-invariance gate. Allowing a
CUDA detector in the revised gate could change detector numerics and AEGIS
actions, making the reference comparison ill-posed. Reservation, allocation,
publisher, and receipt validation therefore all require the CPU detector.

## ADR-0020: Bind outcomes to the complete outcome-blind label publication

Accepted. Evaluation requires the exact 1,600-row Codex label manifest and
its independently validated publication receipt. Every canary result must
embed the exact selected ordinal-100 label record, and receipt regeneration
rechecks the same binding. A copied action reference, a changed label row, a
different mode/arm mapping, or a label publication with any outcome execution
is rejected.

## ADR-0021: Keep the full population blocked until diagnostics scale safely

Accepted. Enabling diagnostics in the population runner is insufficient by
itself. Before release, the CPU publisher must deeply validate every case's
GroundingDINO boxes, point clouds, filtering and MVEE outputs, QP attempts,
step-resolved contacts, native goal-progress trace, terminal lossless frame,
and decoded video. Validation and aggregation must stream or compact each pair
so they do not retain all 3,200 full diagnostic result dictionaries in memory.

## ADR-0022: Accept the action-invariant canary, not a population claim

Accepted from terminal allocation-backed evidence. Exact Slurm task `28590_0`
completed on `worker-2` from clean release commit
`5105894faebd40d2e27e2011d33688b25c2578dc`. Receipt SHA-256
`6a80afce3ce019bf43090da65c85717344b6dafb4a983d5a9efcf6bbecd46d95`
validated all four frozen ordinal-100 rollouts. Diagnostics off/on preserved
exact action bytes, policy-query schedules, outcome semantics, and video bytes
for both arms. The baseline collision and task failure and the AEGIS
collision-free task success reproduce the historical canary. This passes
`A03-one-case-paired-canary` and permits work on `A04-population`; it does not
establish AEGIS population efficacy or authorize a launch before the
memory-safe deep population validator passes.

## ADR-0023: Stream, recompute, and byte-bind the population publication

Accepted for implementation; allocation-backed validation remains pending.
The publisher may hold at most one full result pair in memory. It deeply
validates that pair, writes a compact receipt row, and releases the full
objects before reading the next case. After all 1,600 pairs are present, an
independent finalizer reloads the immutable manifest and configuration,
recomputes the aggregate table, exhaustive failure report, and indexed video
gallery from the actual result tree, and requires their exact hashes and bytes
to match the staged publication. It retains every scientific and apparatus
failure. Static-support contacts remain reported but are not counted as
robot--obstacle collision mechanisms.

## ADR-0024: Bind failure explanations to live simulator authority

Accepted for a fresh allocation-backed canary; population authorization
remains pending. The active obstacle is bound to the settled-input contract,
not inferred from a mutable result field. Each episode freezes the complete
MuJoCo body-parent and geom-to-body tables, exact named/unnamed ID tokens,
body-to-joint ranges, joint ownership/types/names, robot body IDs, the
active-obstacle root body, and the native task-object mapping. Every step
stores the ordered raw MuJoCo contact ledger; canonical events, normals, body
lineages, dynamics, mobility, roles, and goal membership are re-derived
against that authority. Goal arguments are also checked against the immutable
BDDL file. Direct goal objects, parented goal sites, and parentless fixed
arena sites are distinct bindings. Any ambiguous or unnamed non-world role is
`unknown` and fails publication. This contract is explicitly versioned as
contact schema `v3`, model-authority schema `v2`, and source-bound paired
canary receipt schema `v2`; the population reservation rejects stale receipt
schemas before allocation.

Independent validation reconstructs the exact robot-body ID set from frozen
MuJoCo body names rather than trusting the producer's list. It also requires
every task-object and goal-site-parent root ID to resolve to that object's
exact frozen body name or registered instance prefix. Coordinated,
fully-rehashed attempts to omit a contacted robot body or exchange two task
roots are rejected. A malicious rewrite of the complete frozen authority,
raw ledger, and all hashes remains the explicit clean-source producer trust
boundary; the fresh source-bound canary must validate that boundary before
population work.

These observer changes were made after validated task `28590_0`. That task
remains valid historical integration evidence for `A03`, but it cannot
authorize a population from the new source. The final clean release therefore
requires a new source-bound checkpoint receipt and the same four-rollout
ordinal-100 action-invariance canary before any population submission.

## ADR-0025: Start multi-link geometry as an opt-in read-only AEGIS observer

Accepted. Branch `codex/multilink-ellipsoid-qp` starts from clean current
AEGIS reproduction commit `1592aa59361f431ba96c6ddcbebcb596f6c20853`.
Completed Table-1 artifacts are immutable inputs. The first gate reruns only
primary case `vlsa-t1-goal-ii-t0-e05` and requires exact action-ledger equality
with archived AEGIS while evaluating the new geometry and QP before each
unchanged `env.step` call. This separates the questions "does the geometry see
the arm?" and "how expensive is the coupled QP?" from active-control efficacy.

Every contact-participating collision geom on Panda link 1 through link 7 has
its own conservative enclosing ellipsoid. This is a task-independent whole-arm
set; the historical collision link is not an input. The obstacle remains the
released frozen AEGIS perception MVEE. The correction variable is a physical
seven-joint velocity, derived from the executed translational AEGIS action only
as a read-only resolved-rate nominal. No neural model is trained.

## ADR-0026: Preregister optimizer clearance, simulator evidence, and timing

Accepted. The first shadow uses `alpha=10 s^-1`, `D_opt=0.01 m`, physical
joint-velocity bounds of `+/-0.5 rad/s`, resolved-rate damping `0.05`, and an
end-effector-preservation metric `I + 10 J_ee^T J_ee`. OSQP tolerances and all
parameters are frozen in
`configs/vlsa_multilink_ellipsoid_shadow_e05.v1.json` before H100 execution.

`D_opt` is the support-gap buffer used by the optimizer. `D_sim` is raw MuJoCo
nonpositive contact distance plus the separately reported active-obstacle
displacement. They remain distinct in every step record and validation
receipt. QP setup, solve, solver-reported, and total observer times are all
retained. Infeasibility is a reportable result, never silently dropped. The
only claim available from this gate is oracle geometry/QP implementation and
timing evidence; active safety requires the dependent `E02` gate.

## ADR-0027: Accept certified mesh bounds and retain hard-QP infeasibility

Accepted after validated Slurm job `36757`. The live Panda collision model has
one mesh geom on each of links 1 through 7. For this gate, each mesh remains
inside MuJoCo's compiled broad-phase `geom_rbound` sphere. The independent
validator recomputes every recorded sphere semiaxis from that authority and
requires all seven rows to enter the same QP at every action. We do not infer a
tighter mesh ellipsoid from `geom_size` without a separate mesh-vertex-bound
certificate.

The conservative geometry produced 182 solved and 55 primal-infeasible hard
QPs under the preregistered `+/-0.5 rad/s` limits. This is a valid, retained
oracle outcome, not an apparatus failure and not evidence for active safety.
Any `E02` active-control gate must first preregister either tighter certified
mesh ellipsoids, an explicitly penalized slack hierarchy, or both. A KKT/VI
network cannot be trained against silently discarded infeasible targets.

## ADR-0028: Narrow the next geometry gate to surface-fitted links 5, 6, and 7

Accepted by direct user instruction. `E02` excludes links 1 through 4 and
uses exactly three independent ellipsoids: one each for `robot0_link5`,
`robot0_link6`, and `robot0_link7`. These are not grouped multi-body
ellipsoids. Each bound is fitted from that rigid link's compiled MuJoCo
collision-mesh vertices using a Khachiyan minimum-volume fit, inflated just
enough to contain every vertex, and
retains the ordinary analytical rigid-link Jacobian used by the QP.

The obstacle remains the frozen released-AEGIS MVEE, and the QP receives
exactly three simultaneous constraints. This gate remains read-only until its
H100 visualization, containment certificate, action-invariance check, and
runtime receipt pass. Links 1 through 4 are deliberately outside the claim.

The validation rollout replays the immutable archived AEGIS `env.step` inputs
instead of asking the GPU policy to resample them. This is required after
attempt `36774` reproduced the settled simulator and policy-noise seed contract
but returned a different first policy chunk before the shadow observer ran.
Exact replay preserves the executed-action horizon while independently
recomputing MuJoCo contacts, task progress, three-link constraints, and timing.

## ADR-0029: Execute the first multi-CBF counterfactual in AEGIS XYZ space

Accepted. SafeLIBERO exposes a translational OSC action rather than direct
joint-velocity control. The first active experiment therefore maps executable
XYZ through the live damped resolved-rate kinematics, maps each L5/L6/L7
barrier row back to that XYZ variable, and solves all three inequalities in
one QP. This avoids claiming that an arbitrary joint-velocity solution was
executed when the simulator accepts only Cartesian actions.

The nominal sequence is the immutable released-AEGIS `env.step` ledger. This
is a controlled counterfactual for the safety layer, not closed-loop policy
efficacy. The QP may change only XYZ; rotation remains zero and the gripper
command is exact. Raw protected-link contact, paper CAR, and native goal state
jointly determine whether the primary report problem is solved.

## ADR-0030: Retain the negative Cartesian multi-CBF outcome

Accepted after clean H100 job `36782`. The exact-action controller solves all
three L5/L6/L7 inequalities at every action, but direct L5 contact and paper
CAR occur at action 189 and the native task does not complete. This is a
scientific controller failure, not an apparatus failure: source, initial
state, archived nominal ledger, geometry, solver status, and raw simulator
contacts are all complete.

We do not tune `alpha`, `D_opt`, ellipsoid size, or a fallback after observing
this result. The outcome shows that satisfying the continuous resolved-rate
linearization in the executable XYZ action is insufficient for discrete
SafeLIBERO OSC contact avoidance in this case. A later experiment must
preregister direct joint-space execution or nonlinear discrete-time
rollout/step reduction; KKT/VI learning is not a remedy for this model-to-
execution mismatch.

H100 video replay `36788` independently re-executed the accepted safe-action
ledger and matched the recorded end-effector, obstacle, and raw-contact traces
with zero error. A finite-difference audit identifies the concrete certificate
failure: action 186 predicted positive L5 barrier recovery (`+0.044412 m/s`)
while the next simulator state moved inward (`-0.143104 m/s`). This reinforces
the decision: retain the negative outcome and change the execution model in a
new preregistration, not the QP solver or KKT implementation.

## ADR-0031: Retain the direct joint-velocity pilot as baseline-incompetent

Accepted after paired H100 producer job `36802` and independent validator job
`36806`. The `pi05_droid` checkpoint directly drove all seven SafeLIBERO
joint velocities, and the active arm executed one simultaneous QP containing
the three independent L5/L6/L7 ellipsoid constraints. Pairing is exact until
the first material safety intervention at action 71, and every QP in both
225-action arms is valid.

Neither arm contacted the protected links or crossed the paper CAR threshold,
but neither arm ever satisfied the native task goal. Most decisively, the
unfiltered baseline never issued a close-gripper command. This makes the
baseline incompetent for the requested task, so the collision-free result is
not evidence that the three-constraint filter solves the original
task-completing AEGIS failure. We retain the videos, timings, raw evidence,
and negative interpretation without tuning the checkpoint or safety
parameters after outcome inspection. Any learned joint-space replacement
requires a new preregistration and a competent paired baseline before safety
efficacy or KKT/VI training can be considered.

## ADR-0032: Retain the AEGIS Cartesian-to-joint bridge as controller-incompetent

Accepted after paired H100 producer job `36824` and independent validator job
`36826`. The bridge starts from the exact archived AEGIS OSC-settled simulator
state, converts every immutable AEGIS XYZ action to live seven-joint velocity,
preserves every archived gripper command, and in the active arm executes one
simultaneous QP containing the separate L5, L6, and L7 ellipsoid constraints.
The two arms have identical pre-step simulator states and nominal joint
velocities through their first material difference at action 200; every one
of the 474 QPs is valid.

This replacement does not preserve the task or the original collision
mechanism. Neither arm completes the SafeLIBERO goal. Both push the obstacle
with the hand/gripper beginning at action 159 and displace it by about 0.245 m,
while neither records L5/L6/L7 contact. Consequently the single late L5--L7
intervention cannot answer whether the filter prevents the released AEGIS
link-5/link-6 failure. The retained interpretation is
`bridge_incompetent_no_safety_efficacy_claim`, not a safety success.

We do not tune the frozen Cartesian scale, damping, QP parameters, or
ellipsoids after inspecting this outcome. An archived Cartesian ledger is
feedback-dependent: once a different low-level controller changes the state,
later open-loop commands no longer represent what the policy would choose.
The next controller experiment therefore requires a new preregistration with
a task-competent Cartesian policy queried on live observations and competence
established in the unfiltered joint-controller arm before applying L5--L7
multi-CBF. KKT/VI learning remains gated on that oracle comparison.

## ADR-0033: Test a cloned-step discrete L5--L7 multi-CBF before learning

Accepted by direct user instruction. The accepted continuous XYZ multi-CBF
remains frozen as the first comparison. A new opt-in arm replaces only its
resolved-rate derivative with seven cloned SafeLIBERO transitions per action:
one nominal transition and clipped central perturbations in the three XYZ
dimensions. Their three next-step L5/L6/L7 buffered clearances define one
simultaneous three-row QP. The candidate is then executed in the clone and may
be reduced by the frozen scales `0.5, 0.25, 0.125, 0` toward stop until an exact
next-step check passes.

Every probe begins from a synchronized copy of the complete MuJoCo and
stateful OSC/gripper transition state. The primary `env.step` must reproduce
the accepted clone state and clearances within the preregistered tolerance.
No unverified action is executed. `D_opt=0.01 m`, the accepted L5/L6/L7 MVEEs,
the frozen obstacle MVEE, archived rotation/gripper commands, and the 237-step
nominal ledger remain unchanged. The experiment measures oracle ability and
runtime on the primary case only; it is not claimed to be deployable in real
time and does not authorize KKT/VI or universal-formula training.

## ADR-0034: Retain the cloned-step multi-CBF as safe refusal, not task success

Accepted after authoritative H100 job `36875`. The discrete controller used
exactly three L5/L6/L7 next-clearance constraints, all 186 QPs were valid, and
each of the 185 executed primary transitions reproduced its accepted cloned
transition with zero state error. It produced no robot contact, no paper CAR,
and negligible obstacle displacement, but it stopped before executing action
185 and never completed the native task. Therefore
`primary_problem_solved=false`; collision avoidance obtained by terminating
the task is not counted as a safety-method success.

The terminal evidence distinguishes the remaining failure from the original
continuous resolved-rate mismatch. The finite-difference QP predicted its
large correction would put L5 exactly on the buffered boundary, while exact
simulation put it `3.697 mm` inside. More importantly, even the predefined
zero-XYZ stop action ended `6.852 mm` inside the buffered boundary on the next
step. Thus exact one-step verification correctly vetoed every action, but the
state had already left the one-step viable set for the frozen `D_opt=10 mm`
requirement. Three ellipsoid rows are present and jointly solved; adding a
neural approximation of this same QP would not create a feasible action.

We do not reduce `D_opt`, weaken the exact verification tolerance, or add an
unregistered retreat after observing this outcome. A follow-up must be newly
preregistered and address anticipation, for example a multi-step viability
margin or direct simulator-in-the-loop nonlinear search begun before the
one-step stopping boundary. The observed mean complete-filter time of
`131.895 ms` also exceeds the 20 Hz control period, although the QP itself
remains fast at mean `1.182 ms`. KKT/VI or universal-formula learning remains
gated because it can only approximate a controller after the oracle target is
both feasible and task competent.
