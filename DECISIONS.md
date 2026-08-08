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

## ADR-0035: Test predictive full-body flow guidance with cloned OSC dynamics

Accepted by direct user instruction before implementation or outcome review.
The next `E02` experiment moves the safety correction from a post-sampling,
one-action QP into every Euler step of the ten-step pi0.5 flow sampler. It uses
40 discrete trajectory-CBF inequalities: ten future steps for each of the
separate L5, L6, L7, and released AEGIS end-effector ellipsoids. The L5--L7
MVEEs, frozen obstacle MVEE, `D_opt=0.01 m`, paper decay `gamma=0.9`, and
translational-only action protocol remain fixed.

The trajectory model is not the rejected resolved-rate approximation. It is
locally identified by centrally perturbed ten-step rollouts from a complete
clone of the current MuJoCo, stateful OSC_POSE, and gripper state. The affine
constraints are transformed into pi0.5's normalized action coordinates with
scale only for displacement, then projected after each neural Euler update.
The returned chunk must pass an exact cloned ten-step OSC rollout before any
of its first five actions may execute. One frozen outer relinearization is
allowed; failure after that is retained as method failure, never passed
through or hidden by stopping-task success.

This is a single-case oracle capability test, not a real-time or population
claim. The default OpenPI sampler and ordinary AEGIS evaluator remain
unchanged without the reserved opt-in control envelope. No KKT/VI or neural
approximation training is authorized by this decision.

## ADR-0036: Retain predictive flow guidance as safe but task-incompetent

Accepted after clean H100 job `37024`. The live initial observation and first
pi0.5 chunk exactly match the immutable Table 1 task-success episode. All 18
guided proposals used cloned stateful OSC_POSE horizon models, every executed
guided chunk passed exact ten-step verification, and the 300 executed actions
produced no robot contact, no paper CAR, and negligible obstacle motion.

This is not counted as solving the primary problem because the robot never
moved the bowl or satisfied any native goal atom. All 16 modified policy
queries occurred in steps 0--79. The minimum L5/L6/L7 buffered clearances
remained `108.667/129.095/95.291 mm`, while the end-effector proxy reached
`0.333 mm`; therefore the predictive end-effector constraint, not a distal
link constraint, caused every intervention and diverted the closed-loop
policy before the archived L5/L6 collision regime.

The retained interpretation is
`predictive_full_body_safe_but_task_incompetent_no_safety_efficacy_claim`.
The result verifies that an OSC-consistent cloned transition and exact rollout
gate can enforce the specified horizon inequalities, but it also demonstrates
that those hard inequalities can remove the task route. We do not weaken the
margin, shorten the horizon, remove the explicitly requested end-effector
constraint, or tune activation after observing the outcome. Any follow-up is
a new preregistered experiment and must recover a task-competent paired
baseline before KKT/VI or neural approximation training is reconsidered.

## ADR-0037: Require decoded-pixel verification for simulation videos

Accepted after auditing the job-`37024` MP4. A readable container, plausible
frame count, and valid first frame are insufficient evidence: that file became
visually corrupted after frame zero. It is retained only for provenance and
must not be presented as simulation evidence.

Replacement videos are reconstructed on H100 from the immutable accepted
action ledger. Acceptance requires exact replay of recorded primary states,
end-effector coordinates, obstacle motion, contact, CAR, and task outcome;
visible L5/L6/L7/end-effector and obstacle ellipsoid overlays; successful
decoding of every expected frame; and distributed decoded-pixel comparison
against the in-memory source frames. H100 job `37031` satisfied this contract
and completed normally. A renderer receipt written before a nonzero Slurm exit
is not sufficient: prior attempt `37029` is an apparatus failure even though
its bytes happened to pass the content checks.

## ADR-0038: Test EmbodiSteer's metric and schedule as an OSC surrogate

Accepted by direct user instruction before the outcome is known. The primary
reference is Wang et al., *EmbodiSteer* (arXiv:2606.12965). We adopt two
specific mechanisms: the task-preservation metric from Eqs. (6), (14), and
(15), and the reverse-step guidance schedule from Eq. (16). We extend the
paper's one safety inequality to one simultaneous projection containing every
registered L5/L6/L7/end-effector horizon row.

We do not label the experiment full EmbodiSteer. The released, task-competent
SafeLIBERO path executes Cartesian `OSC_POSE` actions; converting a corrected
joint null-space trajectory back to Cartesian commands would erase the paper's
central redundancy. The preregistered surrogate instead identifies the
end-effector-trajectory metric and barrier derivatives through exact cloned
OSC transitions, applies metric multi-row projection after each pi0.5 flow
Euler update, and verifies the proposed final chunk by exact cloned rollout.
It isolates whether the paper's task-preserving metric plus late guidance fixes
the earlier Euclidean hard-projection task diversion.

The four paper baseline groups are used as an explanatory evidence matrix, not
as a falsely matched reproduction. Existing immutable evidence supplies the
Cartesian baseline, direct-joint diagnostic, and post-hoc multi-CBF outcomes;
only the scheduled task-metric flow arm is newly executed. The direct-joint
diagnostic remains baseline-incompetent and uses a different checkpoint, so it
cannot support a cross-group efficacy claim. Success for the new arm still
requires native task completion, no robot contact, no paper CAR, and exact
verification of every executed guided chunk.

## ADR-0039: Retain task-metric guidance as safer but still task-incompetent

Accepted after clean H100 job `37042`. The run exactly matched the immutable
initial pi0.5 chunk, executed 300 actions with no contact or CAR, and accepted
only chunks passing exact cloned-OSC verification. It did not move the bowl or
complete the task. Therefore it is a negative capability result, not evidence
that the requested primary problem is solved.

EmbodiSteer's metric and late schedule materially reduced over-steering versus
the earlier Euclidean flow projection: guidance ended 40 actions earlier and
the end effector approached the bowl by another `72.707 mm`. The remaining
failure is structural. At the initial paired state the successful nominal
route violates only the released end-effector-proxy horizon constraint by
`12.015 mm`, while every distal-link constraint is far inside its safe set.
Any hard projection must change that route; minimizing end-effector deviation
cannot make a declared-infeasible route feasible.

This result does not reject full EmbodiSteer. The tested surrogate lacks the
paper's defining joint-space sampling and null-space execution authority.
Conversely, implementing true joint-space denoising requires a task-competent
joint controller baseline, which the prior direct-joint and bridge pilots did
not establish. We do not post-hoc remove the end-effector row, change its
geometry, weaken the margin, or tune the schedule. Such changes require a new
preregistered comparison.

## ADR-0040: Establish matched EE and Joint competence before geometry

Accepted by direct user instruction before outcome review. The previous
`pi05_droid` direct-joint pilot used a different checkpoint, and the Cartesian
bridge applied joint conversion only after Cartesian sampling. Neither is the
`Joint` baseline defined in EmbodiSteer Appendix B.4. The new paired pilot
therefore uses one frozen `pi05_libero` checkpoint for both arms and performs
FK and damped-Jacobian lifting after every reverse denoising step in the joint
arm. The Cartesian arm remains an ordinary unmodified policy request.

All collision interventions are disabled. No L5/L6/L7 ellipsoid is built, no
clearance gradient is queried, no barrier inequality is formed, and no QP is
solved. The obstacle remains physically present so raw MuJoCo contact, CAR,
and native goal evidence are still observed. Acceptance requires one H100
allocation, exact settled-state pairing, the same query-index noise schedule,
valid decoded videos for both arms, and truthful reporting of failure or
infeasible direct-joint tracking.

This pilot adapts the paper because its policy and ours use different action
representations and generative schedulers. EmbodiSteer uses 10D chunk-start
relative Cartesian poses and a 16-step DDPM; SafeLIBERO pi0.5 uses 7D
incremental OSC commands and ten flow-Euler steps. Incremental commands are
composed into chunk-start poses for Eqs. (3), (4), and (8)--(10), then converted
back before each frozen pi0.5 query. The result may support or reject this
adaptation's competence on one primary case, but cannot be reported as an
exact reproduction of the paper's population result.

The user's L5/L6 geometry concern is accepted as a separate blocking issue for
future guidance. EmbodiSteer represents the robot with multiple link-attached
cuRobo collision spheres and aggregates the top four signed distances. One
mesh-enclosing MVEE per link can be a valid enclosure yet still be too loose,
hide local contact structure, and supply a qualitatively different gradient.
The existing MVEEs remain disabled and untrusted for paper-fidelity claims
until audited or replaced by a sphere/capsule decomposition.

The sampler regression is accepted in physical rather than bitwise units.
Interleaving simulator FK/Jacobian work between reverse steps requires ten
separately compiled Euler calls, whereas ordinary pi0.5 fuses them in one JAX
while-loop. Allocation-only diagnostics before any arm rollout measured at
most `45.767 micrometers` translation error, `0.000124 rad` rotation error,
and identical first-five gripper signs. The frozen gate is `0.1 mm`,
`0.001 rad`, `0.005` raw action units, and exact gripper-sign equality. This
calibrates numerical equivalence only; it does not change any task, geometry,
controller, or safety parameter.

## ADR-0041: Reject v1 Joint execution and correct the absolute-target adapter

Accepted after auditing clean H100 job `37055`. EmbodiSteer returns the joint
configuration chunk `Q_0`; those samples must be supplied as position targets.
SafeLIBERO's `JOINT_POSITION` controller instead interprets each normalized
input as a delta from the current joint state. The v1 adapter used a
`0.05 rad` output range and clipped `(q_target-q_current)/0.05`, causing
saturation in 220 of 300 actions and up to `0.963 rad` post-step target error.
That is not a faithful direct-joint baseline, regardless of its no-contact
outcome.

Protocol v2 keeps the same `Q_0`, checkpoint, controller gains, control
frequency, action horizon, obstacle scene, and no-guidance condition. It sets
the delta encoding range to `6 rad`, larger than the maximum Panda joint range,
so normalization represents every bounded absolute target without clipping.
Zero encoding saturation is an acceptance condition; physical PD tracking
error remains measured rather than assumed.

The v1 joint video is also rejected. All 301 frames decoded, but source pixels
became visibly corrupted, proving that shape/count validation is insufficient.
V2 executes each arm in a fresh process to isolate MuJoCo/OSMesa contexts and
checks adjacent-pixel variation on every source and decoded frame. The
threshold is fixed before v2 execution from valid Cartesian frames near
`2/255` and corrupted joint frames above `17/255`; acceptance requires at most
`8/255`. Neither correction authorizes L5/L6 geometry or safety guidance.

## ADR-0042: Align the barrier-free stress pilot with the paper's control rate

Accepted after H100 v2 job `37058` failed before a valid Joint result. Running
the two arms in fresh processes did not prevent the fixed agent-view stream
from flipping at action 10 and failing the pixel-integrity gate at action 11.
The failure is apparatus/control evidence only; it is not a task, collision,
or EmbodiSteer outcome.

The paper executes its whole 16-action joint chunk at `10 Hz`. The frozen
pi0.5-LIBERO checkpoint has a ten-action horizon, so v3 executes all ten
available actions at `10 Hz` and explicitly records the remaining horizon and
action-representation differences. It retains the exact absolute-target
adapter, checkpoint, initial state, query-seed schedule, controller gains,
primary obstacle scene, and zero-guidance condition. The primary obstacle
case is a stress test and must not be equated with the paper's obstacle-free
`Joint` baseline population.

The visual gate is extended before v3 execution to catch both high-frequency
corruption and 180-degree orientation changes. On failure, raw and processed
frames, action/joint-state history, and fixed-camera poses are atomically
retained. No failed visual run is promoted to scientific evidence, even if
its MP4 decodes.

## ADR-0043: Eliminate the second OSMesa environment from joint denoising

Accepted from the fail-closed trace of H100 job `37061`. The first direct
joint target was well behaved: zero encoding saturation and at most
`0.299 rad/s` observed velocity. The fixed MuJoCo camera transform remained
bitwise identical, but the source image became an almost exact 180-degree
rotation immediately after action zero. This rules out the sampled target or
physical camera motion as the cause of that frame.

The remaining context violation was the separate rendered environment used
only for kinematic queries inside the Joint worker. The repair reuses the
live MuJoCo model for FK and Jacobians, then restores the full flattened state
and requires bitwise equality before action execution. This preserves exact
kinematics while avoiding a second OSMesa context. It is an apparatus repair,
not a change to EmbodiSteer equations or the baseline protocol.

## ADR-0044: Scale the FK-pose serialization gate by float32 precision

Accepted after H100 job `37062` passed the renderer repair but stopped on a
fixed `1e-7` comparison between a float64 FK action and the same action after
server-side normalization, float32 model storage, and affine decoding. The
comparison is an apparatus-integrity check; it does not constrain safety or
task behavior.

The replacement bound is `32 * eps_float32 * max(1, |input|, |decoded|)`.
It scales only with IEEE float32 precision and the serialized value magnitude,
and is frozen before observing its live error. Each denoising step records the
error and bound. The policy input, denoising update, joint trajectory, control
rate, execution horizon, and all acceptance outcomes remain unchanged.

## ADR-0045: Reject the pi0.5 direct-Joint adaptation as a matched baseline

Accepted after validated H100 job `37067`. The apparatus gates passed: one
frozen checkpoint and settled state, no collision guidance, zero joint-target
encoding saturation, bitwise state restoration after live-model kinematics,
precision-valid sampler round trips, and two valid 301-frame videos. The
negative outcome is therefore retained rather than repaired by more tuning.

Neither arm completed the primary obstacle task. Cartesian contacted link 6
at step 193; Joint contacted link 5 at step 79. Joint target tracking remained
poor (`0.255 rad` mean and `1.225 rad` maximum error) and aggressive (maximum
`3.503 rad/s` observed velocity), despite exact target encoding. The result
does not reproduce the paper's competence-preserving `Joint` baseline.

This does not falsify EmbodiSteer. Its released description assumes a 10D
chunk-start relative pose diffusion policy, a 16-action chunk, a native joint
execution interface, and reports `Joint` competence without obstacles. Our
frozen pi0.5-LIBERO checkpoint produces ten incremental 7D OSC flow actions
and the primary SafeLIBERO stress scene retains its physical obstacle. Future
EmbodiSteer safety work is blocked until a competent Joint baseline is
established with a natively compatible policy/controller. If geometry is
reintroduced afterward, use audited link-attached spheres/capsules and
top-critical-distance aggregation; do not present one MVEE per L5/L6 link as
the paper's geometry.
