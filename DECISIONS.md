# Reproduction decisions

## ADR-0057: Gate execution-margin learning behind an earlier exact-box oracle

Accepted by user instruction after job `37180`. Do not train on the late
action-192 state, where the exact proxy safe set has already been left.
Collect actions 180--192 from the immutable job-`37109` ledger and initially
test an exact-box, exact-substep recoverable crossing at action 191 before training.
The data artifact keeps seven optimizer margins distinct from raw simulator
contact and transient obstacle motion.

If that gate passes, learn only the nonnegative clearance loss from the
interval-start value. This structural form cannot claim that an interval
minimum exceeds its first sample. Calibrate per-row lower bounds on separate
state groups, apply seven simultaneous bounded QP rows, and exact-clone verify
the proposal. Eight millimetres is warning/activation only; zero is the hard
transition target.

Live jobs `37192`/`37193` require split runtimes: legacy evaluation PyTorch is
not H100-kernel compatible, while OpenPI PyTorch is. Use the former for
MuJoCo, the latter for H100 training, and a NumPy weight archive between them.
No learned steering is run before the oracle-analysis artifact validates.

Job `37194` showed that the shared oracle generator has state-dependent unique
counts because it deduplicates after clipping to action bounds. Retain that
generator and freeze its exact 85--87 counts (1,125 transitions total); do not
repeat clipped actions merely to make all state groups have equal size. This
apparatus repair does not change a candidate value or scientific threshold.

Validated job `37195` rejected action 191 before training and identified action
188 as the only recoverable crossing. The v2 experiment may use 188 only as an
explicit post-oracle feasibility target. To prevent direct leakage, train on
180--185, calibrate on 186--187, and hold out 188--192. This can demonstrate
one-step mechanism feasibility but cannot support an unbiased efficacy claim.

Job `37197` authorizes no scientific inference: it completed recollection but
failed before the first training operation because resolving the OpenPI Python
symlink escaped the virtual environment. Preserve the absolute launcher path;
the Slurm preflight remains responsible for proving H100 PyTorch execution.

Validated job `37198` rejects the frozen two-layer residual MLP mechanism gate.
The training groups never enter the unsafe proxy region, so the learned model
does not recover a useful boundary normal under the held-out state shift. Its
conservative calibration makes three action-188 rows negative while the tiny
learned Jacobian cannot recover them inside the fixed bounds; the QP is primal
infeasible. Do not tune away this result or present the model as a QP
replacement.

The representation remains controllable because exact safe candidates exist.
A future learned experiment therefore requires a new, explicit data-coverage
hypothesis: boundary-focused cloned rollouts with state/episode-separated
evaluation. Training on action 188 itself may test implementation capacity but
cannot validate generalization or the paper claim. Denoising-flow guidance is
still future work and must be called barrier-inspired unless separately proved.

## ADR-0056: Use the exact compiled obstacle boxes as the privileged oracle

Accepted after freezing job `37175`.  The second requested test returns to the
base zero clearance target and changes only the obstacle representation.  The
15 collision-active moka-pot geoms are boxes, so use their exact half sizes
and live MuJoCo poses instead of the frozen perception MVEE or their
`sqrt(3)`-inflated Loewner ellipsoids.

All robot/EE proxies, candidates, affine/QP settings, OSC transition, and
ordered decision gates remain fixed.  Negative interval-start gaps are valid
late-activation evidence.  This test is explicitly privileged simulator
geometry and can demonstrate feasibility only; it is not deployable
perception or a population safety result.

Validated H100 job `37180` repairs geometry authority: every L6/g12 raw
contact is contained by the exact source box and accepted robot slab with a
negative support gap.  It nevertheless rejects action-192 efficacy.  Two
proxy rows are already negative at interval start, no candidate is proxy-safe,
and the QP is primal infeasible.  The exact proxy detects the impending raw
contact, but the hard minimum-substep formulation is activated after its safe
set has already been left.  Any further test must move earlier in the
immutable ledger rather than tune geometry or solver settings post hoc.

## ADR-0055: Test the fixed 8 mm warning margin before exact obstacle boxes

Accepted by direct user instruction.  Before using privileged MuJoCo obstacle
geometry, retain the accepted L5--L7/EE ellipsoids and frozen released-AEGIS
obstacle MVEE, then replace the zero clearance target by `8 mm` on all eight
QP rows.  This is exactly `h_corrected = h_AEGIS - 8 mm`; it does not resize
or reinterpret any ellipsoid.

The test is passed only by a valid QP action whose complete cloned OSC
transition is raw-contact-free, moves the obstacle by at most `0.1 mm`, and
maintains the registered `8 mm` proxy margin.  Because job `37137` already
proved that the frozen obstacle MVEE misses the physical contact surface, even
a pass is only a single-state activation heuristic.  The exact MuJoCo
obstacle-box oracle remains the second, privileged geometry test and will be
run after this margin outcome is frozen.

Validated H100 job `37175` rejects the margin at the registered action-192
state.  It labels the nominal unsafe, but two rows begin below `8 mm`, no local
candidate can satisfy the minimum-substep margin, and OSQP returns primal
infeasible.  Since no QP proposal exists, collision avoidance is not
demonstrated.  This is retained as a late-activation result; the value is not
tuned after inspection.  The second exact-obstacle-box test may now proceed.

## ADR-0054: Use exact source boxes before declaring conservative late activation

Accepted from validated job `37163`.  The live primitive union repaired contact
authority, but the moka-pot collision model contains 15 exact boxes rather than
meshes.  Replacing each box with its single Loewner enclosing ellipsoid expands
all half-axes by `sqrt(3)` and produced up to `33.621 mm` negative clearance at
the immutable action-192 interval start.  No registered action could erase an
initial-state violation.

Before moving the trigger earlier, the next bounded test uses the exact live
oriented boxes for obstacle support and containment, retaining the robot
slabs, state, candidates, affine/QP settings, and decision rule.  An exact box
is already a conservative representation of its MuJoCo collision geom and has
less empty space than its ellipsoid enclosure.  If the exact-box barrier is
negative at action-192 start, late activation is established for this proxy and
an earlier-ledger audit may then be preregistered.  No neural training is
authorized.

## ADR-0053: Replace only the failed obstacle proxy with a certified live union

Accepted by user instruction after job `37137`.  The next E02 mechanism test
does not resize the accepted L5--L7 slabs or train a network.  It replaces the
single frozen perception MVEE with one conservative enclosing ellipsoid per
collision-active geom in the active obstacle lineage.  Mesh bounds carry exact
compiled-vertex enclosure certificates and follow their source geom's live
pose; non-mesh fallbacks require closed-form certificates.

The eight model outputs remain the minimum pair gap for each of the seven
distal slabs and released EE proxy.  This preserves the base affine/QP audit
while making each output conservative with respect to all registered obstacle
parts.  Exact contact-geom coverage is checked independently from optimizer
gap.  A negative interval-start gap is retained and interpreted as late
activation, not repaired post hoc.  No existing Table-1 or completed oracle
artifact is modified.

## ADR-0052: Repair the obstacle label before learning transition corrections

Accepted from the preregistered H100 outcome.  Job `37137` rejects the proposed
Physics NN when its target is the current AEGIS obstacle-MVEE support gap.  An
exact internal-substep trace reproduced 15 raw L6 contacts, but the certified
robot slab covered every contact point while the frozen perception MVEE covered
none.  The proxy gap therefore stayed positive, the oracle-affine QP treated
the colliding nominal action as safely feasible, and exact execution collided
again.  A learned map that accurately reproduces these proxy labels cannot
repair missing obstacle geometry.

Five registered local Cartesian candidates were raw-safe and proxy-safe, so
the experiment does not show uncontrollable dynamics.  It localizes the next
gate to the safety label: use a conservative union of obstacle collision-mesh
primitives, or explicitly learn a raw signed-distance/contact-risk residual
that covers the missing surface.  Re-run the unchanged oracle mechanism audit
and require geometry authority before any training, held-out evaluation, or
neural-versus-QP claim.  Do not add more L5--L7 robot ellipsoids to address this
specific failure; their registered L6 slab already covered the contact.

## ADR-0051: Test geometry authority and an oracle affine barrier before learning

Accepted by direct user instruction before outcome review.  The next bounded
`E02` experiment does not train a neural steering model.  It gives the proposed
Physics NN oracle access to exactly the quantities it would predict: the local
drift and XYZ sensitivity of every L5--L7/EE minimum-substep proxy barrier
under the complete OSC transition.

The experiment first audits the more fundamental assumption exposed by jobs
`37109` and `37114`.  It exactly replays the hash-bound executed actions 0--191
from completed job `37109`, then captures every internal MuJoCo state and raw
direct protected-link contact for action 192.  This is the registered
false-safe transition: all eight endpoint proxy clearances are positive while
raw MuJoCo reports direct L6 contact.  The original Table-1 artifact remains
the immutable initial-state, pairing, task, and frozen-perception authority.
A contact is consistent
with the registered geometry only when its position lies within both the
matching robot slab union and the frozen obstacle MVEE and the corresponding
support gap is nonpositive.  If this implication fails, the current barrier is
not an authoritative physical label and the transition-only residual story
stops before training.

If geometry is consistent, the registered cloned candidates define the exact
minimum-substep clearance map.  The best nominal-anchored affine coefficients
are fitted inside a fixed `L_infinity <= 0.5` action trust region, shifted by
their maximum one-sided candidate overprediction plus `1e-6 m`, and supplied
to one exact eight-constraint OSQP.  Its proposal is executed only in the
clone and checked against all substeps, raw contact, and `0.1 mm` within-step
obstacle motion.  A local GO requires geometry consistency, a jointly raw- and
proxy-safe local candidate, zero calibrated candidate false-safes, a feasible
QP, and a raw/proxy-safe exact QP transition.  No setting may be tuned after
outcome inspection.

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

Superseded by ADR-0047.  The paper's `103 ms` / `9.61 Hz` number is inference
throughput for one guided call containing all reverse denoising steps; it does
not establish a `10 Hz` simulator control protocol.  The v3 and AEGIS-EE v1
runs remain immutable historical evidence, but their rate must not be called
the paper's control rate.

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

## ADR-0046: Isolate the released AEGIS end-effector constraint

Accepted as the next bounded diagnostic.  L5/L6/L7 ellipsoids and their CBF
rows are disabled, not refit.  Both controller arms receive exactly the
released Table-1 end-effector proxy and archived E05 obstacle MVEE.  Reusing
the archived MVEE is valid here because the evaluator must reproduce its
settled simulator-state and camera hashes before control; the completed Table
1 result is never modified or reinterpreted.

For Cartesian EE, the released six-variable translational CBF-QP is executed
unchanged, including its virtual direction state and `0.05 s` internal update.
For Direct Joint, the same QP is evaluated on the joint trajectory's
equivalent Cartesian translation and only its translation correction is
lifted through the live damped EE Jacobian into the joint target.  This tests
whether changing the downstream controller representation helps the original
AEGIS constraint.  It is not a claim of an exact EmbodiSteer implementation,
native joint-space barrier, full-body protection, or matched paper baseline.
Raw simulator contacts and task state remain authoritative over QP clearance.

H100 attempt `37077` showed that exact physical-state equality does not imply
byte-identical RGB after that state is transplanted into a fresh 10 Hz render
context.  Requiring the new rendering to equal the old perception input was
therefore the wrong provenance check: the new run does not refit perception;
it consumes the archived numeric MVEE.  The revised fail-closed binding keeps
the exact 170-value simulator-state hash and checks the active obstacle world
position against the archived Table-1 value within `1e-9 m`.  Old and current
camera hashes are both retained, with mismatch expected rather than hidden.

The validated outcome from H100 job `37080` rejects both possible optimistic
interpretations.  The Cartesian EE arm completed the task, but L5/L6 contact
and CAR occurred while every single-row QP satisfied its optimizer
constraint.  At first protected contact, released barrier `h=-0.003259` and
the QP applied a `0.245590` action-unit translation correction; optimizer-row
feasibility was not physical-link safety.  Direct Joint contacted L7 at step
27 with `h=-0.054550`, later contacted L5/L6/L7, and never progressed the
native goal despite 245 nontrivial lifted corrections.

The accepted conclusion is limited but decisive for this diagnostic: changing
from OSC to the current Direct-Joint adapter cannot make an end-effector-only
barrier protect unmodeled links.  It also adds controller-tracking and
linearized-Jacobian error; its maximum lifted correction was `0.453 rad` and
mean target tracking error was `0.289 rad`.  No L5--L7 geometry was enabled,
so this run does not evaluate or rehabilitate the earlier distal ellipsoids.

## ADR-0047: Restore released AEGIS timing and classify Direct Joint correctly

Accepted after re-auditing the paper and validated H100 jobs `37083` and
`37084`.  The released AEGIS reproduction runs at `20 Hz`, executes five of
the ten predicted actions per query, and uses an internal QP step of `0.05 s`.
The corrected AEGIS-EE v2 protocol restores those values.  AEGIS-EE v1's
`10 Hz` simulator step was inconsistent with its unchanged `0.05 s` safety
model and is not the primary result.

EmbodiSteer's `Joint` baseline means joint-space denoising with FK calls to the
frozen Cartesian denoiser and damped-Jacobian residual updates, with collision
guidance disabled.  Our no-guidance arm implements that structural pattern and
the published heuristics `alpha=0.1`, pseudoinverse damping `0.001`, and joint
update clip `0.5 rad`.  It is still only an adaptation: pi0.5-LIBERO supplies
ten incremental 7D OSC flow actions rather than the paper's chunk-start pose
DDPM representation and the targets are executed through SafeLIBERO's
`JOINT_POSITION` controller.  The arm named `joint_denoising_with_aegis_ee`
adds a post-hoc EE correction and therefore is not the paper's `Joint`
baseline.

At restored timing, no-guidance job `37083` validated both 300-action videos;
neither arm completed the task.  Direct Joint contacted L5 at step 97 and had
mean/max target tracking error `0.218/0.978 rad`.  In job `37084`, Cartesian
AEGIS completed at step 231 but contacted L5 and failed CAR at step 196.
Direct Joint + AEGIS avoided L5--L7 contact in that rollout but contacted the
hand/finger at step 15, failed CAR at step 16, displaced the obstacle by
`0.034719 m`, and never completed the task; tracking error remained
`0.183/0.701 rad`.  Thus the timing repair does not establish a competent
Direct-Joint baseline or successful safety method.  The remaining mismatch is
architectural, not a missing scalar heuristic from the paper.

## ADR-0048: Retain the released EE proxy and use exact hull slabs for L5--L7

Accepted after hand review of the H100 geometry renders. The AEGIS
end-effector proxy remains the released site-attached ellipsoid with semiaxes
`[0.06, 0.12, 0.11] m`; changing it would confound the requested comparison.
The additional L5--L7 bounds are opt-in geometry and do not reinterpret or
modify the completed Table 1 results.

The common-apex facet partition from job `37086` is mathematically enclosing
but rejected as the final geometry because one L5 part remained nearly as long
as the original one-link MVEE. The replacement partitions the convex hull into
contiguous dominant-axis slabs and encloses each exact clipped polytope. This
gives a formal no-gap union certificate while allowing each ellipsoid to
remain local to one short section of the physical link.

H100 job `37087` passed the allocation numeric gate and produced the accepted
seven-part render: three L5, two L6, two L7, plus the unchanged cyan AEGIS EE
proxy. No active controller or QP is changed by this decision. Geometry must
be promoted under a separate preregistered control protocol before any safety
claim is evaluated.

## ADR-0049: Use exact cloned contact and obstacle motion as oracle authority

Accepted after clean H100 job `37115`. For the requested simplest ability
demonstration on primary case `vlsa-t1-goal-ii-t0-e05`, the ellipsoid
finite-difference multi-constraint QP remains a candidate generator and
activation heuristic, but its scalar center-direction `support_gap` is not
treated as a conservative collision certificate. Job `37109` demonstrated
physical L6 contact with positive proxy gap. Raw contact only at the end of a
step is also insufficient: job `37114` demonstrated transient contact through
`3.295 mm` obstacle motion even though the final cloned state was separated.

The oracle acceptance rule therefore executes each candidate in a complete
cloned OSC transition and requires both zero raw distal contact and at most
`0.1 mm` active-obstacle displacement for that step. A privileged replay of
the known successful immutable AEGIS action plan supplies the next end-effector
reference used to choose among accepted candidates. The executed simulator
then must match the accepted clone exactly.

This bounded heuristic solved the single case: task success at step 192, no
robot contact, no protected-link contact, no CAR, and only `0.0394 mm` maximum
obstacle displacement. It is accepted as proof that a verified correction
exists around the original successful motion. It is not accepted as proof
that the current ellipsoid QP alone is correct, as a deployable causal policy,
as population evidence, or as whole-arm safety. The next research question is
whether the privileged reference and expensive cloned search can be replaced
by a causal proposal model while retaining the exact simulator verification
gate.

## ADR-0050: Replace the damaged recording with an exact single-context replay

The striped job-`37115` MP4 is retained as immutable provenance but rejected
as visual evidence. Re-encoding cannot recover pixels corrupted during source
rendering. The replacement must therefore replay the already accepted action
ledger in exactly one freshly rendered MuJoCo environment on H100, without
the probe/reference environments used by the online candidate search.

The replacement is accepted only if its simulator trace exactly reproduces
the successful result and both source and decoded frames pass explicit
striping, degeneracy, frame-count, and codec-fidelity gates. This changes no
executed action or scientific outcome. Until that allocation-backed receipt
exists, the clean replacement video remains pending rather than inferred from
local rendering.

H100 job `37120` supplied that receipt. It reproduced all 193 accepted actions
with exact state/contact/task equivalence and passed both source and decoded
pixel-integrity gates. The clean single-context MP4 is therefore accepted as
the presentation artifact for the successful oracle run; the original striped
job-`37115` MP4 remains provenance only.
# ADR-0058: Gate boundary-supervised neural capacity at action 188 before generalization

**Status:** Same-state local capacity accepted; generalization pending (2026-08-09).

Validated job `37198` rejected nominal-trajectory supervision but retained 16
exactly safe local actions at action 188. The next diagnostic therefore keeps
the existing residual MLP and seven-row projection fixed while changing only
the supervision distribution. A deterministic 1,000-action trust-region grid
and 64 balanced boundary anchors provide exact cloned-OSC margins; 384 central
differences provide witness-audited gradients. Complete anchor groups remain
in one split. The margin-only and margin-plus-gradient arms are paired.

No model may train before the no-training dataset validator authorizes it.
Success is limited to same-state local capacity and requires better critical
near-boundary RMSE than the baseline, zero conservative false-safe candidates,
mean active-gradient cosine at least `0.8`, a valid seven-row QP, and an exact
proxy/raw-safe cloned transition. Multi-task generalization and closed-loop
task success remain later gates.

H100 job `37205` stopped after all cloned transitions but before dataset
serialization because the protocol incorrectly required a nonempty far-safe
stratum. The trust-region grid contains no critical-row action above +5 mm,
so the recoverable safe set is entirely boundary-safe. The apparatus repair
records all grid labels but excludes outside-band groups from learning; the
two boundary signs receive equal weight and retain grouped 70/15/15 splits.
No action, exact label, band, trust region, model, QP, or success threshold is
changed, and job `37205` supports no neural outcome.

Retry `37208` wrote all 1,384 records and confirmed 365 boundary-safe versus
635 boundary-unsafe grid actions. All 64 selected critical gradients were
witness-stable (41/11/12 train/validation/test). Training nevertheless stayed
blocked because the collector accidentally included an empty `excluded_far`
entry in the minimum stable-anchor count, while its independent validator
recomputed only the three learning splits and rejected the mismatch. The
repair swaps the affected summary comprehensions; it changes no recorded
transition or registered scientific setting. Job `37208` supports no neural
outcome.

Clean H100 dataset job `37209` passed the repaired no-training gate with 365
boundary-safe and 635 boundary-unsafe grid actions and 41/11/12 stable
critical anchors across train/validation/test. Clean paired job `37210` then
passed independent validation for both fixed-architecture arms. Held-out
critical RMSE was 0.01396/0.01440 mm for margin-only/gradient-supervised,
versus 3.70047 mm for the current-clearance baseline; both retained zero
conservative false-safe candidates and gradient cosine 0.99987/0.99997.

Both seven-row QPs solved and passed exact cloned OSC verification. The
gradient arm achieved minimum exact L5_part_1 clearance 1.391 mm, all seven
distal gaps nonnegative, zero raw distal contact, and zero obstacle motion;
QP and complete projection wall times were 1.344/5.549 ms. This accepts only
same-state local capacity for the execution-aware surrogate. It does not
accept generalization or closed-loop efficacy, and it does not show that
explicit gradient supervision is necessary because the margin-only arm also
passed with nearly identical accuracy and exact safety.

# ADR-0059: Require task-group generalization before neural closed-loop E05

**Status:** Completed NO-GO (2026-08-09).

The action-188 result establishes local representational capacity only. The
next experiment therefore freezes complete episode and task-level groups
before collecting any new labels. Primary E05 and its selected `goal-II-t0`
siblings are test-only. Training uses two different task groups; calibration
uses a third unseen task group. Outcome-conditioned episodes with protected
contact already present after settling are excluded because they cannot
provide a recoverable safe-to-unsafe transition.

The no-training gate requires an exact recoverable crossing and balanced
boundary supervision in every selected episode. The learned decision uses the
physically active minimum-clearance row per record rather than a fixed E05
row. Closed-loop E05 is ordered after, and conditional on, every held-out
episode passing model accuracy, zero conservative false-safe, active-gradient,
seven-row QP, and exact cloned OSC verification. This preserves the distinction
between one-step grouped generalization and later causal closed-loop task
completion.

H100 job `37211` exposed a constructor-only incompatibility before data
collection: one historical Table-1 PCA obstacle basis is reflective. The
exact-box probe never reads the geometry object's obstacle after construction,
so the accepted repair uses the proper primary-E05 MVEE solely to construct
the shared robot slab geometry. The live exact 15-box union remains the only
obstacle used by clearance and contact-witness evaluation. This is an
apparatus repair, not a representation or threshold change.

Retry `37212` was also rejected before a dataset because one selected E18
payload hash was transcribed incorrectly even though its raw-file and action
sequence hashes matched. The corrected value is taken directly from the
immutable Table-1 `result.json`; all ten selected receipt identities were
cross-checked before retry. This is provenance repair only and does not alter
case selection, grouping, actions, labels, or gates.

Clean job `37213` is the accepted v1 dataset decision: training is blocked
because only 3/10 first-crossing states supplied balanced derivative-eligible
anchors. This is evidence against first-crossing-only sampling, not against
the already accepted same-state model capacity.

V2 is a distinct preregistered follow-up rather than a reinterpretation of
v1. It may select one of the three immediately preceding states when the
first-crossing grid is unbalanced. It also permits derivative probes, but not
anchors or QP actions, to extend 0.02 beyond the anchor trust edge while
remaining inside global action bounds. The grouped split, exact labels,
network, calibration, projection trust region, and held-out gates remain
unchanged. Training is still forbidden unless every episode passes the new
no-training gate.

Clean H100 job `37214` is the accepted v2 dataset decision. It completed all
ten cases and passed the independent receipt/grouping validator, but the
dataset gate remains `NO-GO`: 7/10 episodes were eligible. The failed cases
were train `goal-II-t2-e15` and held-out test `goal-II-t0-e05/e10`; no failed
case is removed from the decision. E05 is the decisive mechanism failure. At
step 185 every registered candidate was more than 5 mm safe, while at step
186 every candidate in the nominal-centered L-infinity 0.5 action box was
unsafe. The nominal step-186 z command is -0.694, so that box admits only
z <= -0.194 and excludes stop/upward retreat. The one-step boundary dataset
therefore has no recoverable E05 action at the active state.

No neural model or closed-loop policy is run from job `37214`. The next
scientific choice must be preregistered, not tuned into this result: either
(a) retain a one-step model but allow the full physical action bounds / an
explicit stop-retreat recovery direction, or (b) predict a multi-step action
chunk so state 185 can be changed before the step-186 controllability loss.
The latter matches the execution-aware action-chunk draft more directly.
Separately, commit `0d7be4c` repairs the held-out projection evaluator so
each test environment replays its own immutable episode actions; E05 geometry
remains only the constructor placeholder. This defect did not affect job
`37214`, which was a no-training collection stage.

# ADR-0060: Diagnose full-bound E05 recovery before changing the learned method

**Status:** Completed partial GO (2026-08-09).

Job `37214` showed that the nominal-centered L-infinity 0.5 region excludes
stop/upward retreat at E05 action 186. The next experiment changes only the
candidate and QP action bounds to the physical normalized box `[-1,1]^3`.
It retains the immutable AEGIS input, accepted seven L5--L7 slabs, exact
15-box obstacle, zero-margin target, minimum-deviation objective, every-OSC-
substep measurement, raw contact veto, and 0.1 mm obstacle-motion veto.

The diagnostic reports a central-finite-difference seven-row QP separately
from a 9-by-9-by-9 exact cloned-simulator recovery lattice. The lattice is a
privileged discrete oracle, not the final method or an exact continuous
optimizer. An exact-safe candidate plus clone/execution identity authorizes
larger-region data collection. It does not authorize closed-loop E05. If no
full-bound candidate is safe, the one-step formulation is rejected in favor
of multi-step action-chunk prediction.

Clean H100 job `37226` completed the frozen diagnostic in `00:02:02` and its
independent validator passed. The finite-difference seven-row QP solved, but
its exact clone retained a -1.718 mm L5_part_1 gap. None of the 731 registered
full-bound candidates preserved all seven nonnegative proxy gaps. The best
candidate was the physical-bound corner `[-1,1,1]`, with L5_part_1 still
-0.907 mm; the three coordinate slices improved monotonically toward that
corner. Thus widening the action box at step 186 does not restore one-step
ellipsoid-set invariance.

All 731 candidates were raw-contact-free with zero obstacle motion. This does
not mean the proxy is irrelevant: action 186 is the pre-contact barrier
crossing, while raw L5 contact occurs later. It means the filter must act at
an earlier safe state using a horizon long enough to see the future crossing.
Larger-region one-step training and closed-loop E05 remain unauthorized. The
next method-level experiment is a two-step, then five-step, cloned-OSC action-
chunk clearance oracle starting at step 185; neural training remains after
that oracle gate.

# ADR-0061: Test paired two- and five-step E05 chunk recovery before training

**Status:** Decided; two-step partial GO and five-step NO-GO (2026-08-09).

The multi-step oracle begins at immutable action 185 and retains the accepted
seven slabs, exact 15-box obstacle, raw contact/motion authority, and released
AEGIS suffix. It keeps one-shot first-action editing distinct from distributed
residual velocity at horizons two and five. The distributed two-step weights
are `[1,-1]`; the five-step weights are `[1,0.5,0,-0.5,-1]`. Corrections are
constructed inside the jointly feasible action box without clipping and sum
to zero, so endpoint preservation is exact.

Each candidate synchronizes the complete simulator/controller clone once and
then advances the chunk sequentially through the real OSC. A candidate passes
only if every internal substep keeps all seven proxy gaps nonnegative, has no
raw L5--L7 contact, and moves the obstacle at most 0.1 mm per control step.
The smallest passing candidate is re-executed with per-step clone identity.
A pass authorizes grouped chunk-data collection, not neural training or
closed-loop E05. A failure rejects only these registered low-dimensional
families, not every possible action chunk.

Clean H100 job `37262` completed all 2,922 registered chunks in `00:21:44`
and passed independent validation. Both two-step arms passed: one-shot had
459/731 safe candidates and endpoint-preserving distributed residuals had
121/730. The minimum-correction winner was two-step one-shot, changing action
185 XYZ to `[0,0.5,-0.25]`; its worst L5_part_1 gap was +0.478 mm, with zero
raw contact, zero obstacle motion, and exact clone/execution hashes at both
steps. The smallest distributed safe correction also passed with +0.145 mm
and exact zero summed XYZ correction.

Neither registered five-step family passed. Even the best one-shot and
distributed chunks reached -15.128/-15.036 mm and every five-step candidate
had raw protected contact by chunk steps three to four. This rejects a single
first-action edit and the frozen one-parameter taper for open-loop execution
of the entire five-action suffix. It does not reject multi-step prediction.
The evidence supports a receding two-step execution-margin filter that
re-evaluates at each simulator step while consuming the buffered pi0.5 chunk.
Grouped chunk-boundary collection is authorized; neural training and
closed-loop E05 remain later gates.

# ADR-0062: Compare global and factorized two-step margin models before E05

**Status:** Decided; current simplest paired method is NO-GO (2026-08-09).

Job `37262` proves that the primary action-185 state has control authority in
the two-step one-shot family, while job `37226` proves that waiting until
action 186 is too late. The next learned experiment therefore keeps the
second released-AEGIS command immutable and varies only the first XYZ command.
This preserves a three-dimensional projection and directly tests receding
two-step lookahead without introducing a six-dimensional chunk optimizer.

The previous ten-case grouped split and exact 15-box obstacle authority are
retained. A two-stage protocol prevents training unless all episodes provide
a balanced exact two-step boundary and witness-stable derivative supervision.
E05 and its complete task-level group remain test-only. D_opt is the minimum
seven-row exact-box clearance over both OSC transitions and all substeps;
D_sim raw contact and obstacle motion remain separate acceptance authorities.

The paired arms isolate representation. The `global` arm extends the earlier
monolithic execution-loss MLP with the second action. The `factorized` arm
uses a shared scalar MLP per constraint with relative robot--obstacle geometry.
Both use the same samples, Huber and gradient loss, normalization source,
validation-only conservative calibration, and bounded seven-row QP.

Closed-loop E05 is ordered after every held-out factorized model and exact
projection gate. It recomputes the learned model after every executed action
on the immutable successful AEGIS nominal sequence, executes only the first
correction, and does not consult a cloned candidate oracle online. This is a
fast replay-level feasibility diagnostic, not yet a live-policy or population
claim. Failure at the data, generalization, projection, collision, CAR, or task
gate is retained as a research-direction NO-GO rather than tuned away.

H100 job `37270` passed the data gate with all 10/10 episodes eligible and
4,938 learning records, so the negative result is not caused by a missing
two-step oracle boundary. Paired job `37273` found that both models improve
held-out margin RMSE and have zero conservative false-safe candidates, but the
factorized gradient cosine mean/minimum was only 0.793/0.777 on E05,
0.674/0.576 on E10, and 0.783/0.644 on E15. It therefore failed the frozen
0.8 gate on every complete test episode.

Validation-only row calibration reached 7.234--19.679 mm for the factorized
arm. With the learned local gradients and registered action bounds, all three
held-out seven-row QPs were correctly reported primal infeasible. The global
arm was also infeasible in all three cases. Thus closed-loop E05 was not run
and no D_sim or task-success claim was created. The supported interpretation
is narrower than rejecting the research direction: value prediction
generalizes better than the no-motion baseline, but this simplest global
calibration plus local-gradient QP does not reproduce the two-step oracle on
unseen state groups. Any follow-up must be separately preregistered and target
conditional uncertainty/gradient generalization rather than merely train the
same model longer or relax the gate post hoc.

# ADR-0063: Test the affine safe-set representation with a privileged oracle

**Status:** Decided; representation GO, coefficient collection authorized (2026-08-09).

Job `37270` shows that exact two-step safe actions exist in every selected
episode, while job `37273` shows that the residual MLP's unseen-state values
are useful but its gradients and global calibration make all held-out QPs
infeasible. Before collecting more states or training a different network, the
next gate isolates whether one seven-row affine safe set is expressive enough.

The oracle uses the immutable 512-action exact grids for held-out E05/E10/E15.
For each candidate, every row is a minimum-L1-gradient affine lower envelope
over the complete sampled grid with 1 micrometre one-sided padding. Candidates
are tried in a fixed minimum-correction/index order. The resulting seven rows
must make the ordinary bounded QP valid, and the QP's own continuous solution
must pass a fresh two-step exact OSC rollout. The original released-AEGIS EE
proxy is retained as an exact eighth-row outcome gate rather than learned or
reinterpreted.

All three cases must pass for `representation_go=true`. That outcome only
authorizes a new grouped state-to-coefficient dataset gate. It does not
authorize a neural model, closed-loop E05, or a safety claim. A failure is a
NO-GO for this registered single-affine sampled-grid construction; it does not
exclude piecewise-affine or nonlinear safe sets. This ordering prevents a
larger model or relaxed calibration from hiding a representation failure.

Attempt `37277` is an apparatus failure: its allocation numerical tests
passed, but a `1e-12` m replay-clearance equality receipt rejected the first
case before output. The fixed repair records and bounds that pairing
difference by `1e-8` m, still 100 times below the scientific affine padding.
No optimization or acceptance setting changes.

Retry `37278` exposed a 0.660 mm difference and is also an apparatus failure;
the receipt is not relaxed. The verifier must reproduce the dataset
collector's first-crossing probes and controller-inclusive snapshot/restore
before evaluating the QP action. It must match both the recorded crossing step
and selected-state clearance within the existing `1e-8` m receipt.

Clean H100 job `37280` reconstructed every registered crossing/state exactly
and passed independent validation. E05/E10/E15 each admitted a seven-row
affine certificate and valid QP whose fresh exact two-step distal minimum was
0.736/0.505/0.826 mm. The unchanged released-AEGIS EE proxy retained
8.439/27.612/13.172 mm; raw contact and obstacle motion were zero. Therefore
the QP representation is not the limiting mechanism observed in job `37273`.

The next research object is a state-conditioned predictor for these oracle
affine coefficients and a conditional conservative residual, trained on many
distinct state groups rather than dense actions from only a few states. The
current GO authorizes target collection only. A separately validated grouped
dataset is required before training, and closed-loop E05 remains a later gate.

# ADR-0064: Run a privileged receding affine-oracle E05 efficacy diagnostic

**Status:** Decided; preregistered before execution (2026-08-09).

The isolated-state affine representation gate passed on E05/E10/E15, while
the grouped residual MLP did not provide a feasible held-out QP. Before
collecting affine-coefficient targets or training another model, the user
authorized one bounded privileged experiment to test whether the target
mechanism itself can solve the complete primary episode.

At every executed E05 state, a cloned OSC rollout checks the next two immutable
released-AEGIS actions. Unsafe horizons trigger the same 512-action trust
region, seven candidate-conditioned lower envelopes, and one seven-row
minimum-deviation QP validated in job `37280`. The released AEGIS EE proxy
remains unchanged and is exact-gated; it is not added as an eighth QP row.
Only the freshly verified first action executes, after which the oracle is
recomputed. A certificate, QP, exact-gate, or clone-hash failure stops without
executing a fallback.

Passing requires nonnegative executed all-eight proxy margins, zero protected
contact, zero paper CAR, exact clone/execution state agreement, and native task
success. Actual execution is recorded to MP4/JPG. This can establish a
privileged target for later learning, but cannot establish learned
generalization, deployability, or formal whole-body safety. Neural training
remains blocked regardless of the result.

Clean H100 job `37283` validated a scientific NO-GO for the seven-row version.
It executed 15 actions with nonnegative all-eight margins, zero protected
contact/CAR, and exact clone hashes, then failed closed at action 15. The
nominal two-step EE margin was `-1.535 mm`, while all distal margins exceeded
`90.866 mm` and the grid contained 201 all-eight-safe candidates. Because the
EE row was only a veto, all seven distal certificate gradients collapsed to
zero and the seven-row QP returned the nominal action. This is neither an OSC
nor an OSQP failure; it is the registered constraint omission.

# ADR-0065: Add the released-AEGIS EE proxy as the eighth affine QP row

**Status:** Decided; preregistered before execution (2026-08-09).

V2 changes only the observed missing constraint. The exact released-AEGIS EE
margin is fitted by the same candidate-conditioned lower-envelope procedure
and becomes the eighth hard QP row. Geometry, grid, bounds, padding, objective,
nominal sequence, receding horizon, exact veto, clone receipts, and success
criteria are unchanged. This does not discard job `37283` or relax safety; it
tests whether the registered all-eight safe candidates can be reached when the
optimizer is actually informed about all eight required constraints.

Clean H100 job `37285` validated V2 as a scientific NO-GO. The eighth row
repaired the early omission and produced seven exactly safe interventions,
including action 185. At action 186, one of 512 grid actions remained
all-eight proxy/raw safe, but no affine lower envelope for active L5-part-1
could both lower-bound the complete grid and certify that isolated action.
The method stopped before execution with zero protected contact/CAR after 186
exactly clone-matched actions. This supports a nonempty but nonconvex or too
narrow sampled recovery set, not a missing-action or simulator-model failure.

# ADR-0066: Replace the corrupted multi-context video with a single-context replay

**Status:** Decided; evidence-only repair preregistered (2026-08-09).

The job-`37285` MP4 decodes but its frames are visibly striped because the
evaluation process created main and probe OSMesa render contexts together.
The scientific state/action artifact remains independently valid. Visual
evidence must be regenerated by replaying the immutable 186 executed actions
in one fresh rendered environment on H100. Every dynamic-state hash, reward,
done flag, obstacle displacement, and pairing receipt must match the accepted
result. Source and decoded frames additionally fail closed on adjacent-pixel
MAD above 12/255. No action, safety decision, or scientific outcome changes.

Clean H100 replay job `37289` verified all 186 dynamic-state hashes and all
187 frames in one render context. Six distributed decoded samples had maximum
adjacent-pixel MAD `2.088/255` and maximum encoder MAE `1.592/255`; the final
JPG is visually clean. This replacement is the sole accepted visual evidence
for job `37285`; the original multi-context MP4/JPG remain provenance-only.

# ADR-0067: Test the disconnected safe component with direct exact selection

**Status:** Decided; preregistered before execution (2026-08-09).

Job `37285` proved that one of 512 actions at step 186 was exactly all-eight
proxy/raw safe, but a single affine lower envelope over the complete grid could
not certify that isolated component. The user authorized a separate privileged
upper-bound diagnostic: select the closest exactly safe sampled action
directly, execute only its freshly verified first transition, and recompute at
the next state.

The grid, trust region, two-action horizon, immutable nominal plan, L5--L7
slabs, released AEGIS EE proxy, raw contact/obstacle-motion gates, and clone
receipts remain fixed. A 1 micrometre candidate margin is required. No affine
fit, QP, learned model, or fallback is permitted. A GO shows only that this
zeroth-order sampled receding controller can solve primary E05; a NO-GO is
limited to the registered search family. Image observables are disabled in the
scientific run and visual evidence must be regenerated through one exact
single-context replay.

Clean H100 job `37294` validated the registered method as NO-GO. It safely and
exactly executed ten selected corrections through action 186, but action 187
had zero eligible candidates among all 512 samples. The controller stopped
with no contact or CAR and with minimum executed distal/EE margins
`15.371/0.183 mm`, but native task progress remained zero. Therefore the
isolated safe component at the prior action is not a sufficient receding
solution; it only shifts the unrecoverable boundary forward one action. No QP
or learned-model efficacy follows from this result.

The scientific run intentionally has no renderer. Visual evidence is a
separate H100 replay using one OSMesa context, exact accepted-action/state
receipts, and corruption gates. It is encoded at 10 fps for presentation only,
yielding approximately 18.8 seconds for 188 frames while retaining one frame
per executed simulator action.

Clean H100 replay job `37299` verified every accepted transition and all 188
frames with zero state/obstacle-trace error. Maximum decoded adjacent-pixel MAD
and encoder MAE were `2.082/1.476` of 255, and direct inspection found the
terminal frame clean. This replay is the accepted visual evidence for job
`37294`; it does not alter the scientific NO-GO.

# ADR-0068: Continue nominally after the empty safe set

**Status:** Decided; preregistered before execution (2026-08-09).

At the user's request, a separate diagnostic removes the fail-closed stop. It
does not weaken or reinterpret job `37294`. The accepted actions 0--186 are
replayed with exact state receipts; from action 187 onward the safety filter is
latched off and the immutable AEGIS nominal suffix executes until task success
or action-horizon exhaustion. Candidate search and QP are disabled, and proxy
violations, raw contacts, and CAR are measured but do not terminate execution.

Because an empty safe set is knowingly bypassed, `safety_success` is always
false. The arm answers only whether continuing restores task completion and
when physical collision occurs.

Clean H100 job `37301` executed the full suffix and validated the answer:
proxy violation and protected L5 contact began at action 188, paper CAR began
at 189, maximum obstacle displacement reached `24.554 mm`, and the task still
had no predicate progress after action 236. Removing the stop therefore
converts the safe NO-GO into collision without recovering task success. A
single-context 20 fps replay is required for the final visual evidence.

Clean H100 replay job `37303` exactly matched all 237 action/state/obstacle
receipts and verified 238 frames at native 20 fps. Maximum adjacent-pixel MAD
and encoder MAE were `1.997/1.586` of 255, and the final frame is visually
clean. This is the accepted visual record of the collision-prone continuation.

# ADR-0069: Test coordinated Cartesian waypoint Best-of-N

**Status:** Decided; preregistered before execution (2026-08-09).

Accepted by direct user instruction after validated exact-candidate job
`37294`. The next `E02` diagnostic does not enlarge or reinterpret that
one-step result. It freezes four-action Cartesian waypoint chunks: constant
full-lattice commands, two-phase cardinal turns, and one-/two-step diversions
followed by the immutable nominal suffix. Every orientation and gripper
channel remains unchanged.

All eight exact proxy rows, raw protected contact, per-step obstacle motion,
and clone/execution equality remain hard gates. Candidate selection uses a
parallel read-only successful AEGIS replay only to provide the future EE
waypoint, with task completion ranked first. This is deliberately privileged
mechanism evidence: a positive result establishes that coordinated multi-step
control can solve E05, not that a learned or live VLA method supplies the
waypoint. A negative result rejects only this registered library. No QP or
neural model is called and neural training remains unauthorized.

Clean H100 job `37308` returned the informative mixed result. All 237 actions
executed with 16 verified interventions, every hard proxy/raw/clone gate
passed, and there was no contact or CAR. However, native task success remained
false: the final bowl-to-plate XY center error was `83.5 mm`, while the
successful reference ended at `26.7 mm`. The registered safe candidates
tracked the future EE only to `10.8--32.3 mm`, which is too coarse for the
placement predicate. Therefore waypoint Best-of-N solves the collision
avoidance subproblem but not the combined safety-and-task problem. The next
method must use state-conditioned live task proposals or object-aware terminal
ranking, not merely a larger safety margin or another local QP. This result
does not authorize neural training.

Clean H100 replay job `37319` exactly reproduced all 237 actions and state /
obstacle receipts in one render context. Its native-rate 20 fps video passes
source and decoded pixel-integrity gates and visually confirms the scientific
interpretation: the obstacle remains undisturbed, while the bowl ends outside
the plate. This replay is presentation evidence only and does not change the
validated safety-GO/task-NO-GO result.

# ADR-0070: Isolate object-aware local suffix refinement

**Status:** Decided; preregistered before execution (2026-08-09).

Job `37308` proves that a collision-free coordinated trajectory exists but its
future-EE-only coarse ranking misses placement precision. Before adding live
pi0.5 replanning, the next `E02` test isolates that single proposed repair.
Actions 0--229 must replay the validated job-`37308` trajectory with exact
state hashes. Only actions 230--236 may change.

The same two-action activation, four-action coarse chunks, exact OSC substeps,
seven distal rows, released AEGIS EE row, raw-contact veto, obstacle-motion
gate, and execute-first/replan rule remain. Candidate ranking changes to the
bowl pose relative to the plate from a read-only successful replay. Two local
first-action refinements, `0.25` then `0.10`, search all 26 ternary directions
around the four best safe anchors. Native task success outranks pose error and
safety is never traded against task score.

A GO establishes only that a locally refined object-aware safe suffix exists
from the validated action-230 state. A NO-GO rejects this fixed seven-action
suffix family, not live-policy or earlier-state recovery. No QP or learned
model is used and neural training remains unauthorized.

Initial H100 attempt `37333` failed before suffix search because its prefix
receipt applied `array_sha256` to the flattened MuJoCo state rather than the
controller-inclusive dynamic-state vector used by the accepted job-`37308`
ledger. The repair uses the identical `_dynamic_state_vector` hash contract.
The failed attempt is retained as apparatus evidence and changes no scientific
setting.

Retry `37335` was rejected by the clean-source preflight before Python startup
because the submission supplied an incorrectly expanded commit hash. It is a
submission-only apparatus failure with no effect on the registered protocol.

Clean H100 job `37337` validated the registered suffix as NO-GO. Its 230-action
prefix and all seven suffix executions matched exact state receipts; all eight
proxy margins, raw contact, and CAR gates passed. Three object-aware searches
evaluated 702 candidates and improved terminal bowl-to-plate XY error from
`92.1 mm` at the suffix start to `66.5 mm`, but native task success remained
false. Thus object-aware ranking helps placement relative to future-EE-only
ranking, but beginning at action 230 and refining only the first action of each
short chunk cannot recover enough task precision. The next justified arm must
introduce object-conditioned proposals earlier (for example live pi0.5
replanning with exact safety verification); merely densifying this terminal
local search is not supported. Neural training remains unauthorized.

Visual evidence for job `37337` must use a separate single-context H100 replay.
It concatenates only the accepted job-`37308` prefix and job-`37337` suffix,
requires every dynamic-state and obstacle receipt, and cannot alter the NO-GO.
Initial replay `37349` failed before its first frame on the result schema's
nested case identity. Reading the frozen `config.primary_case.case_id` is an
evidence-only apparatus repair.

Clean replay `37355` matched every accepted state/obstacle receipt and passed
all source and decoded pixel-integrity checks for 238 frames at 20 fps. Direct
inspection confirms a clean final view with the bowl outside the plate. This is
the accepted visual evidence and does not alter job-`37337` NO-GO.

# ADR-0071: Separate exact-contact ranking from neural gradient steering

**Status:** Decided; preregistered before execution (2026-08-09).

The installed MuJoCo binding does not expose a continuous arbitrary-mesh
distance query, so the first neural feasibility test must not describe an
ellipsoid proxy or contact logit as exact clearance. It instead learns three
binary labels from all internal OSC substeps: contact of L5, L6, or L7 with
the active obstacle. A 0.1 mm obstacle-motion veto remains part of fresh
`D_sim` verification, while seven-ellipsoid/exact-box support gaps remain
separate `D_opt` diagnostics.

The same model is audited in two modes. Conservative Best-of-N selection asks
whether it can rank a finite registered candidate set and choose a minimally
changed exact-safe action on held-out states. Negative-gradient steering asks
the stronger question needed by the proposed differentiable residual. A
ranking GO with gradient NO-GO supports candidate selection only; it cannot
support the draft's gradient-guidance equation. Neither arm authorizes a
closed-loop or generalization claim.

# ADR-0072: Gate ellipsoid learning on nonempty ellipsoid-safe support

**Status:** Decided; supersedes the unexecuted contact-label training arm at
the user's direction (2026-08-09).

The immediate experiment uses the accepted seven L5--L7 ellipsoids and the
unchanged released-AEGIS obstacle MVEE only. Exact raw contact remains a
diagnostic, not a training target or feasibility condition. Before fitting an
MLP, the registered candidate family must contain at least one action whose
seven minimum substep margins are all nonnegative at each primary failure
state. If this oracle support gate fails, an MLP cannot correctly select an
ellipsoid-safe action from that family, regardless of optimizer or network
capacity, so training stops.

Job `37407` validated this gate as NO-GO. The safe set disappeared at state
186 and stayed empty through 192; even the best candidates at the two primary
test states retained `-11.651 mm` and `-12.602 mm` worst margins. The active
rows were `L5_part_1`, with `L6_part_0` also negative. Consequently no
ellipsoid-margin MLP is trained on this suffix. A future learned experiment
must either start before the first ellipsoid violation, define an explicitly
recovering barrier for already-negative states, or change the registered
geometry/candidate family. It may not report this empty-set result as a neural
optimization failure.

# ADR-0073: Use ellipsoid features but raw contact supervision

**Status:** Decided; preregistered before V2 execution (2026-08-09).

The hard ellipsoid safe set is empty at the primary states, but exact substep
diagnostics still contain 34 and 33 zero-contact actions. Therefore the next
local feasibility test must not train on the impossible hard-ellipsoid target.
It retains the ellipsoid geometry as state information and learns three binary
execution-contact risks without calculating mesh distance.

Candidate ranking and gradient steering are distinct gates. Ranking GO alone
supports an MLP-assisted Best-of-N filter. Only a separate gradient GO
supports the proposed differentiable residual story. Both are local
same-episode mechanism results; neither authorizes closed-loop success or
generalization claims.

Job `37416` supports only the gradient-mechanism half of this decision. The
learned negative-risk direction yielded freshly contact-free actions at both
held-out states, but the calibrated classifier produced 52 test false-safes
and selected an unsafe nominal action at state 190. Therefore an MLP gradient
may be a useful proposal generator, but this model cannot be the safety
authority or Best-of-N filter. Exact verification remains necessary, and
closed-loop use is not authorized. Because all positive training labels were
L5, the result also provides no learned L6/L7 evidence.

# ADR-0074: Require a matched-random directional advantage

**Status:** Decided; preregistered before execution (2026-08-10).

Two safe learned-gradient transitions are insufficient evidence because the
local action space may contain many safe directions. The learned direction is
therefore compared with 256 uniformly sampled, action-bound-feasible unit
directions per state at identical correction radii. Reusing directions across
radii permits a first-safe-radius comparison rather than four unrelated
Bernoulli tests.

Only an add-one empirical equal-or-earlier-safe value at most 0.05 on both
held-out states authorizes boundary-focused multi-state/L6--L7 data
refinement. Failure requires rethinking supervision or the model before any
closed-loop test. Task error remains diagnostic and cannot compensate for
contact.

H100 job `37427` resolves this decision as NO-GO for the current model. The
values were `0.2218` at state 187 and `0.4786` at state 190, so neither state
passed. The current binary-contact classifier's action gradient is not shown
to contain more steering information than a random direction. More examples
under the same pointwise loss are therefore not the next experiment.

The next admissible model test must explicitly supervise local direction:
paired opposite perturbations at matched radii, exact controller-rollout
ordering labels, and a directional-ranking loss evaluated on held-out pairs.
It must again beat the matched-random gate before multi-state/L6--L7 scaling
or closed-loop E05 is considered.

# ADR-0075: Audit gradient implementation before changing supervision

**Status:** Decided; preregistered before execution (2026-08-10).

Failure against random directions does not by itself distinguish a weak
objective from a sign, normalization, or action-indexing defect. The immutable
job-`37416` model must therefore first satisfy a symmetric local identity:
its maximum contact probability is lower at `u-epsilon*g_hat` than at
`u+epsilon*g_hat`, and its smallest-step central derivative matches the
autograd norm. Both commands are then compared under paired cloned OSC
transitions.

An internal failure sends the work back to implementation repair. Internal
consistency with simulator preference for the positive-gradient command is
direct supervision/model misalignment. Internal consistency without such a
disagreement still does not rehabilitate the gradient, because job `37427`
already showed that it fails to beat random directions. No outcome of this
audit authorizes closed-loop execution.

Job `37441` passed the implementation audit. Both states passed every internal
ordering/indexing check, the finite-difference/autograd errors were below
`1.4e-9`, and cloned OSC preferred the negative sign in all ten paired tests.
There is therefore no evidence for a sign, normalization, or XYZ-indexing bug.
Because all radii through `0.1` remained contacting and job `37427` showed
many equally effective random directions at recovery radii, the retained
diagnosis is insufficient directional discrimination from the pointwise BCE
objective. The next model may add paired directional-ranking supervision;
closed-loop remains blocked.

# ADR-0076: Add directional ranking without changing the contact model family

**Status:** Decided; preregistered before execution (2026-08-10).

The first refinement reuses symmetric actions that were already executed by
the cloned OSC dataset rather than collecting a broader population or changing
geometry. Weighted link-contact BCE remains, while an auxiliary hinge loss
requires the lower-contact member of each opposite pair to receive lower
maximum contact risk. Complete state groups and the test holdout are preserved.

The revised gradient is judged at radius 0.1, where job `37427` found no random
collision-free direction, using the same 256 random directions and a finer
lexicographic raw-safety/contact-count/penetration ordering. This tests whether
the objective learned an exceptional local improvement rather than benefiting
from large corrections. Passing does not authorize closed-loop execution; it
only supports scaling directional supervision to broader state/link data.

Job `37463` returns a partial but overall NO-GO. The model ordered every
informative finite pair correctly, including 30/30 held-out pairs. Its actual
nominal gradient was exceptional at state 187 (`p=0.00778`) but not at state
190 (`p=0.16342`), and neither radius-0.1 action was contact-free. Therefore
finite endpoint ranking is not sufficient evidence for the derivative used by
control. The preregistered requirement was both unseen states, so no closed-loop
or broader-data continuation is permitted.

The next defensible model change, if separately authorized, is direct dense
small-radius derivative supervision or regression of a continuous executed
contact burden. Retuning the ranking weight or training longer on the same 28
informative training pairs would not answer the observed generalization gap.

# ADR-0077: Replace contact logits with direct signed rollout margins

**Status:** Decided; preregistered before execution (2026-08-10).

The binary-contact gradient implementation is internally correct, but jobs
`37427` and `37463` show that its learned field is not a reliable recovery
direction. The next experiment therefore changes the target, not the gradient
operator. It directly regresses the seven signed minimum controller-rollout
margins from the validated job-`37270` exact two-step dataset.

The primary E05 state remains entirely held out. Six complete episodes train
the model, a separate task episode calibrates conservative lower margins, and
the complete selected `goal-II-t0` group remains test-only. The shared
factorized network uses smooth Softplus hidden activations and a linear output
because margins can be positive or negative. Value loss is supplemented by
both witness-stable finite-difference direction and magnitude supervision;
cosine alone is not sufficient for a projection Jacobian.

At E05 action 185, the smallest predicted margin supplies the action gradient.
Its exact two-step margin improvement at radius `0.1` is compared with 256
uniform feasible directions at the same norm. A separate calibrated seven-row
QP must pass fresh exact proxy and raw-simulator verification. All model,
false-safe, gradient, matched-random, and QP gates must pass for a mechanism
GO. No outcome authorizes closed-loop execution or task-success claims; a
failure distinguishes grouped margin/gradient generalization from the already
established same-state capacity result.

Clean H100 job `37508` resolves this decision as NO-GO for the registered
model. The direct target reduced E05 boundary RMSE from the `29.279 mm`
no-motion baseline to `23.076 mm`, but its gradient cosine was only `0.643`,
its magnitude was severely underestimated, and validation calibration left
`783` held-out false-safe candidates. At the decisive nominal state it
predicted `21.803--33.409 mm` positive margins while exact L5-part-1 was
`-3.442 mm`.

The learned radius-0.1 direction did improve exact clearance by `0.741 mm`,
but 43/256 random directions did at least as well (`p=0.1712`). Both raw and
calibrated predictors therefore accepted the unchanged unsafe nominal chunk
and never invoked OSQP. This rules out the simple explanation that binary
labels alone caused the previous failure. The gradient operator remains
correct, and continuous supervision contains some directional signal, but
cross-state value, derivative magnitude, and conservative-error prediction
remain inadequate.

Do not run closed-loop E05 or train this same absolute-margin model longer.
The next defensible learned object is the state-conditioned local affine
coefficient target already shown representable by privileged job `37280`,
together with conditional lower-bound residuals and broader distinct-state
coverage. That requires a separate preregistered dataset gate.

# ADR-0078: Learn state-conditioned affine QP rows, gated by exact targets

**Status:** Decided; preregistered before execution (2026-08-10).

Job `37508` did not implicate OSQP: its learned field falsely declared the
unsafe E05 nominal chunk safe and supplied no active correction row. The next
test therefore predicts the local affine constraint consumed by the QP rather
than differentiating another global absolute-margin approximation.

The target is `(b, a, e)` per protected row, where `b` is the exact nominal
two-step margin, `a` is the candidate-conditioned conservative affine
coefficient, and `e = b - intercept` is the state-conditioned one-sided
error. A separate validation-group overprediction calibration remains because
the neural coefficient prediction itself is uncertain.

Target collection expands state coverage, not merely action count at one
state: five registered states from each of ten complete episodes, with the
same task-group train/validation/test separation and E05 test-only. All 50
states must have exact safe support, a seven-row lower-envelope certificate,
a valid QP, and a fresh safe cloned-OSC QP transition before training is
authorized. This strong no-training gate prevents a neural failure from being
confounded with nonexistent or invalid targets.

Passing the later learned test would establish only multi-state two-step
mechanism feasibility. Closed-loop task completion remains a separate future
gate.

H100 attempt `37554` exposed a target-construction omission before producing a
scientific artifact: the clipped five-point grid need not contain the nominal
action, so its sampled lower envelope need not lower-bound the separately
measured nominal margin. The correction explicitly anchors every certificate
with that nominal rollout. Clamping a negative `e` would hide the violated
lower-bound assumption and is rejected.

The repaired strict job `37580` remains a dataset NO-GO. Its failure is not
treated as evidence that coefficient targets are absent: all 50 certificates
are valid and all have exact safe support, while ten failures come only from
the separately released AEGIS EE proxy and one from OSQP reaching its
iteration cap despite a certified safe candidate. All 15 held-out test states
pass the full original gate.

The next action is therefore a separately registered, post-outcome target
analysis—not a retroactive pass. Training may use the immutable dataset only
if this new audit validates all 50 seven-row targets without filtering. The
learned test still requires zero held-out false-safes, gradient cosine at
least 0.8, and exact all-eight safe QPs on every held-out state.

The scoped target audit passed in H100 job `37621`. Learned attempt `37627`
then failed before a checkpoint because the pinned evaluation PyTorch has no
H100 `sm90` kernels. The apparatus-only repair trains this small deterministic
network on CPU inside the H100 allocation; it changes no scientific setting
or simulation authority.

Clean learned retry `37630` resolves the current formulation as NO-GO. All 50
targets exist, but the state-to-coefficient MLP generalizes poorly: held-out
gradient cosine is `0.5818` and 48 E15 candidates are falsely declared safe.
The metric gate therefore prevented learned QP simulation.

The retained diagnosis is target identifiability, not optimizer failure. The
candidate-conditioned minimum-L1 envelope is a region-dependent LP selection,
not a unique physical Jacobian; inactive rows commonly collapse to zero and
the selected affine region can switch across nearby states. Do not add epochs
or run closed-loop with this model. Any continuation must canonicalize the
target or explicitly model the active region, then repeat the same held-out
false-safe and gradient gates.

# ADR-0079: Test the affine representation before choosing another neural target

**Status:** Decided; preregistered before execution (2026-08-10).

Job `37630` cannot distinguish whether the MLP failed because it regressed a
non-unique minimum-L1 coefficient selection or because a single local affine
row is intrinsically too coarse over the registered action box. The next test
therefore contains no training and uses fresh exact simulator labels.

At every immutable state, one arm fits each L5--L7 row directly from binary
safe/unsafe two-step OSC labels and moves its half-space threshold beyond all
unsafe fitted actions. The other arm uses the exact nominal margin, exact
central finite-difference coefficient, and a grid-calibrated one-sided error.
The same 4,800 new action queries evaluate both arms, preserving state,
second-action horizon, and simulation pairing.

An oracle is useful only if it produces zero false-safe actions while retaining
at least 20% of exact safe actions and accepting a safe action in at least 80%
of states with fresh safe support. Passing either arm supports a later direct
half-space/one-sided loss with held-out tightening. Failing both redirects the
method to smaller trust regions or multiple affine regions. OSQP is outside
this diagnostic because it cannot repair a false safe-set representation.

Clean H100 job `37649` resolves this comparison. The direct label-only
half-space fails with 30 false-safe fresh actions. The canonical finite-
difference lower affine bound passes with zero false-safes, 93.34% safe recall,
and 48/50 safe-support state coverage. Thus one local affine row per distal
constraint is supported on this sampled action family, while the old
minimum-L1 coefficient regression target is rejected.

The next neural object should be judged through its induced half-space values,
not coefficient equality to an arbitrary LP optimum. Use canonical finite-
difference anchoring, action-level one-sided lower-bound/safe-unsafe losses,
and held-out state-group tightening. Do not run the QP or closed-loop until
that learned field independently reproduces zero false-safe and useful-recall
behavior. This sampled GO is not a formal continuum guarantee.

# ADR-0080: Canonicalize the affine target with boundary-weighted ridge-Huber

**Status:** Decided; preregistered before execution (2026-08-10).

The finite-difference result in job `37649` supports the affine representation
but missed all safe actions in two low-support states. Before learning again,
the gradient target must be both conservative and stable under changes in the
fitted candidate set.

The new target fixes the nominal value to exact `m(u0)` and estimates only the
three action coefficients. Boundary-weighted Huber regression limits the
effect of far/nonlinear candidates, while ridge regularization selects a
unique coefficient vector. A state-specific maximum-overestimate error then
converts the fitted plane to a sampled lower bound.

Stability is not inferred from fit loss. Ridge-Huber and minimum-L1 are
refitted on the same 16 deterministic 80% subsets. Active rows must preserve
cosine at least 0.9 and relative norm within 0.25. Both targets use identical
off-grid exact labels and new QP rollouts; failures and infeasible states stay
in the population.

No MLP is trained unless ridge-Huber passes every oracle gate. A later model,
if authorized, combines coefficient loss with action-level one-sided false-
safe loss and grouped validation tightening. Closed-loop E05 is a separate
dependent gate.

Clean H100 job `37688` resolves the ridge-Huber oracle as NO-GO under the
registered all-state gate. Ridge-Huber stabilized all 41 active rows and
raised safe recall from minimum-L1's 87.86% to 95.76%, with zero false-safes
for either target. This confirms that regularization/canonicalization fixes a
real minimum-L1 target instability.

It does not fix conservative safe-set coverage. Ridge-Huber and minimum-L1
both accepted no safe off-grid action at the same two sparse states. Ridge-
Huber's QP was primal infeasible at goal-II-t2-e00 step 30 despite one exact-
safe action, because its calibrated L5-part-2 plane remained 1.678 mm below
that action's exact margin and therefore negative.

Training a state-conditioned network on this target is rejected: it would
learn a stable representation already known to delete required safe support.
The next representation-level test must use multiple local affine regions or
condition the region on safe-support geometry. The separate minimum-L1 OSQP
iteration stall at spatial-I-t1-e00 remains a deployment diagnostic, not the
cause of the ridge-Huber NO-GO.

# ADR-0081: Test a fixed overlapping multi-region affine safe set

**Status:** Decided; preregistered before execution (2026-08-10).

Job `37688` produced stable conservative single-affine rows but deleted exact
safe support in two sparse states. The next experiment changes only the safe-
set representation: the existing action box is covered by 27 fixed overlapping
regions, and ridge-Huber rows are fitted independently from the already
executed 27 grid points in each region.

Every region is optimized separately with hard regional bounds and all seven
L5--L7 constraints. Fresh cloned-OSC verification is mandatory for every valid
regional proposal, including proposals later rejected by selection. The final
choice is the verified-safe proposal with the smallest nominal-action change.
This is a privileged oracle test of piecewise-affine representational capacity,
not a deployable online filter.

Zero margin and +1 mm clearance are independent frozen arms. Overall GO needs
both arms to have zero false-safes, stable active fits, complete exact-safe QP
coverage, and recovery of the two zero-margin safe-support misses. No MLP or
closed-loop E05 runs in this gate. A failure distinguishes inadequate fixed
regional affine coverage from the already rejected single-affine target; a
pass only authorizes designing a later learned region-conditioned model.

Clean H100 job `37690` resolves the fixed 27-region oracle as NO-GO. The
regional union is materially more expressive: at zero margin it has no false-
safes, 99.85% safe recall, and exact-safe QP selections in every state, versus
95.76% recall and 49/50 QPs for one affine region. It recovers one of the two
registered sparse-support states, and finds an exact-safe QP even in the other.

The strict representation is nevertheless rejected. State index 19's sole
immutable safe action remains outside every conservative regional certificate;
13 active L5-part-2 rows across five states fail norm stability; and one +1 mm
QP proposal misses its buffer by 0.000511 mm. Do not train an MLP or run
closed-loop E05 from this gate.

The retained mechanism insight is narrower: multiple regions repair most
single-plane conservatism and all zero-margin QP feasibility, but fixed equal
partitioning with only 27 points per region creates boundary coefficient
variance and still misses an isolated safe component. Any continuation must
address adaptive boundary coverage and confidence, then repeat the immutable
support/stability/fresh-rollout gates; it cannot reinterpret job `37690` as GO.

# ADR-0082: Judge regional resampling at the decision level

**Status:** Decided; preregistered before execution (2026-08-10).

Raw affine coefficients are not uniquely meaningful when overlapping regions
and conservative error calibration can produce the same accepted safe set and
QP action. Therefore job `37690` remains NO-GO, but the 13 coefficient-norm
failures are not automatically treated as controller instability.

The decisive audit refits coherent seven-row regions on shared 80% subsets and
recalibrates every one-sided error on the complete 27-point region. It compares
off-grid Boolean acceptance and the selected QP action against the full fit,
then freshly verifies each selected action. Coefficient cosine or norm is no
longer a decision gate.

Only stable accepted sets, retained support, zero false-safes, exact-safe QP
selections, and small selected-action displacement may classify coefficient
non-uniqueness as harmless. Passing may clear a separate learned-model design
gate, but this experiment trains no model and never retroactively changes the
strict result of job `37690`.

Clean H100 job `37701` passes every registered decision-level gate. Across
76,800 replicated off-grid decisions it has zero false-safes; minimum global
and state/replicate Jaccard are `0.999495` and `0.952381`; and all 800 selected
QP actions pass fresh exact cloned-OSC verification. Selected-action p95 and
maximum L2 shifts are `0.002991` and `0.031053`, far below their frozen limits.

The selected region index changes in 107/800 replicates, but overlapping-region
identity is not the control decision: those switches remain action-stable and
exact-safe. The observed raw coefficient non-uniqueness is therefore accepted
as decision-harmless for this sampled zero-margin oracle. A separately
preregistered grouped MLP experiment is now permitted. No learned model or
closed-loop E05 run has been authorized by this result, and job `37690` remains
strict NO-GO under its original zero/+1 mm all-gates criterion.

# ADR-0083: Learn regional lower planes and gate closed-loop E05

**Status:** Decided; preregistered before execution (2026-08-10).

Job `37701` shows that the fixed regional controller is stable even when its
raw ridge-Huber coefficients vary. The learned object is therefore the induced
regional lower affine plane, not equality to one arbitrary coefficient fit.
A shared region/constraint MLP ensemble predicts the lower anchor and three
action coefficients from state-relative geometry, nominal two-action context,
and fixed region coordinates.

Training uses only complete train groups. Its loss directly evaluates every
predicted plane over the 27 regional grid actions and penalizes exact-margin
overestimation. Ensemble disagreement supplies a state-dependent guard;
complete validation groups supply a separate fixed one-sided calibration.
E05/E10/E15 cannot affect the model, calibration, early stopping, or gates.

Neural reproduction is a control-decision claim: zero unseen false-safes,
accepted-set overlap, retained safe support, valid regional QPs, and fresh
exact selected-action rollouts are mandatory. Raw coefficient cosine is
diagnostic only. Closed-loop E05 is conditionally executed only after every
learned gate passes.

The E05 controller retains released AEGIS EE filtering, switches to live pi0.5
replanning after its first learned correction, and recomputes the regional QP
after every executed action within a frozen 50 mm current-clearance activation
boundary. Farther states execute the released AEGIS nominal so the network does
not extrapolate beyond its registered state support. The cloned simulator
records exact evidence but cannot influence selection or stopping. This
prevents a simulator oracle from silently becoming the deployed controller.
Collision-free stopping is not a pass: native SafeLIBERO completion is required
with zero L5--L7 contact and no paper CAR.

Clean H100 job `37722` is a valid strict NO-GO. The learned rows had zero
false-safes but also zero accepted actions across all 1,440 unseen off-grid
tests, versus 1,235 accepted by the exact regional oracle. No learned QP was
defined, and the conditional E05 arm correctly did not run.

This is not a QP or uncertainty-padding failure. Removing both calibration and
the ensemble guard in a read-only audit still yielded zero accepted actions.
Raw unseen affine-anchor prediction had 144.200 mm RMSE and -131.545 mm mean
error. Test inputs reached 10.182 training standard deviations, compared with
3.158 maximum inside training. The decision is therefore to reject this model
and current grouped training support, preserve the multi-region oracle result,
and keep closed-loop blocked. Any learned continuation must first resolve
state-support/representation extrapolation and pass the same unseen safe-set
gate.

# ADR-0084: Diagnose state support before replacing the regional MLP

**Status:** Decided; preregistered before execution (2026-08-10).

Job `37722` failed before QP execution and test features reached 10.182
training standard deviations. Changing architecture without distinguishing
unsupported extrapolation from a poor output parameterization would confound
the next result. The next gate therefore trains no model and runs no simulator.

It ranks the frozen 62-dimensional feature shifts, defines state support from
cross-episode nearest-neighbor distances using training data only, and compares
the induced multi-region union margins and accepted sets at matched normalized
actions. Decision-level oracle values are used instead of raw coefficient
equality because job `37701` already established benign coefficient
non-uniqueness.

If any test state lacks registered support, collect additional complete-episode
boundary states and preserve grouped splits. Only if every test state is
supported and its oracle behavior lies within the training reference envelope
may an action-conditioned conservative safety-value model be preregistered.
Closed-loop E05 remains blocked under every outcome of this diagnostic.

Clean H100 job `37733` resolves the diagnostic: supported test states are
`0/15`, so additional grouped boundary episodes are required before another
model. The cross-episode training p95 distance is `1.264877` RMS z, while test
nearest-neighbor distances are `1.410731--3.212344`. Every nearest neighbor is
from spatial-I-task-1 episode 18 rather than the held-out goal-II-task-0
population.

Eight features exceed five z. The largest are joint velocity 3 (`10.182`),
nominal first-action x (`9.342`), nominal second-action x (`9.010`), and joint
velocities 4/2 (`8.862/8.258`). The induced oracle is smooth under the frozen
decision thresholds for 11/15 pairs, but that cannot compensate for absent
state support. Do not authorize the action-conditioned model yet. Collect
additional complete-episode boundary states, retain E05/E10/E15 as test-only,
and repeat support before learning or closed-loop E05.

# ADR-0085: Expand same-task episode coverage before changing the model

**Status:** Decided; preregistered before execution (2026-08-10).

Job `37733` shows that the current learned failure occurs outside its training
support. Replacing the coefficient-output architecture before measuring a
supported same-task population would confound feature coverage with model
parameterization. The next experiment therefore changes data coverage only.

Goal-II-task-0 E00/E20 are complete training episodes and E30 is a complete
validation episode. E05/E10/E15 remain test-only. Although these episodes
share one task-level group, the claim is explicitly same-task unseen-episode
generalization and the indivisible split unit is the complete episode.

Collection reuses the frozen two-step cloned-OSC grid. Regional labels reuse
the validated 27-region ridge-Huber construction. No model is trained until
all 15 test states lie inside the training-derived state-support envelope and
the induced regional oracle is smooth relative to cross-episode training
pairs. A pass authorizes a separate rerun of the current region-aware MLP; it
does not authorize closed-loop E05. Only a supported-state learned failure
would motivate the action-conditioned conservative value model.

Clean H100 job `37742` resolves this first expansion as a strict support
NO-GO. E00/E20/E30 reduce the maximum test shift from `10.182 z` to `5.601 z`
and recover all five E05 states plus two late E15 states, but only 7/15 test
states pass the training-derived support gate. All five E10 states and the
first three E15 states remain unsupported.

The fixed regional oracle is smooth for 15/15 matched test/training pairs, so
there is no evidence here that the oracle mapping must be replaced. Conversely,
partial coverage is insufficient to retry the MLP: doing so would knowingly
retain unsupported extrapolation. Collect more complete same-task episodes
that cover E10 and early E15. Preserve E05/E10/E15 as test-only, and rerun the
same support gate before learning or closed-loop execution.

# ADR-0086: Exhaust same-task episode coverage without held-out selection

**Status:** Decided; preregistered before execution (2026-08-10).

Job `37742` leaves five E10 states and three early E15 states unsupported. The
remaining archived goal-II/task-0 episodes are E25/E35/E40/E45. All four are
included as training episodes, so the experiment does not choose a favorable
subset by inspecting E05/E10/E15. Held-out features and labels do not influence
new state registration.

Each new episode instead supplies its own closest nonnegative five-state
L5--L7 boundary window. Its complete replay scan records action, joint
configuration, and joint velocity coverage; the registered states receive the
unchanged cloned-OSC action grid and fixed regional oracle. This is the last
available archived same-task coverage expansion, not a new model experiment.

The current region-aware MLP remains blocked unless the expanded 60/10/15
population passes support and induced-oracle smoothness for all 15 held-out
states. A support pass authorizes only a separate preregistered retraining run.
If that supported-state model still fails its zero-false-safe, per-state safe
support, QP, and exact-rollout gates, the coefficient-output parameterization
is rejected in favor of an action-conditioned conservative value model.

# ADR-0087: Retry the current MLP only after full same-task support

**Status:** Decided from validated H100 evidence (2026-08-10).

Job `37771` resolves the data-coverage prerequisite: all 15 E05/E10/E15 states
pass both the training-derived state-support gate and the induced regional
oracle-smoothness gate. Maximum test shift is `3.555 z`, no feature exceeds the
five-z bound, and E25/E45 provide the closest neighbors for the previously
unsupported goal-II/task-0 states.

This result does not validate the learned controller. It removes the stated
reason for refusing to test it. The next experiment must keep the architecture,
regional targets, grouped episode split, loss, ensemble guard, validation-only
calibration, off-grid test actions, regional QP, and exact verification gates
unchanged apart from the expanded immutable dataset cardinality.

Closed-loop E05 remains blocked until that retrained model has zero unseen
false-safes, safe-action support in every test state, valid seven-row QPs, and
fresh exact-safe selected-action rollouts. If it fails with supported inputs,
stop extending the coefficient-output MLP and preregister the action-conditioned
conservative safety-value model.

# ADR-0088: Hold the model fixed for the supported-state attribution test

**Status:** Decided; preregistered before execution (2026-08-10).

The purpose of the next run is causal attribution: job `37722` failed with
unsupported inputs, while job `37771` now supplies supported inputs. Changing
the architecture or objective simultaneously would prevent deciding whether
coverage was the cause. Therefore the region-aware MLP, losses, ensemble,
uncertainty guard, calibration rule, regional QP, and learned gates remain
unchanged.

E30's five validation states require the same deterministic off-grid labels as
the original validation population. Collecting those 480 validation-only
rollouts is part of completing the unchanged calibration contract, not new
test supervision. E05/E10/E15 remain isolated.

Do not run closed-loop in the retraining job. If all unseen acceptance, QP,
fresh exact rollout, EE compatibility, and selected-action-shift gates pass,
preregister closed-loop E05 separately. If the supported-state model fails,
stop treating data coverage as the primary blocker and move to the
action-conditioned conservative safety-value parameterization.

# ADR-0089: Replace coefficient prediction with action-conditioned values

**Status:** Decided from validated H100 evidence (2026-08-10).

Job `37795` tested the unchanged coefficient-output MLP after job `37771`
established adequate training support for every held-out state. The network
still rejected all 1,440 test actions in all 15 states, whereas the exact
simulator and fixed regional oracle supplied 1,236 and 1,235 safe/accepted
actions. Thus the prior coverage hypothesis is rejected for this model.

Do not enlarge, retrain longer, or loosen calibration on the coefficient
network. The next learned gate predicts quantitative two-step constraint
margins from state, constraint identity, and candidate Cartesian action.
The validated ridge-Huber routine fits regional QP rows to guarded predicted
values; validation-only one-sided tightening and fresh exact rollout
verification remain mandatory.
No held-out test label may select architecture, calibration, or checkpoints.
Closed-loop E05 remains unauthorized until the replacement passes every
unseen acceptance, support, QP, and exact-verification gate.

# ADR-0090: Learn rollout values and derive regional rows deterministically

**Status:** Decided; preregistered before execution (2026-08-10).

The coefficient target is replaced, but the validated regional safety/QP
mechanism is retained. The MLP predicts exact two-step margins at candidate
actions. At runtime it evaluates the fixed 125-action grid, applies
validation-only lower-confidence tightening, and uses the validated
ridge-Huber fitter to derive the 27 regional half-spaces.

This separates what must generalize across states (quantitative executed
safety values) from what can be computed deterministically at the current
state (local QP coefficients). The direct model uses a current-clearance
residual and normalized local action coordinates. E05/E10/E15 remain test-only;
the simulator cannot select a proposal. No closed-loop run is authorized by
the preregistration itself.

# ADR-0091: Reject a global uncertainty pad as a completed safety model

**Status:** Decided from validated H100 evidence (2026-08-10).

Job `37807` shows that direct action-conditioned value learning is a better
representation than coefficient regression, but the plain ensemble plus one
validation-calibrated per-row pad is not a deployable lower bound. It is
overconfident at E05 step 185 and eliminates all support at three other unseen
boundary states.

Do not run QP or closed-loop on this result, and do not report its low RMSE as
safety success. Before proposing another network, perform a no-retraining
compatibility audit: determine whether any additional scalar tightening can
simultaneously yield zero test false-safes and preserve safe support in all 15
states. This is diagnosis using held-out outcomes, not a selectable deployment
threshold. Failure of that interval directs state-local/nonparametric residual
adaptation; success directs a learned state-conditioned uncertainty bound.

The interval is empty without another run. Any nonnegative tightening is
monotone: it cannot turn a rejected action into an accepted action. Job `37807`
already has only 12/15 supported states before additional tightening, so no
larger global pad can meet the 15/15 support gate. The next preregistration must
estimate state-local uncertainty from episode-grouped non-test residuals; it
must not tune a threshold on E05/E10/E15.

# ADR-0092: Audit target authority before local uncertainty learning

**Status:** Decided; preregistered before execution (2026-08-10).

The next uncertainty model is meaningful only if nonnegative seven-row
ellipsoid rollout margins do not hide protected raw MuJoCo contact. Therefore
all existing exact grid and off-grid rollouts are audited before generating
leave-one-episode-out residuals. The gate is zero dangerous false-safe target
labels over the complete 13,025-action population.

This audit is narrower than whole-arm safety: it covers the registered seven
L5--L7 ellipsoids against the exact moka-pot collision primitives and raw
protected distal contact. EE compatibility, CAR, and task success remain
separate later gates. A pass authorizes local residual calibration only.

# ADR-0093: Accept the seven margins as scoped residual targets

**Status:** Decided from validated H100 evidence (2026-08-10).

Job `37808` found zero dangerous geometry-target false-safes over 13,025 exact
rollouts. Proceed with leave-one-episode-out local residual calibration using
the seven quantitative margins. Keep the claim limited to the certified
L5--L7 ellipsoid union and exact moka-pot primitives; retain EE, CAR, and task
success as independent later gates. Closed-loop remains unauthorized.

# ADR-0094: Calibrate the useful value model with episode-OOF local errors

**Status:** Decided; preregistered before execution (2026-08-10).

Retain the action-conditioned MLP because its mean-margin RMSE materially
improved, but replace its inconsistent global uncertainty pad. Generate
dangerous prediction errors from complete held-out training episodes, never
from E05/E10/E15. Use a fixed, simple nearest-neighbor maximum residual per
constraint with a 1 mm pad before regional affine fitting.

This gate asks whether conservative calibration can simultaneously remove the
E05 false-safes and recover the E10/E15 rejected safe support. It is not a new
large network and cannot tune on held-out outcomes. Receding closed-loop E05 is
authorized only by zero false-safes, 15-state support, 15 valid QPs, and 15
fresh exact-safe proposals.

# ADR-0095: Reject the fixed local-KNN error bound

**Status:** Decided from validated H100 evidence (2026-08-10).

Job `37829` recovered safe support in every unseen state but retained 13
false-safe actions, so it cannot authorize QP execution or closed-loop E05.
Do not tune `k`, the 1 mm pad, or a new threshold from these held-out outcomes.

The result also exposes an evidence-unit limitation: action and constraint rows
within one simulator state are correlated, while only 12 complete training
episodes generated the OOF pool. A nominal 0.999 row-level quantile therefore
must not be presented as a 0.999 episode-level guarantee.

The next read-only attribution must compare pointwise conservative values with
the final regional-affine accepted decisions on the same immutable actions. If
pointwise values fail, replace the uncertainty estimator and obtain more
episode groups. If only interpolation fails, retain the value/bound model and
replace or certify the regional projection step. Closed-loop remains blocked.

# ADR-0096: Replace proxy-target assumptions with native physical-row discovery

**Status:** Decided; preregistered before H100 execution (2026-08-10).

The zero raw-contact false-safe audit does not establish that the seven
ellipsoid margins are accurate quantitative regression targets. Before another
uncertainty model, discover the actual compiled L5--L7 collision geoms and
query their signed distances to collision-eligible active-obstacle geoms.

Do not manufacture seven native rows. Freeze one row per actual protected geom
and use link/global minima only as aggregates. Keep `D_sim` signed distance and
raw contacts separate from `D_opt` ellipsoid support gaps. Include `k=0` only
for prevention states; label initially negative states as recovery cases.

This inventory gate precedes all native-target collection, model training, QP,
and closed-loop work. A pass authorizes only a separately preregistered
physical candidate-rollout collection experiment.

# ADR-0097: Reject large-cutoff `mj_geomDistance` as the native target

**Status:** Decided from validated H100 evidence (2026-08-10).

Job `37863` validates the compiled grouping—three L5--L7 protected geoms
against 15 active obstacle geoms—but rejects the 1 m distance query as a
quantitative target. Negative queried distances without raw contact and
41--47 mm differences from individual contact depths show that the frozen
contact-consistency contract does not hold.

This agrees with the MuJoCo API warning that large positive `distmax` can be
approximate for general convex collision pairs. Do not reinterpret the 29
negative query states as confirmed collisions, relax the gate post hoc, or
train on these values.

Preserve the semantic inventory and next test the smallest useful native
measurement: `distmax=0` for overlap plus a fixed sequence of small positive
cutoffs for near-boundary clearance. It must agree in sign with raw collision
state and avoid negative/no-contact contradictions before any candidate target
collection.

# ADR-0098: Test zero-cutoff overlap before rejecting native distances

**Status:** Decided; preregistered before execution (2026-08-10).

The large-cutoff bug does not by itself prove that the zero-cutoff collision
query is unusable. Freeze an adaptive measurement: zero cutoff for overlap,
then powers-of-two millimetre cutoffs up to 64 mm for safe clearance. Never
treat a right-censored result as an exact target.

Require pair-level equivalence between zero-cutoff negative sign and registered
raw contact throughout the cloned transition. Any cutoff-induced negative or
repeated-distance inconsistency rejects the API for the research target. This
gate cannot authorize training or control directly.

# ADR-0099: Keep MuJoCo native authority binary, not differentiable

**Status:** Decided from validated H100 evidence (2026-08-10).

Job `37875` establishes exact pair-level agreement between zero-cutoff negative
sign and raw protected contact, but rejects every positive-cutoff construction
tested. Retain zero-cutoff/raw contact as binary `D_sim` authority only.

Do not train a continuous execution-margin network on unstable positive
distances or assume a contact classifier gradient supplies avoidance. The
research method now needs an explicit choice: a certified analytic proxy field
with binary native verification, or a separately validated external physical
distance engine. This is a core claim decision and should be confirmed with the
advisor before another learned-model experiment.

# ADR-0100: Compare the analytic proxy only to binary native contact

**Status:** Preregistered before H100 execution (2026-08-10).

After the positive native-distance NO-GO, keep optimizer clearance `D_opt`
distinct from simulator verification `D_sim`. Audit the seven ellipsoid
minimum rollout margins against registered protected L5--L7 contact on the
complete existing paired population. Do not reinterpret an ellipsoid-margin
band as physical millimetre clearance.

Report false-unsafes as well as false-safes across frozen thresholds and
proxy-boundary strata. This diagnostic may authorize only the next
initial-state classification gate. It cannot authorize training, calibration,
QP execution, or closed-loop E05.

# ADR-0101: Retain the ellipsoid margin as a conservative smooth proxy

**Status:** Decided from validated H100 evidence (2026-08-10).

Job `37879` found zero observed ellipsoid false-safes across 466 protected
contact actions but 1,911 false-unsafes across 12,559 contact-free actions.
Retain the analytic margin as `D_opt` for the simplest research continuation,
with raw MuJoCo contact as independent `D_sim` verification. Do not claim the
ellipsoid value is exact physical clearance.

The proxy may reduce task-compatible support, but it does not explain the
learned model's false-safe predictions against its own target. Do not tune the
-2 mm observed threshold from this population. Proceed only to a separately
registered initial-state classification before revisiting learning.

# ADR-0102: Classify k0 from contact, not rejected positive distance

**Status:** Preregistered before H100 execution (2026-08-10).

Reevaluate every paired state's initial cohort using raw protected L5--L7
contact and only zero-cutoff MuJoCo sign. Do not reuse the rejected 1 m native
distance result that labeled 29 states as recovery without raw contact.

Preserve all states in the report. Prevention states may later supervise the
complete rollout minimum including `k=0`. Any genuine recovery state requires
an explicitly separate future-only objective because its fixed unsafe `k=0`
cannot be repaired by the action.

# ADR-0103: Reject unsafe k0 as the current learning failure

**Status:** Decided from validated H100 evidence (2026-08-10).

Job `37882` found all 85 states contact-free at `k=0`, with exact raw-contact
and zero-cutoff sign agreement. Treat every current state as prevention for the
scoped L5--L7 experiment. Do not create a recovery split from rejected positive
native distances.

The prior action-conditioned model's failure cannot be attributed to an
irreparable initial collision. Because that model already used the same
episode-grouped states, complete q/qdot and relative proxy geometry, candidate
action, and quantitative ellipsoid rollout targets, do not rerun it unchanged.
Require a separately preregistered prediction-model change before calibration
or QP.

# ADR-0104: Test complete OSC input sufficiency before changing the model class

**Status:** Preregistered before H100 execution (2026-08-10).

The previous 56D action-conditioned model omitted full rotation/gripper
commands, controller orientation goals and internal state, auxiliary simulator
control state, and complete proxy transforms. Recollect the same 85-state,
10,625-action population with those inputs and fresh two-action labels.

First require every identical snapshot/action replay to reproduce bitwise
ellipsoid margins, canonical raw contacts, and both next-state hashes. Only
then train the same action-conditioned residual concept with the complete
input. Keep grouped episodes and E05/E10/E15 test isolation. Do not add
calibration, a QP, pi0.5 features, or closed-loop execution in this gate.

If replay is deterministic but the registered prediction gates still fail,
missing OSC input is not a sufficient explanation and the remaining problem
is the model/data representation. A pass authorizes only a separate
calibration and contact-verified QP preregistration.

# ADR-0105: Reject missing OSC state as the sufficient MLP explanation

**Status:** Decided from validated H100 evidence (2026-08-10).

Job `37910` proves that the saved complete simulator/controller snapshot and
full two-action command deterministically define the scoped two-step ellipsoid
target: 10,625/10,625 duplicated pairs reproduced exactly. Hidden or omitted
OSC state is therefore not the immediate source of label ambiguity.

Reject the raw complete-state/plain-MLP representation. It produced 271 test
false-safes and worse margin accuracy than the earlier compact model, despite
accepting safe actions in all states. The apparent recall gain is invalid
because every test action was accepted. Do not add calibration, a QP, or
closed-loop execution to this predictor.

Any continuation must be a separately preregistered representation change,
such as a compact structured controller/geometry state or learned task-aware
latent encoder, evaluated on the same immutable deterministic dataset and
complete-episode splits. It must pass prediction gates before control.

# ADR-0106: Isolate input content with a matched fresh-label ablation

**Status:** Decided from validated H100 evidence (2026-08-10).

Do not compare the prior 56D result to the complete-input result across
different label collections, schedules, or seeds. Reconstruct the old 56D
projection from the fresh semantic states and train both arms against the same
immutable seven-row rollout-margin targets with identical grouped splits,
architecture, loss, seeds, normalization, and schedule.

Interpret only frozen gates: complete-only test passage supports omitted input
as the root cause; two training fits followed by two test failures identify
state/setup coverage; two near-boundary training-fit failures identify the
target or plain-MLP representation. Mixed outcomes remain inconclusive. This
diagnostic cannot authorize Poisson fields, calibration, a QP, more data, or
closed-loop control.

Job `37925` resolves the frozen decision as a training-fit failure for both
arms. The compact arm is materially better but still misses the boundary-fit
gate; the complete arm has worse train/test error and more false-safes. Reject
omitted inputs as the root cause for this plain residual MLP. Because training
itself fails, do not classify this as the registered state/setup coverage
failure. The deterministic target remains defined, so prioritize a structured
or local representation of the execution-margin function before revisiting
calibration, a QP, additional data, or closed-loop E05.

# ADR-0107: Prove old-input insufficiency only with controlled conflicts

**Status:** Decided from validated H100 evidence (2026-08-11).

Do not infer that an omitted variable is causal merely because a larger input
model performs differently. Hold the historical 56D vector byte-identical,
change one omitted variable family at a time from the exact same simulator
state, and execute the same two-action translation through cloned OSC. Declare
the old input insufficient only if one identical hash maps to both safe and
unsafe seven-row rollout margins.

Reuse the already validated exact duplicate-replay population for target
determinism. Recheck the immutable complete-input model rather than retraining
or tuning it. No audit result authorizes calibration, a QP, or closed-loop E05
unless the separate prediction gate reaches zero false-safes and 15/15 safe
support.

Job `37933` proves insufficiency: 83 byte-identical old56 groups cross the
proxy safety class and 41 also cross raw-contact class. All four omitted groups
affect the target, but rotation action has the largest effect and the most
crossings. The existing complete-input model still fails with 271 false-safe
test actions. Follow the user's conditional rule with one targeted
old56-versus-old56-plus-rotation ablation. Do not expand to a six-arm study,
Poisson field, calibration, QP, or closed loop at this decision point.

# ADR-0108: Factor controller execution before changing safety geometry

**Status:** Preregistered before H100 execution (2026-08-11).

The user supersedes the targeted rotation-only MLP ablation with a structured
execution pilot. Preserve job `37933` as proof that full rotation/action state
must be present, but do not claim omitted inputs are the only failure: both
matched direct-margin arms also failed to fit their training boundary.

Collect a fresh matched population because the previous dataset stores only
rollout minima and varies only first-action XYZ. At all 85 grouped states,
record every OSC substep for both complete 7D actions. Use one nominal, 28
all-coordinate finite-difference probes, and 64 deterministic antithetic local
actions. Supervise both the 51-by-7 joint trace and its actual clipped
14-coordinate action secants.

Compare a direct minimum-margin arm and factorized joint-execution arm on the
same complete input, action rows, splits, trunk, seeds, optimizer, and
schedule. The factorized arm must compute safety only through known MuJoCo
forward kinematics and the existing ellipsoids. Require joint/link accuracy,
action-space sensitivity agreement, fewer false-safes than the fresh direct
arm and prior 72-action result, recall, 15/15 support, and boundary accuracy.

Do not add Poisson/SDF, calibration, a QP, policy inference, or closed-loop
E05. Passing authorizes only a separate geometry-pipeline pilot; it is not a
whole-body or invariance claim.

Initial job `37948` exposed the already known pinned-PyTorch lack of H100
`sm_90` kernels before producing data. Keep simulation in the H100 allocation
and run both matched learning arms on the same eight allocation CPUs. This is
an apparatus repair, not an experimental-arm change; discard the partial run.

Retry `37960` exposed only Python-3.8 dictionary-union incompatibility before
writing a dataset. Replace it with equivalent dictionary unpacking and retain
every frozen scientific setting. Do not reuse the failure-only run.

Retry `37966` also rejects a scalar `[-1,1]` bound for every archived 7D
coordinate: the immutable E05 gripper input can slightly exceed one. Preserve
the nominal command exactly, clip only pose coordinates, and apply local
gripper perturbations without harness clipping so the unchanged controller
retains formatting authority. This is an action-domain correction made before
any dataset or result, not post-outcome tuning.

Job `37980` fixes the scientific decision at NO-GO before any downstream
experiment: useful learned sensitivities do not offset 44 false-safe actions,
13/15 support, or 11.560 mrad joint-trajectory RMSE. Do not calibrate, solve a
QP, run closed loop, or add Poisson/SDF. The validator's only failed predicate
is an implementation error: evaluation deliberately stores NaN in non-test
geometry rows, and `numpy.array_equal` rejects identical paired NaNs by
default. Retain the failed receipt, change equality to accept only matching
NaN masks while requiring exact finite values, and validate the immutable
job-37980 result separately. This cannot change any model, metric, gate, or
scientific conclusion.

Job `38025` validates the repair with `0.0 m` maximum finite difference and an
identical NO-GO decision. Treat factorization as useful mechanism evidence,
not an intervention-ready method: it strongly improves direct-regression
accuracy and learns action sensitivity, but its 44 proxy false-safes block the
safety claim. The exact-joint geometry arm localizes the next work to execution
prediction. Do not spend the next gate on Poisson/SDF, calibration, QP, or
closed loop before the joint/substep error responsible for false-safes is
reduced on held-out episodes.

Prefer surface-position supervision over another uniform joint loss only if
the 44 false-safes exhibit the proposed spatial failure. Freeze a diagnostic
before inspection: compare the exact-normal ellipsoid support-point error at
each unsafe action's worst constraint against same-state nearest-margin
correctly rejected controls. Require 0.5 mm median excess, 65% paired wins,
and 0.5 signed-error correlation. Passing authorizes only a later matched
surface-loss pilot; failing redirects discussion to one-sided uncertainty.
The diagnostic itself cannot train, calibrate, optimize, or execute actions.

Job `38036` rejects generic symmetric surface-position loss under the frozen
gate. False-safe Euclidean surface and center errors are not larger than
matched unsafe controls at the population median, despite 72.7% paired wins.
The useful signal is directional: all failures are L5 at terminal substep 50,
with about 0.35--0.37 mm additional signed center/surface clearance
overestimation and 0.635 correlation with margin overestimation. Treat this as
terminal one-sided execution bias, not missing Poisson geometry or active
witness switching. Before any downstream controller experiment, decide
between terminal L5 normal-direction supervision and a state-conditioned
one-sided uncertainty model; do not relabel this result as support for generic
surface loss.

Choose exactly one targeted refinement before uncertainty calibration: a
matched one-sided per-step geometry-loss ablation. First require the immutable
job-37980 model to reproduce terminal L5 optimism on validation episodes. Fit
a local `dh/dq` only from each train/validation state's nominal and 28
registered finite-difference candidates; test episodes never supply this
training signal. Penalize signed clearance overestimation more strongly near
the boundary and late in the 51-state horizon while keeping the network output
as the joint trajectory. At evaluation, retain known FK plus the existing
ellipsoid geometry as safety authority. Require zero unseen false-safes and
15/15 safe support before any uncertainty model. Calibration, QP, and closed
loop remain blocked.

Job `38064` preserves the registered L5-specific conditional NO-GO and skips
training: validation has 93 false-safes, all at substep 50, but every exact
worst row is the first L6 row rather than L5. The local signed geometry model
is accurate (0.201 mm RMSE, 0.974 cosine), so the failure is not its numerical
quality. Do not relabel v1 as passing. Because the already frozen auxiliary
loss sums across all seven distal rows, register a separate validation-adapted
v2 whose only change is accepting terminal optimism on any L5--L7 row. Keep
the loss, model, test set, and zero-false-safe/15-support gates unchanged.

Job `38070` validates the one-sided geometry loss as useful but insufficient.
It removes every test false-safe (44 to 0), retains 93.84% safe recall, and
preserves joint/margin sensitivity cosine at 0.977/0.921. It nevertheless
supports only 12/15 test states, worse than the flat baseline's 13/15, and
validation support is only 3/10. Preserve the strict NO-GO. Do not apply a
global or local subtractive buffer next: it cannot restore rejected safe
support. The next matched model must replace the flat 51-by-7 output layer
with shared time-conditioned execution decoding `q_hat_k=F(x,A,k)` while
retaining the successful one-sided geometry supervision. Classifier,
calibration, QP, and closed loop remain blocked.

The user now supersedes immediate decoder training with a causal audit of the
immutable job-37980 ensemble. Before another model is fit, recompute
`predicted - exact` static ellipsoid clearance at every L5--L7 row and all 51
substeps for the 64 out-of-fit actions at every state. Compare train,
validation, and diagnostic E05/E10/E15 episodes. Material terminal optimism in
train/validation implicates the loss or flat decoder; optimism only in the
diagnostic episodes implicates state coverage/generalization. Evaluate exact
five-member geometry disagreement on false-safes and matched boundary
controls. Replace the permissive 2 mm fitted-Jacobian gate with central-FD
known-FK geometry and require at most 0.5 mm near-boundary error, signed
agreement, and candidate-resampling stability. No training is authorized until
this audit resolves both questions. E05/E10/E15 may no longer support a final
generalization claim; reserve new complete episodes for that role.

Job `38071` is an apparatus failure before geometry measurement. Retain the
failed artifact and repair only ensemble arithmetic order: use the original
job-37980 predictor (mean float32 residual, then add float64 q0) for the mean
trajectory, while retaining baseline-plus-residual member trajectories solely
for exact disagreement. Do not relax the immutable prediction equality gate or
change any audit threshold.

Job `38072` resolves the causal audit in favor of a current training/decoder
failure rather than test-only generalization. L5 and L6 terminal optimism is
material in train, validation, and diagnostic test; every false-safe is at
substep 50. Do not rely on ensemble uncertainty: exact member disagreement is
not elevated on false-safes. Do not run the proposed one-sided local-Jacobian
loss: although exact `dh/dq` is accurate for actual small action-induced
motion, it is unstable under the frozen candidate-resampling gate and is not a
valid linearization over the flat model's larger prediction errors. The next
matched mechanism test must replace the 357-output flat decoder with
`q_hat_k=F(x,A,k)` while initially retaining only existing joint and
sensitivity supervision. Reaudit residuals before adding geometry loss.
E05/E10/E15 remain diagnostic only; any later final test uses newly reserved
complete episodes.

The independently replayed one-sided result remains a strict NO-GO while its
three unsupported states are audited. Do not train the time-conditioned
decoder or fit a correction first. Freeze job `38070`, inspect the 64 already
collected random actions in each unsupported state, and classify rejection by
training-state support, exact ensemble-member support, sampled-region support,
conditional signed bias, and joint-trajectory error. This diagnostic cannot
authorize calibration or intervention and cannot change the existing verdict.
Config hash is
`0b81c045da7fb8121abe2bf7b90a1ebed71af6c6d6d3fe33bb7658f28ca39d1b`.

Job `38073` is an apparatus failure before geometry. Job `38074` establishes
that support loss is heterogeneous: states 39/44 have no exact-safe sampled
action, whereas state 49 has 44 exact-safe actions rejected by the model.
Audit all three, but do not call the first two model false rejections. Region
insufficiency takes priority when the exact-safe count is zero. No outcome or
threshold is otherwise changed.

Job `38075` resolves the three-state audit into two distinct mechanisms. Do
not fit one correction across all three. E05 state 39 and E10 state 44 have no
exact-safe action in the registered region, so first test larger or continuous
exact-oracle recovery regions. E15 state 49 has 44 exact-safe actions rejected
by all five members despite supported state features and ordinary trajectory
error; this is a conditional conservative-bias candidate, not ensemble
aggregation. Any residual correction must be fit on validation episodes only
and must retain zero false-safes. E05/E10/E15 remain diagnostic, and the final
claim requires new episodes.

All three rejected states pass the frozen complete-state support audit, so do
not collect more episodes as the immediate mechanism test. Register one
matched shared time-conditioned decoder against immutable job `38070`. Keep
the one-sided geometry loss and every dataset/split/seed/target fixed; enforce
the known q0 endpoint and evaluate all 51 substeps. Count support only in
states that contain an exact-safe registered action, because no predictor can
create support in E05/E10's current candidate region. A diagnostic decoder pass
may authorize new reserved-episode evaluation only, never calibration, QP, or
closed loop. Config hash is
`452fe00d53f3b6e9c278962116c5cf760821f14b18b06a6d4079a577bd5a2d17`.

Job `38076` rejects the naive shared time-conditioned decoder. It keeps zero
test false-safes and useful gradients but loses safe support and roughly
sixfold worsens boundary RMSE; test terminal joint error also increases. Do
not train it longer or treat temporal weight sharing alone as the missing
component. The absurd source/new validation terminal errors (1341/1153 rad)
also show that the model-selection representation is unsupported or
explosively normalized. Before another architecture, audit normalized complete
inputs per episode and identify constant/padding/presence or causal controller
features responsible for the shift. Collect new grouped episodes only if the
shift is physical rather than representational. Calibration, QP, Poisson/SDF,
new-episode claims, and closed loop remain blocked.

# ADR-0113: Trace the shared input representation before another model

**Status:** Preregistered before H100 execution (2026-08-11).

The impossible validation outputs occur in both the flat and time-conditioned
models, so do not attribute them to positional time or the shared decoder
without first testing their common 2,124D normalized input. Freeze all existing
data, weights, predictions, and split assignments. Recompute normalization
from training rows, map every aligned scalar to its semantic name, and find the
first layer at which validation activations become explosive.

Use training-mean group masks only as a read-only causal diagnostic. Call a
representation artifact or physical support shift causal only when the same
semantic group removes at least 90% of terminal validation error from both
models. The audit cannot authorize retraining, calibration, candidate search,
QP, or closed loop; it selects the next matched experiment.

Job `38092` validates the normalization pathology but does not pass the frozen
single-group causal gate. Both networks explode at their first linear layer on
only goal-II-t3-e35. Raw obstacle rotation entries that are constant at `+/-1`
in training flip sign in that setup; the `1e-6` variance floor turns the valid
rotation into a `2e6` z-score. The same physical orientation is redundantly
encoded in the root transform and 15 primitive transforms. Because masking one
encoding leaves the other, the strongest single mask removes only 59%/73% of
flat/time error. Preserve the registered inconclusive verdict. The only
justified follow-up is a fixed joint mask of those already identified redundant
groups; do not retrain or tune a new representation yet.

# ADR-0114: Confirm duplicated obstacle orientation with one fixed joint mask

**Status:** Preregistered after job `38092`, before confirmation execution
(2026-08-11).

The single-group gate was intentionally too strict for two redundant encodings
of the same physical rotation. Freeze the two groups named by the completed
audit and mask their union to the training mean. Require at least 90% removal
of the exploding episode's terminal error in both immutable models while
preserving ordinary validation and test scale. Also report each single mask and
all variance-floored features, but do not use those auxiliary arms to change
the gate. A pass identifies the representation cause and authorizes only a
matched representation ablation, never direct retraining or intervention.

Job `38093` confirms the cause. The fixed union removes more than 99.999% of
the E35 terminal error from both frozen models and leaves the ordinary
validation episode and diagnostic test essentially unchanged. Classify the
impossible validation trajectories as a raw rotation-matrix scalar
standardization failure amplified by redundant primitive transforms, not a
failure of positional time. Do not treat mean-masking as a deployable fix: it
removes valid orientation information. The next authorized gate is a matched
structured orientation representation ablation using the same data, models,
losses, seeds, and splits. The time decoder remains NO-GO on normal-scale test
states until such a matched comparison says otherwise; QP and closed loop stay
blocked.

# ADR-0115: Replace duplicated rotation scalars with one bounded 6D orientation

**Status:** Preregistered before H100 execution (2026-08-11).

Mean-masking the two rotation groups proved causality but is not deployable
because it deletes valid geometry. Remove the 288 redundant world-rotation
scalars from the learned execution input and append exactly one 6D root
orientation. Keep primitive-local transforms in the known geometry evaluator.
Do not independently standardize presence bits or the bounded rotation.

Retrain both the flat job-38070 model and time-conditioned job-38076 model with
every non-representation variable fixed. This is the decisive test separating
the confirmed input pathology from the remaining normal-scale execution-model
error. Even a pass only authorizes newly reserved episode evaluation; it does
not authorize calibration, a QP, or closed-loop E05.

Job `38095` validates the representation repair but rejects both learned
models. Removing the duplicated scalars and using bounded 6D orientation makes
validation trajectories physical, so positional time and the old numerical
explosion are no longer confounded. Nevertheless, both arms introduce more
than 60 test false-safes and support only 8/13 eligible states. Record the old
flat zero-false-safe result cautiously: its model-selection objective included
the exploding validation episode and may have induced accidental conservatism.

Do not revert to the invalid scalar representation, and do not interpret the
stable-input failure as a QP failure. The learned clearance field is still not
conservative. Freeze QP and closed loop. The next justified action is a
read-only localization of structured-model false-safes; choose data expansion
or a new safety-aware objective only after distinguishing test-only coverage
shift from train/validation terminal optimism.

# ADR-0116: Audit the exact joint-to-safety composition before changing the MLP

**Status:** Preregistered before H100 artifact replay (2026-08-11).

The factorized model artifact must be treated as a joint-trajectory predictor,
not as a direct clearance predictor: its registered output is 51 configurations
of seven joints. Validate the downstream calculation with the same exact and
predicted traces. Write each q into MuJoCo, call nonlinear forward kinematics,
reconstruct the link-attached ellipsoids, and evaluate the exact analytic
support gap. Do not use the QP or a linearized h in this audit.

Decompose `h(predicted q) - h(dynamic rollout)` into `h(exact q) - h(dynamic
rollout)` and `h(predicted q) - h(exact q)`. If exact-q recomposition is
submillimetre, conservative, and substantially more accurate than predicted-q
safety, attribute the dominant current failure to execution prediction rather
than FK or phi. If it fails, stop MLP redesign and audit link transforms and
obstacle-time alignment first.

Job `38155` passes the full decomposition. Exact q through the unchanged
MuJoCo-FK/ellipsoid/phi chain is exact at k0 and has only 0.321 mm boundary
RMSE, zero false-safes, and 99.51% recall despite the deliberately frozen
obstacle used after k0. The learned q trace through the same evaluator has
2.302 mm boundary RMSE and 44 false-safes. Attribute the dominant failure to
the learned execution trajectory. Do not change the explicit geometry backend
or blame QP linearization; no QP was used. Improve the late-horizon joint
predictor, then rerun the same composition gate before intervention.
