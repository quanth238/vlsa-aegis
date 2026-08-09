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
