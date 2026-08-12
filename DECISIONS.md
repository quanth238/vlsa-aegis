# Reproduction decisions

## ADR-0083: Test live continuation locally before a complete receding episode

Accepted before release. Gate 0 established that the safe compound prefix
fails under the immutable raw suffix, so the next experiment must change the
continuation rather than enlarge the same prefix correction. Use exactly one
fresh frozen-`pi0.5` query at the verified post-compound action-187 state and
compare its released-AEGIS five-action chunk with the registered safe
continuation. Run the smooth counterfactual field only when the fresh chunk is
unsafe, using the identical frozen chunk and initial state.

Do not yet run a full episode, train an MLP, or introduce a QP. The decisive
local question is whether feedback supplies a safe continuation or whether the
existing smooth field can repair it. Preserve the registered continuation as
a binding positive control, not as evidence about the live policy. Verify all
internal MuJoCo steps with the same 1 mm L5--L7 clearance, zero-contact, and
paper-CAR authority. A local pass authorizes the longer receding experiment;
a local failure blocks it and identifies missing safe continuation support
without spending a full-episode rollout budget.

## ADR-0081: Prove raw fixed-suffix safe support before multi-start search

Accepted before release. The validated positive compound correction modified
both the first five actions and the continuation after action 186. It is not a
positive control for a search that changes only raw actions 182--186 while
freezing raw actions 187--201. Run the two controls separately under identical
pre-action-182 state binding and internal-substep authority.

Only a safe transplanted compound prefix authorizes the planned radii
`1.0/1.5/2.0`, derivative-free, structured temporal, and spatial-branch
comparisons. If the full compound passes but the transplant fails, stop rather
than interpreting search failure as bad optimization: the fixed raw suffix is
the missing factor. If the full compound itself fails, first audit replay
binding and transient internal collision. Treat every radius as an upper bound
and retain matched-budget and matched-norm comparisons as separate later
analyses.

Attempt `38977` revealed that bitwise action binding must use direct registered
value assignment rather than an algebraically equivalent displacement round
trip. Accept this as a scientifically neutral apparatus repair: displacement
is still reported as compound minus raw, while the applied prefix is copied
from the immutable compound values before explicit action-bound checking.

## ADR-0082: Reject raw fixed-suffix multi-start; require continuation adaptation

Accepted after clean H100 producer `38978` and independent validator `38979`.
The full compound positive control passed strict internal-substep authority,
but its identical first-five prefix followed by the raw suffix failed despite
correction L2 `1.849546`, zero clipping, and bitwise action binding. It delayed
first L5 contact by four actions and improved worst clearance by `9.285844 mm`,
but still reached `-6.789842 mm`, 85 protected-contact samples, and failed CAR.

Accordingly, do not run the preregistered radii or structured multi-start arms
on this fixed suffix. Such a search would no longer distinguish optimizer
failure from an invalid positive-control assumption. The evidence identifies
continuation adaptation after action 186 as necessary for this known safe
trajectory. The next no-learning oracle may optimize a longer horizon or use
short-prefix execution plus live frozen-VLA replanning, but it must preserve
internal-substep clearance/contact/CAR authority. This does not yet prove that
every possible five-action prefix with the raw suffix is unsafe; it proves that
the known safe compound is not evidence of one, so the planned comparison is
not decisive and remains blocked.

## ADR-0080: Reject direct smooth-field attribution; retain it only as a compound repair

Accepted after clean H100 producer `38974` and independent validator `38976`.
The stricter internal-substep audit reproduced raw AEGIS as unsafe at
`-16.075686 mm`. Applying the same `2 mm` smooth all-witness secant procedure
directly to raw actions improved clearance to `-7.009143 mm`, but retained 125
protected contact samples, missed paper CAR, and supplied no verified-safe
scale on the registered correction ray. The final held-out directional cosine
also fell to `0.694629`, identifying loss of local field fidelity where exact
line search stalled.

Do not attribute the prior `+1.673938 mm` result to standalone smooth-field
avoidance. That result was reproduced only by composing the earlier detour
with the smooth second-stage correction; its total correction L2 from raw
AEGIS is `1.849546`. Smooth aggregation remains useful for locally improving a
given detour, but the current local field cannot discover that detour mode from
the raw suffix. Consequently, do not run live closed-loop recovery and do not
train an MLP to reproduce this failed direct field.

The next proposal must target the missing nonlocal or multimodal correction
structure, and must again be tested as a no-learning oracle before scale-up.
Potential proposal generators may preserve multiple candidate detour branches
or use derivative-free/global search, while hard internal-substep ellipsoid
clearance, raw contact, and CAR remain the acceptance authority. This result
does not reject counterfactual OSC supervision or smooth risk evaluation; it
rejects the claim that one locally re-estimated smooth descent path is enough.

## ADR-0079: Attribute the smooth field before closed-loop recovery

Accepted before release. The validated smooth candidate was a compound
correction built on the earlier five-action detour continuation. Do not credit
the smooth field as a standalone avoidance method until the identical
counterfactual procedure is applied directly to immutable raw AEGIS actions
182--201 from the same pre-action-182 simulator state.

Keep the detour and compound candidates as read-only comparators. Fit the direct
field with action-boundary rollouts exactly as before, add independent paired
direction validation, and make final safety authority stricter by measuring all
internal MuJoCo integration steps. Accept only a candidate with at least `1 mm`
ellipsoid clearance, no raw protected contact, and paper CAR. Search scales on
the discovered correction ray and call the selected result only the smallest
registered safe ray scale, never a globally minimum-norm correction.

If direct attribution fails, the smooth result remains a useful second-stage
repair but cannot motivate standalone learned steering. If it passes, the next
independent gate may execute the verified five-action prefix, requery the
frozen VLA with matched policy noise, and repeat receding verification. No
learning or PNCBF claim is authorized by this attribution gate alone.

## ADR-0078: Reject hard top-M selection; retain smooth all-witness evidence

Accepted after clean H100 producer `38966` and independent validator `38967`.
Preserving eight near-active rows in a local epigraph was not sufficient: its
best exact margin remained `-6.678093 mm`, it retained three L5 contacts, and it
missed paper CAR. This is a strict failure of the preregistered multi-witness
gate while the fixed derivative-free safe-support control remains positive.
Do not train a link/time-conditioned hard-row MLP from this result.

The matched `2 mm` smooth maximum of risk is retained as positive, narrower
mechanism evidence. It combined all 140 future link/time rows continuously and
reached `+1.673938 mm` with zero protected contact and paper CAR after seven
updates. Hard top-M selection can change discontinuously and can exclude rows
before they become critical; the smooth arm supplies continuous pressure from
all witnesses. This causal explanation is an inference from the matched result,
not yet a population conclusion.

Do not collapse this into a claim that the primary problem is solved. The arm
used exact counterfactual cloned-OSC labels at one archived state, did not test
task recovery, and had `19.576451 mm` terminal EEF deviation. The next gate must
keep the smooth field frozen and separately test either multi-state avoidance or
live frozen-VLA task rejoining after its verified first five actions. Learning
remains downstream of that oracle evidence.

## ADR-0077: Preserve link/time witnesses in the next oracle

Accepted before release. The fixed-step field failure does not justify another
scalar loss or a larger MLP. Its direct evidence is active-witness switching:
the hard minimum has no unique gradient at the observed boundary. Preserve the
identity and fitted action sensitivity of each future link/time clearance row.

Compare a current-worst row, a smooth minimum, and an explicit multi-witness
epigraph under matched paired-rollout, trust-region, exact line-search, path,
and correction budgets. In clearance notation, the epigraph constraints are
equivalently `-h_j-g_j^T delta <= t`; this avoids reversing the feedback's
risk-space inequality. Select at most eight rows within 5 mm of the worst
clearance, retaining deterministic `(action offset, ellipsoid row)` identity.

Use exact cloned OSC only for counterfactual row labels and fresh acceptance.
Do not impose task or endpoint terms in this avoidance gate. If the coordinated
rows recover the known safe detour and beat the single row, the result supports
learning `G_theta(x,A,j)` rather than one 15D vector. If they fail while the
fixed derivative-free control is safe, the local convex representation is
still insufficient and the next method must represent alternative nonlocal
detours rather than add learning.

H100 attempt `38964` is retained as apparatus failure. A visited action was
near a normalized coordinate boundary, so the old rejection sampler could not
construct all symmetric `±0.05` pairs. Repair only the local sampling domain:
freeze coordinates whose bidirectional headroom is below `0.05` and sample the
same paired directions in the remaining axis subspace. This is the maximal
conservative symmetric neighborhood at the registered perturbation scale; it
does not waive action limits or change the scientific comparison.

## ADR-0075: Test the long-horizon field without task preservation

Accepted before release. Job `38955` already proved that collision-free
five-action corrections exist once endpoint preservation is removed, but its
learned field used exact backtracking and stopped after local witness
switching. This did not answer the narrower avoidance question requested here:
whether repeatedly recomputing the pure long-horizon field and taking small
fixed steps can reach an already-known safe detour.

Freeze an avoidance-only arm at E05 action 182. Re-estimate the field from 32
paired cloned-OSC outcomes at every center, take exactly one normalized `0.1`
step, and evaluate the complete twenty-action continuation containing the
action-197 contact. Do not impose endpoint equality or use task attraction,
task penalties, QP, MLP, or flow injection. Compare against repeatedly using
the initial field direction and retain the existing radius-1 analytical and
derivative-free oracles as no-learning safe-support controls.

Terminal EEF displacement is reported but is not part of this avoidance gate.
Passing requires positive exact L5--L7 clearance, zero protected contact,
paper CAR, radius at most one, and superiority to the fixed initial direction.
If the field fails while the controls remain safe, reject this single-gradient
correction representation before training. If it passes, the next independent
question is whether live frozen-VLA replanning can recover the task after the
temporary detour.

## ADR-0076: Reject the single hard-min action field before learning

Accepted after clean H100 producer `38961` and independent validator `38962`.
The registered field was not useless: ten recomputations improved exact hard
clearance by `7.138110 mm` from nominal and beat repeated use of the initial
direction by `5.736277 mm`. It nevertheless remained at `-7.125095 mm`, kept
two raw L5 contacts around actions 196--197, and missed paper CAR.

This is decisive because radius-1 analytical and derivative-free controls are
already exactly safe under the same avoidance authority. The failure is not
absence of an action-space detour. Every one of the 32 paired probes switched
the active future witness at every iteration, so a single gradient of the hard
minimum averages incompatible local modes and zigzags. Ten full normalized
steps resulted in only `0.749880` net correction norm.

Do not train an MLP to reproduce this vector, and do not proceed to live VLA
rejoining from an unsafe action. If the research direction continues, first
test the smallest multi-mode oracle that preserves separate active link/time
directions and selects a verified branch. Only a verified-safe field output
can authorize learning or closed-loop recovery.

## ADR-0073: Execute one action from every verified five-action window

Accepted as the decisive follow-up to `38881`. Recompute a fresh five-action
VLA window from the measured state after every executed action; do not execute
all five actions from a previously corrected window. A nominal window may pass
only with at least 1 mm exact L5--L7 clearance, no robot contact, and paper CAR.
Otherwise, run the validated task-rejoining five-action SQP, verify the complete
candidate again, execute only its first action, and shift the horizon. Preserve
the frozen VLA and OSC and train no MLP until this exact oracle achieves safe
task completion.

## ADR-0074: Do not learn from the infeasible receding five-action oracle

Accepted after clean H100 producer `38903` and independent validator `38907`.
The receding controller correctly identified a new unsafe window at action 185,
but its best exact five-action SQP candidate achieved only `+0.369923 mm`, below
the registered `+1 mm` buffer. It therefore stopped with zero contact and zero
CAR but without task completion. Learning a direct correction field from this
oracle is not justified because the teacher has no accepted action at an
encountered state. First test earlier intervention or a longer/richer
task-rejoining candidate family; do not relax the buffer after observing the
outcome.

## ADR-0069: Bind executable field recovery to archived action 185

Accepted. Do not interpret array `38848` scientifically. The executable
field-recovery comparison is bound to the exact archived pre-action-185 state;
the ordinary one-step candidate filter is therefore disabled for actions
0--184, which execute byte-for-byte from the immutable successful released
AEGIS artifact. This is a protocol correction, not a safety result.

## ADR-0070: Do not train a field-mixing MLP from the late E05 repair

Accepted after corrected H100 producer `38854` and independent validator
`38867`. Fixed analytical, normal-only, normal-plus-tangent, and unrestricted
two-action post-hoc repairs all prevented every active-obstacle robot contact
and paper CAR from archived action 185 onward, but no arm completed the task.
Normal-plus-tangent therefore offers no task-success value beyond the fixed
analytical repulsion baseline. Learning is not justified from this gate. The
next safety-and-task experiment must intervene earlier and/or optimize a longer
task-aware horizon; late-denoising timing cannot rescue a task mode already
destroyed by the required approximately `0.89--1.00` normalized correction.

## ADR-0071: Compose the early task-rejoining detour with receding verification

Accepted after H100 job `38874`. The zero-sum five-action detour restored task
competence and the frozen policy completed, but L5 collision recurred ten
actions after the detour. Keep the validated detour unchanged and filter each
subsequent released-AEGIS action with the existing exact cloned-OSC candidate
filter. This directly tests whether early task preservation plus receding
physical safety can jointly achieve safe task success; it adds no learning.

## ADR-0072: Reject one-step post-detour filtering and retain the five-action horizon

Accepted after paired producer jobs `38874`/`38876` and independent H100
validator `38881`. The endpoint-preserving five-action detour converted the
archived `-9.313104 mm` minimum to `+1.747199 mm` and allowed the frozen policy
to complete, but unfiltered execution contacted L5 at action 197. The existing
one-step exact filter found no verified candidate at its first post-detour
decision (action 187), despite 87 cloned candidates, and therefore could not
preserve task completion. Persistent safety must be optimized over a receding
task-rejoining horizon. Do not train a field-mixing MLP or enlarge one-step
repulsion from these outcomes.

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

## ADR-0049: Test learned weights on controller-pulled physical repulsion

Accepted as the simplest decisive follow-up to the failed unconstrained MLP
gradients. The learned object is a monotone scalar safety potential over the
seven accepted L5--L7 rollout margins. Exact paired cloned-OSC perturbations
provide the local Cartesian pullback. Consequently its action gradient is a
nonnegative combination of physical clearance-increase directions; learning
chooses strengths but cannot reverse all geometry normals.

The design explicitly separates evidence for a physical repulsive mechanism
from deployability. It uses privileged online clone rollouts and only one
temporally adjacent E05 test state. It therefore cannot establish episode or
task generalization, real-time performance, or SafeLIBERO completion. Those
claims remain blocked even if the local direction passes.

The MLP is frozen from steps 182--184 before test step 185. Learned, fixed
soft-min and paired-random directions are evaluated at equal correction norm.
The learned force must improve exact clearance, outperform the fixed physical
field, and beat the matched random population. If it does not, the correct
conclusion is that learning adds no evidence beyond analytical repulsion. A
two-action receding continuation is conditional on the direction gate and is
not a substitute for full closed-loop task completion.

H100 jobs `38789` and `38791` now reject learned monotone weighting under this
contract. The learned direction beat every matched random direction, so the
physical controller-pulled repulsive basis is meaningful. But it was almost
the same as the fixed analytical field and slightly worse at both registered
radii. The local two-action region also contained no exact-safe candidate;
its best learned hard margin remained `-6.801919 mm`.

The accepted interpretation is not “repulsion failed.” It is that the MLP
learned an already available analytical direction and cannot compensate for
missing safe support. Do not enlarge the MLP, add a QP, or run closed loop
from this result. First establish exact safe support using fixed physical
repulsion under an independently preregistered earlier-intervention or
larger/iterative trust-region experiment. A learned task-conditioned force is
only motivated after that oracle support gate passes.

## ADR-0050: Test fixed repulsion inside denoising before the E05 boundary

Accepted as the smallest follow-up to ADR-0049. The experiment reuses the
analytical controller-pulled L5--L7 direction that already beat matched
random steering. It intervenes from the pi0.5 query at action 180 and targets
chunk slots 2--4, which execute as actions 182--184. Guidance is injected
inside only the final five Euler updates and is bounded in physical normalized
action units.

An equal-budget post-hoc arm is mandatory. Without it, any improvement could
be attributed merely to adding the same vector to the completed action rather
than to denoising interaction. Ordinary pi0.5 remains unchanged unless the
reserved opt-in envelope is present, and physical displacements are converted
to model coordinates by scale only. The arm executes only after exact cloned
OSC verification. Passing would establish local early fixed-force support;
it would not establish learning, generalization, invariance, or task success.

The initial-query `0.005` action tolerance from job `37054` is not a valid
late-query gate. After attempt `38806` exposed that unsupported extrapolation,
late live-versus-archived action differences are diagnostic only. This does
not weaken the paired comparison: ordinary and guided denoising are evaluated
from the same exact replayed state and live observation with the same explicit
RNG seed, while post-hoc repulsion is constructed from the ordinary live
chunk. No outcome or candidate clearance was observed before this correction.

Job `38808` rejects the specific hypothesis that a fixed force becomes more
effective merely by adding it during final pi0.5 Euler updates. It did improve
the ordinary exact proxy clearance, but less than adding the same registered
budget after denoising. The denoiser partially canceled the injected vector.
This is not evidence against analytical repulsion, and all three prefixes
were already positive-margin through action 184. Do not claim that the later
E05 collision was prevented: the comparative gate intentionally blocked
execution and no suffix was run.

Independent job `38809` validates this interpretation. Keep the result as a
NO-GO for fixed-vector injection during denoising and as apparatus evidence
that a reserved flow hook works. It is not evidence that early intervention
prevents the archived E05 collision, because the current live pi0.5 query did
not reproduce that archived query's dangerous action mode.

## ADR-0051: Establish the structured physical-field oracle ceiling first

Accepted after the denoising-insertion NO-GO. The dangerous archived actions,
not a new live policy sample, are authoritative for this gate. Post-hoc exact
cloned-OSC search separates three questions: whether any bounded two-action
repair exists, whether positive mixtures of physical clearance normals can
express one, and whether adding task-tangent motion preserves useful intent.

The tangent is not an arbitrary avoidance vector. It is the nominal end-
effector progress gradient projected into the nullspace of active safety
rows. Nonnegative normal weights retain the physical push-away interpretation.
Exact cloned rollouts, raw contact, episode-relative CAR, and end-effector
progress decide outcomes; affine derivatives only generate directions and
are recomputed after every iteration. Do not train an MLP unless a structured
arm finds a verified-safe, task-progressing repair. Do not call this online
control or task completion even if it passes.

## ADR-0052: Accept structured field expressivity, not online efficacy

Accepted after H100 producer `38834` and independent validator `38838`.
Both registered structured arms repaired the exact archived `-9.313104 mm`
two-action transition without raw L5--L7 contact or paper CAR and retained
more than 82% of nominal end-effector progress. The structured field gate
therefore passes: a positive mixture of controller-pulled physical clearance
normals is expressive enough, and adding the projected task tangent remains
compatible with the verified-safe repair.

This is not yet an online-control result. The smallest retained iterative
structured repairs have correction L2 about `0.89` and only micrometre-scale
positive margins. At matched one-shot norms, no arm is safe below radius
`1.0`. Accordingly, do not promote fixed inside-denoising injection, train an
unconstrained action predictor, or run an unverified closed loop from this
gate. The justified next learned object is only the structured mixture
coefficient/step selector, trained across boundary states and always compared
with fixed analytical mixtures and matched random directions. A positive
buffer plus fresh exact cloned-OSC verification remains mandatory before any
candidate executes.

## ADR-0053: Compare denoising timing by surviving correction norm

Accepted as a mechanism-only follow-up to the earlier uniform-final-five
inside-flow NO-GO. Equal injected forces are not equal interventions because
later policy vector-field updates can cancel an earlier perturbation. The
scientifically paired comparison therefore measures the final output XYZ
correction over the same three chunk slots and constructs a separate post-hoc
arm at that exact surviving L2 norm.

Three schedules isolate timing while holding the injected physical budget
fixed: uniform updates 5--9, a late ramp over updates 8--9, and update 9 only.
The opt-in OpenPI path accepts an explicit nonnegative ten-step schedule and
leaves the ordinary sampler unchanged. This test may identify reduced
cancellation; it cannot establish safety efficacy because the paired live
step-180 action is already safe and is not the archived dangerous mode. No
candidate executes regardless of outcome.

## ADR-0054: Accept late timing for cancellation, retain post-hoc for efficacy

Accepted after producer `38839` and independent validator `38840`. Delaying
the same physical injected budget from uniform updates 5--9 to updates 8--9
and then update 9 raised final correction survival from `41.01%` to `78.85%`
and `97.11%`. This directly validates denoiser cancellation as the mechanism
behind the earlier weak inside-flow correction.

Do not interpret this as a safety solution. Every compared live prefix was
already more than 54 mm proxy-safe and no action executed. At equal surviving
correction norm, the final-step arm was `0.228 mm` worse than post-hoc, which
is consistent with the two becoming equivalent as timing approaches the
completed sample. Retain late scheduling only if future work specifically
requires modifying intermediate flow states; otherwise, post-hoc exact-verified
correction is the simpler and strongest baseline for the archived collision.

## ADR-0055: Learn a local physical action field from paired long-horizon outcomes first

Accepted as the smallest mechanism test for the proposed rollout-supervised
repulsive field. The geometry backend remains the accepted seven L5--L7
ellipsoid slabs; Poisson is deliberately deferred. The learned object is not a
joint trajectory or collision classifier. It is the local action-space
direction that increases a quantitative long-horizon execution utility.

The task-completing five-action detour run supplies a fixed open-loop
continuation through its later action-197 collision. Pairing both signs of a
five-action endpoint-preserving correction against that identical continuation
allows the learned field to see later physical consequences while avoiding
policy-query noise. A separate short-horizon analytical ellipsoid field and
matched random directions determine whether counterfactual learning adds
information beyond ordinary repulsion or abundant safe directions. This is an
offline, single-state expressivity test; exact cloned OSC is label generation,
not the proposed deployment evaluator. Do not train an MLP or modify pi0.5
unless this strict mechanism gate passes.

## ADR-0056: Accept learned long-horizon direction, reject the small local safety mechanism

Accepted after producer `38929` and independent validator `38932`. Paired
counterfactual long-horizon outcomes do teach information that the immediate
analytical ellipsoid field misses: held-out directional Pearson was `0.9973`,
the learned correction improved exact long-horizon clearance while the fixed
field worsened it, and it beat all 32 matched random directions at the
registered norm. This supports the narrow hypothesis that rollout supervision
can learn how a five-action change affects a later physical collision.

It does not support collision avoidance. The action-197 contact remained,
hard clearance improved by only `0.248 mm` from an initial `-14.263 mm`, and
no sampled endpoint-preserving correction at L2 `0.1` was safe. Classify the
experiment as a strict safety NO-GO and do not train an MLP merely because the
directional metrics passed. First find a verified-safe member of a richer but
still task-preserving correction family. Only then is learning that field from
multiple states scientifically justified.

## ADR-0057: Diagnose pure risk, endpoint constraint, and safe support separately

Accepted before the decisive follow-up. Remove the task penalty only from the
field target, not from final task-preservation evaluation. Re-estimate the
field after every accepted correction because the worst future link/time
witness is nonsmooth and may switch. Include radius `1.0`, since earlier
structured-field evidence required correction L2 near `0.89`.

A no-learning derivative-free search is essential. Without it, another unsafe
learned path cannot distinguish a bad gradient from an empty candidate family.
Run exact endpoint preservation first; relax it only after the exact-support
search fails, so the result can identify whether the zero-sum condition caused
the failure. This remains an offline local oracle diagnostic inspired by policy
value learning, not a PNCBF or formal safety result.

## ADR-0058: Reject fixed-tail five-action task preservation, retain the learned direction mechanism

Accepted after producer `38955` and validator `38960`. Pure-risk refitting did
not materially change the learned direction, and the fixed analytical field
was correctly signed over its local five-action horizon but harmful over the
twenty-action outcome. The earlier result was not caused by the task penalty
or a repulsion-sign bug.

Exact endpoint-preserving search found no safe support even at correction L2
`1.0`. Removing endpoint equality allowed multiple zero-contact, CAR-free,
positive-clearance trajectories, but every such retained candidate displaced
the terminal end effector by more than 15 mm. Thus, avoid claiming that the
five-action action space lacks geometric avoidance. The missing component is
task-compatible rejoining beyond that five-action prefix. The next oracle
should provide a longer rejoin horizon or live receding policy feedback; do not
scale direct-field learning against this fixed unsafe tail.
