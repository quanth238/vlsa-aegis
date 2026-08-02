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

## ADR-0025: Accept the complete population through independent terminal verification

Accepted on 2026-07-31. Run
`vlsa-table1-contact-authority-population-20260718a` is the terminal A04
population from source commit
`1592aa59361f431ba96c6ddcbebcb596f6c20853`.

Publisher retry `28940` reached its four-hour limit after producing the
validated prepublish receipt, aggregate, exhaustive case ledger, and strict
gallery, but before atomically writing the planned final publication receipt.
The timeout remains an apparatus event and is not itself accepted as evidence.
The population is accepted because later independent verification established
the same terminal facts without weakening a denominator or result check:

- prepublish receipt SHA-256
  `05df4c759a478069f1df3fe318c6f6237e65aeb667212f208ec94e2ab2a13eaf`
  has status `validated`, binds all 32 completed Slurm tasks, and records
  `complete_paired_population=true` and `no_results_dropped=true`;
- its result inventory contains exactly 3,200 `complete` artifacts and has
  SHA-256
  `f7f28e88b43ac62c183284ef55cff61bd1a195109c02bf67058ae73845269d50`;
- summary SHA-256
  `c2702d40d53436b89e48a5e68d743a76403d076a5c5539fa6c74a850af698330`
  binds 1,600 cases, two arms, 32 groups, and accepted-result ledger SHA-256
  `28822a58683cc54dab915e6f6bc56005bdd51f779138509c71a7c1aae2969285`;
- the independent verifier regenerated the exact complete aggregate from the
  local 3,200-result tree and found no payload, schema, case, arm, or pairing
  mismatch.

This decision does not rewrite job-28940 artifacts, does not relabel its Slurm
state, and does not permit a partial-population claim.

## ADR-0026: Accept the exhaustive terminal analysis and video binding

Accepted on 2026-07-31. A05 is complete because its analysis is an exhaustive
derivation from the accepted A04 ledger rather than a selection of examples.

- The 1,600-row paired case ledger has SHA-256
  `2280c3f1dc7e25755f650e7af0deb140263edec0679e3395db13a539b5a6a780`;
  its source report has SHA-256
  `6d1d74040c55512ad9035cdd4a2976c6eeae15c6d32e17aff6164164cdc44de4`.
- The strict 3,200-entry gallery has SHA-256
  `40781fa0a817931ad23bb12b2b7be2b858e16033f97ed18b7f76b86d291b2bb2`.
- The complete-population reanalysis report has SHA-256
  `9457175a69b84895cd2c8aa18b3fe29e291992e80178e9a692f66588a4492430`;
  its 1,600-row audit has SHA-256
  `acfbb7d3d3f2ebfccb556807a8f980fcc35f0510b7ef148deda5a7080573a7e8`.
- Independent verification recomputed all 1,600 case-row self-hashes and all
  3,200 mirrored MP4 SHA-256 values, totaling 2,263,857,516 bytes, with zero
  mismatch. The adversarial validation addendum has SHA-256
  `cb854b3d25be6452af67ec62c3d3b404883bac428d31d59b34a9238aab4fb259`.

The accepted interpretation retains all 359 AEGIS CAR failures and all 530
AEGIS task failures. The failure categories are observational routing labels,
not unique causal mechanisms. Paper CAR remains distinct from sampled contact
and continuous clearance. The 149 pre-action movable-base contacts are not
equated with the AEGIS proxy condition `h < 0`. The analysis reports 109
literal link-5/link-6 contact cases and 257 strict useful rescues, with zero
strict-zero-translation episodes.

## ADR-0027: Make static simulator-oracle full-body Poisson-CBF the next gate

Accepted. `P01-static-poisson-runtime` depends on passing A05 and is the sole
active feature. `A06-openvla` remains pending and is not a dependency of P01.

P01 is a feasibility experiment, not a retrofit of the frozen reproduction.
It must be opt-in and must preserve the default OpenPI sampler, released AEGIS
paths, and ordinary `env.step` behavior. The first experiment uses exact
static MuJoCo collision geometry and matched joint-velocity controller arms;
it does not claim learned perception, dynamic-obstacle, carried-object, or
learned-steering coverage.

H100 prerequisite job `33249` completed all eight registered checks. Its
result SHA-256 is
`a5d5a5cd17564d16873607b4cb7684d819e075a6088ba26932a72b99d015ec5f`.
The feasibility review and runtime probe SHA-256 values are respectively
`596b59661beaf4746781ef57afaf4aa6b09a7225db91996b183c5c587109b63d`
and
`7532bdce73597d1e0581e9af891846df289b80a9dc7777a8fe6acc0585249a38`.

The registered causal comparison is adapter-only versus the identical adapter
plus Poisson-CBF, with exact settled state restored into both arms. Shadow
mode must first prove action/outcome parity and validate full substep contact,
simulator clearance, voxel-set semantics, Poisson numerics, sample coverage,
Jacobian finite differences, QP residuals, and controller scaling. Infeasible
repairs and failed cases remain in the denominator.

## ADR-0028: Stage the first causal test as adapter-only versus link-5/6 Poisson-CBF

Accepted for P01 bring-up. The first active H100 canary contains exactly two
matched joint-velocity arms: the translational adapter alone and that same
adapter plus the static Poisson-CBF constraints on `robot0_link5` and
`robot0_link6`. End-effector-only and all-moving-link variants are future
experiments, not unexecuted arms in this result contract.

Both arms replay the exact actions actually executed by the completed
historical pi0.5+AEGIS translational episode; no policy server or new pi0.5
query runs. For the first canary this is an outcome-dependent 237-action
exposure, not the full 300-step Table-1 suite horizon. Task success, Paper CAR,
and correction utility are therefore reported only within that registered
exposure and cannot be substituted for a new Table-1 TSR/CAR estimate.

Active physics is permitted only after allocation-backed production-grid
numerics, exact unchanged-OSC replay parity, and read-only static-Poisson
identification pass on the same clean commit and immutable case inputs. A
contact-prevention observation is kept distinct from task-preserving useful
correction, all-robot contact avoidance, and safety achieved by stopping.

## ADR-0029: Fail closed on replay evidence and phase-correct contact timing

Accepted. H100 identification job `33451` is retained as a non-scientific
apparatus failure: it completed the expensive replay but crashed when a
post-run check treated a typed `(high_level, inner_control, physics_substep)`
cadence tuple as a scalar. No partial trace or inferred warning from that run
may authorize active physics.

The replacement gate must validate the complete serialized record both before
publication and after loading it as an active prerequisite. It reconstructs
the first protected-link contact from the settled, live-solver, and forwarded
post-state MuJoCo ledgers; requires `contact_distance_m <= 0`; reconstructs the
primary warning from exhaustive registered per-geom samples in the raw trace;
checks the bound-alpha CBF arithmetic and typed invalid-query evidence;
recomputes the first static-drift threshold crossing over exactly the selected
obstacle geoms; and binds state and observation hashes to exact shadow parity.
Summary fields alone are not authority.

Timing follows physical state boundaries. Poisson trace row `N` is the
post-integration state at `(N+1)dt`. A forwarded post-state contact labeled `N`
shares that boundary, while a live-solver contact labeled `N` belongs to
`N*dt`. Therefore warning lead is `contact_N - warning_N` for post-state
contact and `contact_N - warning_N - 1` for live-solver contact. Only a
strictly positive phase-correct lead can unlock the active canary.
The active gate additionally requires a scheduled 100 Hz filter update that
can consume the warning before contact; a merely positive 500 Hz lead with no
intervening controller decision is not actionable feasibility evidence.

Each warning sample ID is bound to the half-open ID range of its ordered
protected-geom component; a globally valid sample from another link cannot be
used as evidence. Contact records are bound to the resolved robot--selected-
obstacle collision pair, aligned MuJoCo geom names and orientation, and body
provenance. Settled and forwarded post-state physical-contact ledgers must be
exactly the nonpositive-distance subsets of their candidate-contact ledgers.
Each CBF diagnostic serializes its 3-by-7 point Jacobian. The consumer
recomputes `J qdot`, `grad(h)^T J qdot`, `alpha h`, and the CBF residual,
requires one identical instantaneous arm velocity across the callback, and
requires duplicate summaries of the same sample to be identical.

Before active physics, the source environment independently rebuilds the
selected-obstacle contact authority, resolved geometry, Poisson bundle,
protected-surface components, and full-robot sampling ledger. These records,
the arm DOFs, and the exact settled `mjSTATE_INTEGRATION` hash must equal the
shadow-identification construction. An internally valid trace for another
obstacle, field, sample set, or settled state cannot authorize either arm.

## ADR-0030: Treat tracking validity as a bidirectional completion invariant

Accepted after adversarial review. The active runner already terminates with
`controller_tracking_invalid` when measured joint velocity crosses the
registered 0.05 rad/s Linf or 0.02 rad/s cumulative-RMSE apparatus threshold.
The compact episode validator previously checked only the forward implication:
that this completion class required a crossing. It did not reject an
`executed` result that also reported a crossing. Because pair eligibility is
conditioned on `completion_class == executed`, a self-consistent rehashed
compact forgery could otherwise bypass the runtime gate.

The result contract now enforces the reverse implication needed for a positive
observation: `executed` requires that neither tracking threshold crossed. A
regression test constructs the formerly accepted forged result and requires
rejection. Runtime failure priority remains unchanged: static-obstacle drift
and realized-field/invariance failures may be reported instead of tracking
failure when both arise on the same completed physics callback, but no such
partial arm is eligible as executed.

All allocation-backed prerequisites are commit-bound. Therefore the otherwise
valid numeric and parity artifacts from clean commit
`ad738550ef3a8fef18cab73139a4d1ecbe0d18c1`, and identification job `33494`
from that commit, remain historical implementation evidence only. A new clean
commit must rerun numeric, exact parity, and complete phase-correct shadow
identification before active physics.

## ADR-0031: Publish the active canary receipt last and validate raw ledgers deeply

Accepted after adversarial review. A schema-valid compact arm result is not
sufficient evidence for this first active canary. The immutable
`run_receipt.json` is now published last and is reusable only after a deep
validator has loaded both arm traces, reconstructed their scientific
endpoints, recomputed the paired result, and verified every referenced file
and payload hash. A candidate receipt is validated before publication, and the
on-disk receipt is validated again after publication.

For the registered first canary, internal evidence is fixed rather than
self-described: 237 historical translation-only actions, the exact native
`on(akita_black_bowl_1, plate_1)` goal, the moka-pot obstacle tree, the Panda
grip site, the full declared robot--obstacle collision-pair population, and
the settled simulator/controller identity. Contact authority is reconstructed
from raw MuJoCo candidate ledgers. Every issued PSF command requires one solved
QP and a command record binding nominal velocity, executed velocity,
correction, normalization, and gripper. Provider-time and realized barrier
conditions, static-obstacle validity, tracking, motion, task, CAR, and
right-censoring are recomputed rather than trusted from summaries. An executed
arm with any fail-closed record is invalid.

OSQP solutions are canonicalized to the hard velocity bounds before the
returned command and residual diagnostics are computed. This keeps the
serialized command identical to the command accepted by the joint-velocity
adapter even when the raw solver vector lies a few floating-point ulps outside
an active bound.

This decision records what the then-current trace schema v2 contained.
Positive sample-to-OBB clearance could not be independently regenerated
without raw sample coordinates and obstacle OBB poses; compiled model arrays,
field arrays,
upstream manifests, protocol files, checkpoints, historical results, and H100
prerequisite payloads also remain external byte-level trust boundaries.
MuJoCo nonpositive contact is therefore the safety authority, and all external
artifacts must be validated separately before scientific interpretation.

Jobs `33494` and `33575`--`33577` predate this decision. Their artifacts are
preserved but cannot authorize active physics. Numeric, exact parity, and
complete shadow identification must rerun from one new clean commit and unused
immutable root before the active canary can be submitted.

## ADR-0032: Bind every shadow state to the frozen replay and official MuJoCo state

Accepted after the pre-identification adversarial audit. Aggregate sequence
hashes and self-hashed summaries are insufficient when a downstream active
gate is expected to consume them. Exact parity schema v2 therefore carries the
complete 237-entry action-boundary ledger and binds it byte-for-byte to
`HistoricalActionReplay.steps[*].simulator_state_sha256`. Its terminal value
must also equal the frozen historical terminal state. The identification
producer and active consumer repeat this external binding rather than relying
only on parity's aggregate sequence hash.

At substep cadence, the sole simulator-state authority is MuJoCo's complete
official `mjSTATE_INTEGRATION` vector. Parity records one hash for every one of
the 5,925 callbacks, with exact nested action/substep cadence, endpoints, and
ledger hashes. Identification records official state immediately before and
after every read-only callback query, requires equality, and matches the full
sequence to parity. The active prerequisite validator repeats that match. The
legacy `sim.get_state().flatten()` trace field is removed from schema v2
because it was neither the official complete state nor independently consumed.

This decision rejects coherent middle-state and terminal rewrites even when
ordinary parity, callback parity, and identification are all internally
rehash-consistent: the frozen historical replay remains unchanged and is the
external oracle. It does not claim cryptographic authenticity against an
attacker who can replace every upstream byte and the clean source together.
Raw protected geometry, Poisson field arrays, compiled model arrays, and
manifests remain explicit byte-level producer trust boundaries. MuJoCo
nonpositive contact remains the scientific safety authority.

Identification job `33610` was canceled by exact job ID after this gap was
confirmed and before a final artifact existed. Artifacts from numeric job
`33607` and parity-v1 job `33608` remain preserved implementation evidence but
cannot authorize schema-v2 identification or active physics. All prerequisites
must rerun from one new clean commit and immutable root.

## ADR-0033: Require realized arm motion before any nonzero-motion prevention claim

Accepted after an adversarial pair-level audit. Commanded joint motion is not
physical motion. The former pair gate could report every prevention and
task-utility eligibility flag as true when the Poisson arm issued nonzero
commands and corrections but measured joint-motion and end-effector-path
integrals were both exactly zero. Tracking validity did not close this path:
small issued commands can differ from zero realized velocity without crossing
the registered tracking-error thresholds.

Every narrow prevention claim now requires both nonzero issued safe-joint
motion and `measured_joint_motion_integral_rad > 0`. A zero realized integral
adds `psf_realized_no_nonzero_arm_joint_motion` to the base link-5/6
ineligibility reasons; all-robot, Paper-CAR, task-success, preservation, and
rescue claims inherit it. The pair diagnostics serialize both issued and
measured motion so independent reporting cannot silently substitute one for
the other. The deep run validator already reconstructs measured motion from
every 2 ms physics row and regenerates the pair record, so there is one
authoritative eligibility implementation.

End-effector path length remains a continuous diagnostic. It is not required
for the narrow arm-link contact observation because valid null-space elbow
reconfiguration can preserve an end-effector pose while moving the protected
links. Task success and native goal progress remain separate utility
endpoints. Exact positivity establishes only nonzero realized motion; a future
claim of materially useful motion must preregister a numerical/noise floor or
retention threshold before evaluation. No such threshold is selected from an
observed active outcome here.

This source change supersedes root `20260801d` for active authorization even
though numeric job `33640` and parity-v2 job `33641` passed completely.
Identification job `33642` was canceled by exact job ID at 12 minutes 28
seconds, produced no final artifact, and no active job was submitted. Numeric,
parity-v2, and identification-v2 must rerun from the next exact clean commit
and unused immutable root.

## ADR-0034: Audit every actual protected sample before active physics

Accepted after a pre-identification adversarial review. Synthetic Jacobian
tests and a small number of representative sample checks cannot establish that
the live link-5/6 sample transform, MuJoCo body ID, arm-DOF slice, and Poisson
gradient are jointly correct for the full constraint population. Runtime
protocol v2 therefore freezes an all-sample audit at the exact settled
`mjSTATE_INTEGRATION` state.

The audit perturbs cloned state only. It uses `mj_integratePos` rather than raw
qpos addition and reconstructs each full-`nv` tangent with
`mj_differentiatePos`. Every sample must pass all seven point-Jacobian columns
and nine coupled field directions under registered absolute and relative
criteria. Relative criteria are waived only under an explicit near-zero
numeric norm; absolute criteria still apply. Coupled stencils must keep the
base and both perturbations inside one exact valid trilinear cell. The largest
eligible eta is selected, all ineligible attempts and typed reasons are
retained, and no eligible stencil or detected cancellation is fail-closed.

Shadow schema v3 and active trace schema v3 bind the full audit to the exact
ordered protected-sample ledger, seven DOFs, runtime parameter block, and
settled-state hash. Independent consumers reconstruct every matrix, tangent,
query classification, derivative, tolerance, count, and hash. Shadow and live
audits must match in specification, state/sample binding, and complete
classification ledger. Exact full-payload float-bit equality across different
allocations is deliberately not required: both payloads are independently
validated against the same fixed tolerances, while classification equality
detects stencil, eligibility, cancellation, and pass/fail drift without adding
an unregistered cross-host numeric constraint.

Jobs `33653` and `33654` are preserved zero-exit numeric/parity evidence from
superseded commit `8fd43304cf16091035a8c9ebb2f5dd81a535fb2e`.
Identification job `33655` was canceled by exact job ID after this gap was
found and produced no final artifact. None can authorize active physics.

## ADR-0035: Enforce the feasibility ladder before the VLA active canary

Accepted after rereading the user's minimal feasibility plan. Its instruction
to not proceed until the current stage passes is binding. A complete and
actionable shadow warning is necessary but no longer sufficient to submit the
recorded-action active canary.

The required order is: exhaustive actual-sample differential audit; one-step
exact-state counterfactual immediately before the recorded collision; paired
manual control against one world-fixed static box; independently validated
adapter-only execution; then the link-5/6 PSF active canary. The one-step and
manual gates require exact paired state, complete 2 ms exposure, hard-QP
postchecks, measured tracking, simulator contact authority, nonzero correction,
and no shifted robot contact. Any claim that the manual filter gives useful
motion must use a material tangential/joint-motion retention threshold frozen
before the allocation, not mere positivity selected after an outcome.

The earlier text allowing a development-only active canary before the
counterfactual and manual gates is superseded. No active job was submitted
under that exception. Separate immutable artifacts and independent validators
must be implemented for both missing gates before an active submission can be
authorized.

## ADR-0036: Select Stage-13 from the warning and fail closed on an inadmissible nominal velocity

Accepted before Stage-13 H100 execution. Any earlier design that chooses the
latest controller boundary by working backward from contact is superseded.
Stage 13 uses the physical warning `W` and the first scheduled 100 Hz boundary
`B = ceil(W / 5) * 5`. The preserved obsolete complete diagnostic explains the
change: contact-derived `B = 4695` is outside the epsilon-buffered Poisson
field for 352 of 1,531 samples, whereas warning-derived `B = 4510` is
diagnostic all-valid and is 187 simulator substeps before `C = 4697`.

The exact endpoint-derived nominal `qdot` in that obsolete diagnostic also
appears to exceed the frozen `[-0.5, +0.5]` rad/s controller envelope on joints
2 and 4. Stage 13 must not rescue this input by clipping, projection, QP
execution, or arm physics. If the fresh same-commit reconstruction confirms
the violation, it publishes the complete typed negative
`preflight_inadmissible_nominal_velocity`. That outcome means the recorded
nominal transition is incompatible with the registered controller envelope;
it is not evidence that the Poisson field or QP failed.

The obsolete diagnostic motivates this fail-closed branch but cannot decide
the fresh result. Only a same-commit H100 artifact that validates completely
may establish the Stage-13 outcome. Stage 13 claims no more than the registered
one-step exact-state counterfactual and typed admissibility result. An
out-of-bounds nominal command blocks Stage 14 and active physics. P01 remains
`active` and cannot become `passing` from implementation or synthetic evidence.

The accepted artifact authority is also fail-closed. A physical-model v3
contract fixes the compiled `nq`/`nv`/`na` dimensions and the exact flattened
state layout `[time, qpos, qvel, act]`; appended user-data tails are forbidden.
The live joint-velocity controller must match frozen installed source,
configuration, Panda XML, cadence, scaling, gain, joint, and actuator
identities, and the independent consumer rehashes the three installed files.
Only the seven scalar-hinge Panda arm entries of the endpoint-derived velocity
are independently claim-bearing. Non-arm full-`nv` values remain producer
diagnostics and cannot support the Stage-13 result. These bindings apply to the
complete inadmissible terminal as well as the executable paired branch.

## ADR-0037: Preserve the H100 audit negative and redesign the tangent criterion before rerun

Accepted after the complete root `vlsa-poisson-link56-first-canary-20260801g`
identification. Jobs `33726`, `33727`, and independent prerequisite consumer
`33730` passed. Identification job `33731` published a complete retained
schema-v3 diagnostic and exited nonzero; independent consumer job `33732`
validated it as a negative that cannot authorize Stage 13 or active physics.
No downstream physics was submitted.

The failure is confined to the registered point-tangent roundtrip check. Every
one of 1,531 samples passes the actual analytic-versus-numeric point-Jacobian
criteria, and all 13,779 coupled Poisson field/Jacobian directions pass. Only
shared tangent columns 3 and 5 exceed the fixed `1e-10` velocity-space tolerance:
the error is `1.397779669787269e-10`, corresponding to approximately
`1.39778e-16` rad of displacement at the registered interval. The same
displacement roundoff passes when represented as half the requested velocity
over twice the interval. This scale dependence identifies a numerical apparatus
false negative; it does not test contact avoidance or establish Poisson-CBF
feasibility.

The root and all artifacts remain immutable. We will not relax the observed
threshold in place or reinterpret the producer failure as a pass. Any rerun
requires a new preregistered protocol, clean commit, and unused root. The new
tangent-integrity gate must use a principled scale-aware or displacement-space
roundoff bound, preserve the old value as a diagnostic, and rerun numeric,
exact-parity, independent prerequisite validation, and full identification.
Only a passing, actionable identification may authorize Stage 13. P01 remains
`active`.

## ADR-0038: Gate scalar-hinge tangent reconstruction with a two-stage binary64 proof

Accepted before any rerun of the retained root-g negative. Runtime protocol v3
replaces the scale-dependent `1e-10` rad/s authorization threshold with a
two-stage bound derived from the two scalar-hinge operations used by MuJoCo
3.2.3. For sign `sigma`, registered perturbation interval `h`, requested signed
velocity `w`, base position `q`, perturbed position `q_prime`, reconstructed
velocity `r`, binary64 unit roundoff `u = 2^-53`, and
`gamma_2 = 2u/(1-2u)`, every arm coordinate must satisfy both

`abs((q_prime-q)-h*w) <= abs(q)*u + abs(h*w)*gamma_2`

and

`abs(h*r-(q_prime-q)) <= abs(q_prime-q)*gamma_2`.

The first inequality audits scalar-hinge integration (`qpos += dt*qvel`); the
second audits scalar-hinge differentiation (`(qpos2-qpos1)/dt`). Comparisons
use exact rational representations of the observed binary64 values, not a
rounded derived tolerance. The old `1e-10` rad/s result remains frozen and
serialized as a non-gating diagnostic. It cannot authorize or reject the new
audit.

This is not an empirical relaxation. The root-g signature has
`1.397779669787269e-10` rad/s error but only
`1.397779669787269e-16` rad displacement error; it satisfies both derived
bounds. The scale-equivalent `0.5` rad/s over `2e-6` seconds produces the same
perturbed position and also passes. A `1e-8` rad/s reconstructed-velocity
corruption, a `1e-12` rad perturbed-position corruption, wrong sign, DOF swap,
unresolved nonzero perturbation, non-arm leakage, non-finite input, or
subnormal intermediate remains fail-closed.

Audit schema v2 serializes the exact full source `mjSTATE_INTEGRATION` binary64
bits and binds them to the independently supplied parity state hash. The
claim-bearing qpos prefix is fixed at offset one. Runtime v3 also freezes
MuJoCo version 3.2.3 and the Panda scalar-hinge topology: arm DOF, joint, and
qpos indices `0..6`, joint names `robot0_joint1..7`, and matching
`dof_jntid`, `jnt_dofadr`, and `jnt_qposadr` arrays. The pure consumer rejects
type-confused booleans, collusively rehashed state-bit substitutions, remapped
qpos authority, unknown fields, or legacy schemas.

Because the nested evidence contract changed, shadow identification is v4,
active trace is v4, retained differential-audit failure evidence is v2, and
the Stage-13 protocol is v2. The Stage-13 result and receipt schemas remain v1
because their outer field sets are unchanged; their embedded protocol identity
and prerequisites explicitly require runtime v3 and differential audit v2
through shadow identification v4, and therefore reject every older artifact.
The selection manifest and receipt are regenerated against the new runtime
identity. Root g cannot be retrospectively certified because it did not record
the required perturbed qpos and source-state bit evidence.

Primary implementation authorities are MuJoCo 3.2.3's scalar-hinge integration
and differentiation source and the robosuite 1.4.1 Panda joint definition. The
Panda joint-6 upper limit exceeds pi, so the rejected global-pi/gamma-8 draft
would have excluded a valid model state and is not used. A new clean commit,
unused immutable root, H100 numeric gate, exact parity gate, independent
prerequisite validation, and complete identification-v4 artifact are mandatory
before Stage 13. This decision changes apparatus only and supplies no safety or
feasibility result.

## ADR-0039: Bind the settled differential audit to parity boundary zero

Accepted before the first runtime-v3 rerun. Exact parity v2 recorded official
`mjSTATE_INTEGRATION` hashes only after physics callbacks began, so it had no
independent official-state authority for the settled boundary before action
zero. Shadow identification could serialize a self-consistent construction
state, state bits, and differential audit, and its independent consumer could
authorize Stage 13 using only values from that same artifact. Stage 13 would
still reject a mismatch against a fresh live environment before paired physics,
but that later refusal did not make the prerequisite authorization sound.

Exact parity therefore advances to v3. Both the ordinary and callback replay
capture, before their first `env.step`, the exact structured authority
`{physical_boundary: 0, mujoco_state_specification: mjSTATE_INTEGRATION,
state_vector_length, sha256}`. A pass requires the two records to be exactly
equal and adds the literal acceptance
`ordinary_and_callback_boundary_0_mjstate_integration_exact`. Boolean boundary
or length substitutions, missing or extra fields, malformed hashes, unequal
replay arms, or a length different from the compiled physical model fail
closed.

Shadow identification v4 was not yet released, so it incorporates this parity
v3 requirement before publication rather than advancing again. Its pure replay
validator now requires the external settled hash and length, validates the
exact six-field read-only audit, binds the length to the physical-model v3
`mjstate_integration_size`, and reconstructs the differential audit against
that external hash. The independent consumer advances to v2. A retained
differential-audit failure is now externally state-bound but remains only a
diagnostic because its exact protected-sample geometry ledger is still local to
the producer.

The uncommitted Stage-13 protocol v2 likewise incorporates parity v3. Its
producer, pure core, and independent consumer require parity boundary zero to
equal the identification settled state, validate the record length against the
compiled model, and preserve the structured authority in final dynamic
evidence. The independent consumer requires the exact parity-v3 and
identification-v4 acceptance field sets and checks ordinary/callback equality.
Old parity-v2 artifacts, including root-g job 33727, cannot authorize
identification v4 or Stage 13 and remain immutable apparatus history.

This is evidence-authority hardening, not a safety result. A new clean commit,
unused immutable root, numeric job, exact-parity-v3 job, independent prerequisite
validation, and complete identification-v4 job are all mandatory. No old
artifact can be upgraded or reinterpreted in place.

## ADR-0040: Accept the lean H100 result as directional feasibility evidence

Accepted after the user explicitly requested a direct feasibility test instead
of continuing the publication-grade ladder. The lean test is a separate,
opt-in exploratory branch and does not weaken or reinterpret ADR-0035 or any
earlier immutable artifact. Its claim is limited to one post-hoc 0.4-second
window with static simulator-oracle geometry.

H100 job `33907` completed `COMPLETED|0:0` from clean pushed commit
`b1854d6b82836757fdfed50ad0788791a62e651b`. Exact paired restoration and the
fixed adapter-only arm reproduce the link-5 contact at physical boundary 4677.
The adapter-plus-link-5/link-6 Poisson-CBF arm activates at boundary 4500,
solves and postchecks all 40 hard QPs, completes all 200 physics substeps, and
has no literal selected-obstacle contact from any robot collision geom. Its
conservative full-robot clearance lower bound is 24.82 mm. The active intervals
retain 73.36% of nominal command-motion integral, traverse 67.54 mm in
Cartesian space, and reduce target error by 41.57 mm, so the observation is
classified as motion-preserving correction rather than stop-only.

The accepted interpretation is empirical and offline. It establishes neither
task success nor a full-episode, population, perception, dynamic-obstacle,
continuous-time, or real-time safety claim. Tracking exceeds the diagnostic
thresholds, and 35 of 40 OSQP solves exceed the 10 ms controller period before
field/Jacobian and simulator overhead. Increasing the exploratory OSQP budget
from 10,000 to 50,000 iterations is disclosed as outcome-informed numerical
debugging; every accepted solution still has solved status and passes the hard
constraint postchecks. The maximum observed iteration count is 15,150.

The PSF arm's complete-window target progress is 53.33 mm, compared with
116.93 mm for adapter-only. This confirms continued useful motion but prevents
equating the result with task preservation. The follow-up independent validator
reconstructs the raw geometry-bound contacts, visible action/gripper pairing,
adapter commands, all 40 QPs, activation, clearance, active motion, tracking,
and timing without calling the producer classifier. It also rejects missing
restore authority, hidden settled/rollout contact, command-to-physics
substitution, forged clearance certificates, and false apparatus flags. Its 56
focused tests and the complete 632-test repository gate pass; 135
allocation-only tests skip as expected.

This result is enough to continue the full-body Poisson-CBF research direction,
not enough to mark the broader `P01-static-poisson-runtime` gate passing. The
next work should first test persistent OSQP with fixed full-row sparsity,
numeric updates, and cross-step primal/dual warm starts while retaining every
hard constraint and the independent postcheck. It must measure the complete
pre-physics pipeline and target 40/40 updates within 10 ms. Joint-velocity
tracking follows, then a frozen multi-case link-contact set with contact
avoidance and task success reported separately.

## ADR-0041: Test task preservation with one shared prefix and complete suffix

Accepted for the user's requested feasibility test. Re-running joint-velocity
control from action zero would replace a long, already validated successful
OSC trajectory and spend most computation before the relevant collision. The
minimal integration instead replays the exact historical AEGIS/OSC prefix once
through action 179, then branches adapter-only and adapter-plus-link-5/link-6
Poisson-CBF from the same complete pre-contact MuJoCo state. Both consume all
remaining recorded actions 180--236 with fixed cadence. This preserves a real
full recorded SafeLIBERO episode while concentrating the intervention on the
controller segment being tested.

The design is open-loop after the branched states diverge and must be reported
as such. It can establish one-case controller feasibility if the PSF arm still
reaches the native task goal, but it cannot establish closed-loop VLA recovery
or population safety. The historical AEGIS result remains the authority that
the original policy succeeds while contacting the moka pot with link 5; the
new adapter-only suffix must independently reproduce that contact.

A no-contact result is accepted only with complete suffix exposure, positive
full-robot clearance, material pre-contact correction, continued measured
joint/Cartesian motion, nonzero commands, and native task success after the
correction and at terminal. Contact avoidance without task success is a task-
failure result; insufficient post-correction motion is `STOP_ONLY`; any link-5,
link-6, or shifted robot contact is a collision failure. A PSF contact ends the
arm immediately and is interpreted from its complete pre-contact trace because
continuing after leaving the strict `h > 0` safe set would require an
unregistered fallback. Tracking error and solve time remain diagnostics, so
this feasibility experiment does not claim realized-velocity invariance or
real-time deployment.

The independent consumer binds the frozen suffix action array from every
entered controller update and reconstructs contact-authoritative `D_sim` from
the minimum raw signed MuJoCo contact distance rather than from a binary contact
flag. It does not reconstruct the static-obstacle
envelope from a second raw pose/velocity stream; those flags remain bound to
the exact producer commit and runtime protocol. This is an accepted limitation
for the requested lean feasibility test, not authority for a publication-grade
dynamic-obstacle or population claim.

## ADR-0042: Repair the independent activation-row namespace without rerunning physics

Accepted after terminal producer job `34120` and failed CPU consumer job
`34137`. The immutable producer stores the first material correction as an
`activation_trace` row. Consumer v1 independently reconstructed the correct
update but compared that stored row to the associated `command_trace` row;
their schemas are intentionally different, so the otherwise complete artifact
was rejected with the sole discrepancy
`activation_first_material_row_differs`.

The accepted repair associates each reconstructed material command with its
key-matched activation row and compares like with like. Command-trace authority
for QP arithmetic remains unchanged. The launcher separately binds the expected
producer commit and the clean consumer commit, and the consumer records its own
commit in the terminal summary. This is an apparatus-only correction: no
producer code, result bytes, physical replay, protocol, threshold,
classification, or acceptance rule may change. Job `34137` cannot authorize a
scientific conclusion; a new clean CPU-only consumer must return zero exit and
zero discrepancies against the original result SHA-256
`2601fa9087a19bfaea5e2ec3c3af86df41f5cffac07d1df3653c83feef755111`.
