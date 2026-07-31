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
