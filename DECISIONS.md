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
