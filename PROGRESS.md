# AEGIS SafeLIBERO table reproduction

## Objective

Reproduce the translational-action portion of AEGIS Table 1 from the authors'
release, retain videos for every SafeLIBERO rollout, and explain both collision
and task failures without silently excluding apparatus failures.

## Current gate

`A03-one-case-paired-canary` is active. The apparatus and capture gates passed
in dependency order; no population job is authorized yet.

Local implementation evidence on 2026-07-17:

- `./init.sh` passes: 87 tests, three optional dependency tests skipped.
- The immutable manifest contains exactly 1,600 cases in 32 groups of 50;
  paired evaluation requires exactly 3,200 terminal episode results.
- Two independent reviews accepted the translational comparison after strict
  action-ledger, pair-binding, method-failure, and infrastructure validation.
- A paired canary receipt can pass only if the frozen label matches the active
  obstacle, perception reaches `ready`, and at least one `aegis_qp` action
  executes with finite, valid OSQP diagnostics.
- The next allocation-backed steps are one replacement immutable pi0.5
  checkpoint-tree receipt using cross-worker-stable identity, followed by one
  fresh ordinal-100 paired canary.

Checkpoint hash task `28467_0` completed on worker-2 in 16 seconds. Its full
content-tree SHA-256 is
`7c81971fafcdbc677b0e8fd25b3bffd3d624abe4635323784f1918abdcef6f15`;
the small receipt was independently revalidated locally.

Paired-canary attempt `28468_0` terminated during asset preflight on worker-1
after one second, before policy, simulator, GroundingDINO, or QP execution.
The checkpoint bytes had not changed. The hash receipt included Linux
`st_dev=1048662`, while the same shared file is exposed as `st_dev=1048723`
from another cluster mount namespace with identical path, size, inode, mtime,
and ctime. The repair removes only mount-local `st_dev` from the cross-worker
identity and retains the full tree hash plus stable file metadata. Run
`vlsa-table1-paired-canary-20260717a` is immutable and remains an apparatus
failure; it is never reused.

The first capture case is frozen as ordinal 100,
`vlsa-t1-spatial-i-t2-e00`. This is the Level-I version of the qualitative
task shown by the user: pick up the black bowl on the stove and place it on the
plate. The capture is apparatus evidence only; it cannot establish either
baseline failure or AEGIS success.

Capture attempt `vlsa-table1-capture-canary-20260717a`, Slurm task `28460_0`,
terminated before reset on worker-1. Source/protocol preflight passed, but the
legacy Robosuite stack could not initialize EGL because the allocation cannot
open the host `/dev/dri` render devices. The run is preserved as an apparatus
failure with zero reset, settle, policy, perception, QP, or outcome actions.
The repair selects the OSMesa headless path already proven by this project's
VinUni workloads; it does not alter simulator state or policy behavior.

Capture retry `vlsa-table1-capture-canary-20260717b`, Slurm task `28461_0`,
successfully initialized OSMesa but terminated before reset when the
OffScreenRenderEnv constructor could not generate its temporary randomized
object placement. The author evaluator calls `np.random.seed(7)` before that
constructor; the wrapper had omitted this ordering while still calling
`env.seed(7)` afterward. The repair restores the author's pre-construction
NumPy seed. The failed run again executed zero reset, settle, policy,
perception, QP, or outcome actions.

Capture retry `vlsa-table1-capture-canary-20260717c`, Slurm task `28462_0`,
completed on worker-1 in 23 seconds from clean commit
`bce7737369328da06d62d3840ae577a52b46780f`. It executed exactly one reset,
one immutable-state restore, and 20 settling actions, with zero policy,
semantic-selector, GroundingDINO, filtering, MVEE, QP, or outcome calls. The
capture payload SHA-256 is
`4fea432ba3ebc62b2c115e0804ade2e28da7243520a04cbb2fe1adbadedc891e`.
Local revalidation independently reproduced that payload hash, the stored NPY
hash, and settled agent-view array hash
`b8bcd1a309fbfc900d18ed472853bbec56a9e8e1fa98c55462a4960782560e7b`.
Visual review identified the exact active obstacle as `blue moka pot`; the
one-row Codex label manifest was frozen before any outcome at
`labels/vlsa_table1_canary_labels.jsonl`, SHA-256
`2d4d1be5c0a4940c72eb452d00361f6a4935de3f5cbc1058c9671fff96a35a36`.
This is allocation-backed capture evidence only. No baseline or AEGIS outcome
has run.

Evidence established before implementation:

- Source begins at untouched upstream commit
  `57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b`.
- The Table 1 screenshot is the translational protocol, not the released
  full-action README command.
- SafeLIBERO contains 32 scenarios and 50 frozen initial states per scenario:
  1,600 episodes per method.
- The released translational evaluator cannot execute unchanged:
  it omits the required suite argument to `filtering_points`, rejects the
  `safelibero_long` suite name, and has an undefined infeasible-QP fallback.
- The released repository has no SafeLIBERO-only pi0.5 runner, no population
  manifest, no aggregator, and no structured result schema.
- The released GLM-4.5V selector has an empty API key. The user requested Codex
  labels instead; these must be frozen from pre-outcome images.
- VinUni has the pi0.5-LIBERO and GroundingDINO checkpoints. OpenVLA-OFT is not
  installed and the paper does not identify the exact public checkpoint used.

## Gate order

1. `A01-reproduction-apparatus` — manifest, capture, frozen-label contract,
   paired pi0.5/AEGIS evaluator, aggregation, tests.
2. `A02-one-case-capture` — allocation-backed settled-image capture only.
3. `A03-one-case-paired-canary` — one translational pi0.5/AEGIS pair with
   identical initial state and per-request flow-noise schedule.
4. `A04-population` — all 1,600 cases for both arms; every case terminal.
5. `A05-analysis` — Table 1 comparison, indexed videos, and failure taxonomy.
6. `A06-openvla` — optional OpenVLA-OFT row after its exact checkpoint and
   environment are preregistered.

No population job may launch before the paired canary is validated.

## Known interpretation limits

- Paper CAR is operationalized by the release as active-obstacle L1
  displacement strictly greater than 1 mm. It is not a direct collision,
  clearance, or whole-arm-contact measurement.
- Codex replaces only GLM semantic selection. Results must be named
  `AEGIS conditioned on frozen Codex labels`, not exact end-to-end AEGIS.
- The published Long-suite translational CAR of 79.63% cannot be produced by a
  single 400-episode population, whose resolution is 0.25 percentage points.
  This is an unresolved reporting ambiguity.
- Full pi0.5-versus-AEGIS evaluation is 3,200 rollouts and is expected to take
  roughly 50--100 GPU-hours under the two-GPU user limit.
