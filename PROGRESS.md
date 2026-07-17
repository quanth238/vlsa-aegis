# AEGIS SafeLIBERO table reproduction

## Objective

Reproduce the translational-action portion of AEGIS Table 1 from the authors'
release, retain videos for every SafeLIBERO rollout, and explain both collision
and task failures without silently excluding apparatus failures.

## Current gate

`A01-reproduction-apparatus` is active.

Local implementation evidence on 2026-07-17:

- `./init.sh` passes: 56 tests, one optional dependency test skipped.
- The immutable manifest contains exactly 1,600 cases in 32 groups of 50;
  paired evaluation requires exactly 3,200 terminal episode results.
- Two independent read-only reviews permit a fresh capture-only canary from a
  clean synchronized commit. Capture performs reset, exact state restore,
  20 dummy settling actions, and lossless asset export only.
- The paired outcome canary is not yet authorized. Its reviewer requires an
  exact full pi0.5 checkpoint binding and a cross-arm validation receipt.

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
