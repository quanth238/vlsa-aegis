# Progress

Last updated: 2026-07-16 (Asia/Ho_Chi_Minh)

Active branch: `agent/crfs-oracle-harness`

Baseline: THU-RCSCT/VLSA-Aegis commit
`57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b`.

Reviewed R05A sampled-current launch-B implementation:
`7d15c2c7921d9d1201638cb68ee48cc06a94ded1`.

Terminal sampled-current launch-B release:
`06b365b5899c2cb31db12187350cce48a3a0ea20`.

Validated exact-task accounting repair:
`5e595a366cb95d50ee86776f5de701627bf09669`.

Reviewed CFS-00A implementation:
`65bc57772c2648baa0e75a05d42161f01d9e3634`.

Terminal CFS-00A launch-A release:
`77bf9f639abe1da224762b0d8e63c64d85698811`.

Historical detail is preserved, not deleted:

- pre-inverse-flow chronology:
  `docs/archive/progress/2026-07-15-pre-inverse-flow-pivot.md`;
- R05A attempts A/B, CPU apparatus, and CG-00:
  `docs/archive/progress/2026-07-15-r05a-pre-execution-apparatus.md`;
- sampled-current launch A and exact cancellation:
  `docs/archive/progress/2026-07-15-r05a-sampled-current-launch-a.md`;
- sampled-current launch B, GPU diagnostic, and CPU publication failure:
  `docs/archive/progress/2026-07-15-r05a-sampled-current-launch-b.md`;
- immutable decisions: ADR-0028 through ADR-0042;
- compact terminal evidence: `evidence/r05a/`.

## Current research question

Can a paired simulator-verified safe-progress action target be transported
through the exact frozen pi0.5 sampler into a budgeted, time-dependent residual
velocity sequence?

This first tests teacher transport. It does not test collision avoidance,
task-progress efficacy, student learnability, or generalization.

## Baseline facts that the experiment must explain

- R01 direct safe-progress actions exist for all 17 eligible development
  groups: 17/17.
- R03's constant privileged residual transports only 9/17; its eight misses are
  four clearance-only failures and four progress-only failures.
- Equal-norm random, the registered static analytic arm, frozen, and bridge are
  0/17 in the accepted R03 comparison.
- The stronger R03A analytic smoke avoided contact on one case but failed
  progress: 0/1 joint Safe-Progress Success.
- Frozen IFT-00A scientific config SHA-256:
  `c31401867f3cdce2b3f443ad021c39dfb812f573b570e1e7434e1f149f79abfb`.

## Active gate: R05A / CFS-00A same-budget constrained-flow canary

IFT-00 passed as synthetic implementation evidence. There is still no accepted
real IFT-00A or CFS-00A result. CFS-00A is a new diagnostic, not a retrofit or
republication of sampled-current launch B:

- attempt A stopped before pi0.5 and is apparatus-only;
- retry B reached real pi0.5 and produced two deterministic finite 128-update
  misses, but its result was invalid because required host telemetry could not
  be finalized;
- CG-00 proved worker-1 can expose honest exact-job sampled-current telemetry;
- ADR-0036's corrected apparatus passed 65 focused tests, the complete
  491-test gate, all 21 artifact/18 gate audits, and independent scientific,
  HPC, and publication-security reviews.
- sampled-current launch A created held task `27928_0`, but a shell token
  parser rejected its valid adjacent `JobState=PENDING Reason=JobHeldUser`
  record before any receipt or release. The exact task was cancelled with zero
  runtime and no node. This is apparatus-inconclusive, not a transport result.
- sampled-current launch B ran GPU task `27962_0` to completion. Its two frozen
  128-update searches were finite and deterministically nonconvergent, but CPU
  publisher `27963` failed before Python publication because it compared
  Slurm's parent `JobIDRaw=27962` to exact task `27962_0`. No accepted result
  exists; the raw outcome is diagnostic only.
- ADR-0040 freezes one direct test of the user's corrected idea: compute the
  zero-control physical Jacobian with respect to the 75 integrated flow
  increments, solve one fixed same-budget coupling-aware candidate, and use it
  either directly or only as the initialization of the unchanged historical
  Adam solver.
- Arm A is the real historical baseline (`Delta_model / 5` initialization),
  not zero initialization. There is no separate decoded-action budget, no new
  planner target, no tolerance tuning, and no action overwrite.
- CFS-00A launch A used exact release `77bf9f6`, GPU task `28021_0`, and CPU
  validator `28022`. It stopped at `allocation_contract` because the runner's
  blanket no-symlink rule rejected the canonical OpenPI virtual-environment
  Python launcher. Both interpreter launchers are valid executable symlinks.
  No allocation test, model, arm, telemetry, payload, or generated simulator
  action ran. ADR-0042 classifies the run as apparatus-inconclusive and
  permanently consumes its identity.

The frozen scientific experiment remains exactly one case,
`crfs-1069f29a8d76463a`, on source host `worker-1`: one H100, eight CPUs,
exactly 64 GiB host RAM, two hours, array `0-0%1`, no requeue, plus one
zero-GPU CPU `afterany` validator. Launch A did not reach that scientific
boundary, and no teacher- or policy-generated action was executed in the
simulator.

## What this experiment can conclude

| Validated terminal result | Interpretation | Required next action |
|---|---|---|
| `mechanism_pass` | Exact Arm-A reproduction plus Arm-B nonlinear replay or Arm-C replay passes all four physical fidelity gates | One-case transport-mechanism support only; stop and separately decide whether any efficacy gate is justified |
| `frozen_method_negative` | Exact Arm-A reproduction plus valid deterministic finite B/C, but neither passes | Stop automatic expansion; this fixed method is unsupported on the canary, not infeasible |
| `apparatus_inconclusive` | Pairing, Arm-A reproduction, numerical, determinism, constraint, replay, OOM, test, telemetry, schema, publication, or source-job failure | Repair only the demonstrated apparatus defect under a new decision; no scientific conclusion |

Clearance and progress are not measured in IFT-00A because no generated action
is executed. They begin only in a separately authorized IFT-01.

## Current verification state

- ADR-0040 and ADR-0041 preregistered the baseline-faithful method and exact
  release transaction. Exact release `77bf9f6` bound source contract
  `8bf3501c...f96de` and submission receipt `0252d40a...bd01`; all 51 repository
  hashes matched the clean release.
- GPU task `28021_0` failed `2:0` on worker-1 at allocation startup with the
  exact message `missing or symlinked CFS-00A input:
  /mnt/data/quanth/venvs/openpi/bin/python`. CPU `afterany` job `28022`
  correctly observed source `FAILED|2:0`, failed `3:0`, and published no
  result. Both elapsed `00:00:00`.
- The OpenPI launcher resolves to the existing executable
  `/mnt/data/quanth/anaconda3/bin/python3.11`, SHA-256 `c7171890...16bc9`;
  the LIBERO launcher is likewise an intentional executable symlink. The
  demonstrated defect is the blanket no-symlink runtime-input assertion, not
  cluster capacity, memory, source pairing, or CFS mathematics.
- ADR-0042 and `evidence/r05a/cfs00a-same-budget-launch-a.json` preserve the
  terminal apparatus-inconclusive result. The checked-in configs are again
  fail closed with no execution release or H100 authorization.
- The dependency-backed CFS suite passes 69/69 tests across the linearized
  core, opt-in adapter, paired canary, semantic validator, CPU-only publisher,
  and exact Slurm transaction. The complete dependency-light `./init.sh` gate
  passes 590 tests with 198 declared dependency skips plus all 21 artifact and
  18 gate audits. Shell syntax, JSON parsing, source hashes, and whitespace
  checks pass. These are implementation evidence, not scientific evidence.
- The hardened source contract binds 51/51 local and remote paths, including
  the ordinary policy server, both overlay helpers, and all five Transformers
  replacement files. A content-addressed cache validates the reviewed base,
  replacement, and full-overlay tree hashes and rejects cached bytecode,
  symlinks, special files, or runtime-path redirection. Policy-server death,
  an unexpected shutdown status, or OOM cannot become clean terminal evidence.
- ADR-0037 release commit `223667c` selected launch-A run ID
  `r05a-inverse-flow-sampled-current-canary-20260715a` exactly once.
- Task `27928_0` is terminal `CANCELLED by 1073`, elapsed `00:00:00`, start
  `None`, node `None assigned`, and no allocated TRES. Its immutable run root
  contains only `launch-reservation.json`, SHA-256 `bec4583c5c046c7ca9f1155debc1f389d058daede42d3c485a39398f19c36ed4`.
- ADR-0038 freezes the apparatus-inconclusive interpretation and permits only
  independent exact-token Slurm parsing. The GPU/CPU adjacent-field success
  regression and wrong-state/reason/node fail-closed regressions pass locally.
  The focused sampled-current suite passes 38/38; the complete gate passes 505
  tests with 152 declared dependency skips and all 21 artifact/18 gate audits.
  Independent science, HPC, and publication-security reviews report no P0/P1
  finding.
- The apparatus is again unreleased: `ready_to_run: false`, no
  `execution_release`, and no replacement run ID. Frozen scientific content is
  unchanged.
- No checkpoint, pi0.5 search, IFT-01 rollout, label collection, probe, or MLP
  ran in launch A.
- Launch B source contract
  `b9a9e253c9a5e27815018c13c839805c97752545189085caac016db4ee81e1e6`
  bound 28/28 files to clean release commit `06b365b`. GPU task `27962_0`
  completed `0:0` on worker-1 in `00:03:09`; CPU `afterany` job `27963` failed
  `3:0` after `00:00:31` with `state=missing`.
- The raw payload SHA-256 is
  `4c85603c62446d74259820dc7f86451ffe53723b79e0c866d2e38d7a026622cd`.
  Both searches ran 128 updates, produced the same registered schedule/action
  hashes, stayed finite, and missed XYZ fidelity by maximum `0.8631912` and RMS
  `0.3312218`. This is not infeasibility. Failed controls were not applied.
- The CPU publisher never invoked Python; no hidden candidate or
  `results.json` exists. Policy steps, teacher steps, and efficacy rollouts are
  all zero. ADR-0039 preserves the exact no-claim boundary and consumes the run
  ID permanently.
- ADR-0039's exact-task repair and preregistered zero-GPU regression apparatus
  pass all five dedicated tests, including the complete fake SSH/Slurm
  transaction. The final full local gate passes 521 tests with 152 declared
  dependency skips plus all 21 artifact and 18 gate audits. Independent
  science, HPC, and publication-security reviews report GO for the repair and
  CPU-only regression with no P0/P1 finding; they explicitly remain NO-GO for
  an H100 retry.
- The live shell-only regression passed at source task `27975_0` and CPU
  `afterany` validator `27976`; both completed `0:0` on worker-1. The production
  helper observed exact display identity `27975_0|COMPLETED|0:0`, all nine
  bound files matched commit `5e595a3`, and result SHA-256 is
  `0de4b8b736bd750a82e7439cf737b9d16e248d67ee0d2f741d82f09d9bcd6b74`.
  Each job requested one CPU, 256 MiB, and zero GPUs. No Python, model,
  simulator, search, or training ran. This validates only the accounting path.

## Exact next action

Implement only ADR-0042's interpreter identity repair: preserve the canonical
launcher paths, bind their exact link chains, resolved executable paths, and
binary hashes, and retain no-symlink enforcement for immutable scientific
inputs. Run the focused CFS suite and `./init.sh`, obtain independent review,
then separately preregister one zero-GPU shell-only worker-1 identity check.
Do not select another H100 run ID or submit another canary yet. IFT-01 remains
blocked.

Do not resume job `27928`, jobs `27962`/`27963`, or jobs `28021`/`28022`; do
not reuse any consumed launch ID; and do not republish launch B's observed
payload. Do not resubmit retry B, CG-00, or the consumed CPU apparatus run.
Also, do not launch IFT-01 from this gate.

## Non-negotiable stops

- Do not launch `r03a-analytic-kill-population-20260715a`.
- Do not change worker, case, target, checkpoint, noise, solver, iterations,
  learning rate, tolerances, mask, active times, path budget, or per-step cap.
- Do not append the final action delta, clip the returned action, overwrite an
  endpoint, or increase the correction budget.
- Do not interpret clearance without progress as success.
- Do not train a probe or residual-field MLP unless a separate untouched-group
  IFT-03 authorization passes.
- Do not use the 17 development groups for student training or final testing.
