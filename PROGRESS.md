# Progress

Last updated: 2026-07-15 (Asia/Ho_Chi_Minh)

Active branch: `agent/crfs-oracle-harness`

Baseline: THU-RCSCT/VLSA-Aegis commit
`57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b`.

Reviewed R05A sampled-current apparatus implementation:
`d3d51d3d5fe4d318760c89a87113ee4c3c0500d7`.

Terminal sampled-current launch-A release:
`223667c91b05be9ab403e4d92d0cd1a96b45246f`.

Historical detail is preserved, not deleted:

- pre-inverse-flow chronology:
  `docs/archive/progress/2026-07-15-pre-inverse-flow-pivot.md`;
- R05A attempts A/B, CPU apparatus, and CG-00:
  `docs/archive/progress/2026-07-15-r05a-pre-execution-apparatus.md`;
- sampled-current launch A and exact cancellation:
  `docs/archive/progress/2026-07-15-r05a-sampled-current-launch-a.md`;
- immutable decisions: ADR-0028 through ADR-0038;
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

## Active gate: R05A / IFT-00A mechanism canary

IFT-00 passed as synthetic implementation evidence. There is still no accepted
real IFT-00A result:

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

The active experiment is exactly one case, `crfs-1069f29a8d76463a`, on its
source host `worker-1`: one H100, eight CPUs, exactly 64 GiB host RAM, two
hours, array `0-0%1`, no requeue, plus one zero-GPU CPU `afterany` validator.
No teacher- or policy-generated action is executed in the simulator.

## What this experiment can conclude

| Validated terminal result | Interpretation | Required next action |
|---|---|---|
| `completed_converged` | The frozen pi0.5 sampler can realize this one paired target with the registered time-dependent vector-field residual | Mechanism pass only; stop and separately decide whether IFT-01 efficacy is justified |
| finite deterministic `completed_nonconverged` after 128 updates | Negative result for this frozen solver on the registered case | Stop this solver direction; do not tune and do not call the target infeasible |
| pairing, determinism, nonfinite, OOM, test, telemetry, schema, publication, or source-job failure | Apparatus-inconclusive | Repair only the demonstrated apparatus defect under a new decision; no scientific conclusion |

Clearance and progress are not measured in IFT-00A because no generated action
is executed. They begin only in a separately authorized IFT-01.

## Current verification state

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

## Exact next action

Run focused and complete gates plus independent review on the ADR-0038 parser
repair while the apparatus remains unreleased. Commit, push, and synchronize
that exact implementation. Then create one separately reviewed direct-child
release-only commit selecting a new unused immutable run ID and changing only
the apparatus release fields and ADR-0037. Submit the worker-1 64-GiB canary
once only after all release identities match.

Do not resume job `27928` or reuse launch A's run ID. Do not resubmit retry B,
CG-00, or the consumed CPU apparatus run. Also, do not launch IFT-01 from this
gate.

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
