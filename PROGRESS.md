# Progress

Last updated: 2026-07-17 (Asia/Ho_Chi_Minh)

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

Accepted runtime-identity repair:
`5a5f3306b96f1491a4baa52657193afc1bfad2b5`.

Validated zero-GPU identity apparatus implementation:
`a669ef2dd15cc59996d332ead54ffcdf06552635`.

Terminal zero-GPU identity release:
`8415b659a46699757de1e99558713e56b95255b5`.

Terminal CFS-00A run-B release:
`12a7da69d3d34050709d08b9ca903a99f9d30862`.

Reviewed failed-derivative diagnostic implementation:
`768ab1c93824c1c8d089c65793fa55ee57257ecc`.

Terminal failed-derivative diagnostic release:
`3d44b2c5a4779725101d672ed29658a841c85541`.

Accepted terminal-transport normalization implementation:
`ec5d7419d6406d56291bf59ebc585cadd5686ab4`.

Terminal corrected numeric-diagnostic release:
`1e2d36de9cb009a07e75d3535ab6f7070c5ea34c`.

Terminal AF-00A source release:
`bd14f97eeffafd20525454db4d7a52614e1146c4`.

Reviewed AF-00A publication-recovery implementation:
`167dc022cb718eb68121dcf0aa61f7fd1f90a4b7`.

Terminal AF-00A CPU-republication release:
`eb3be3a86c1336a9090413cf7d6c83f28a0f365c`.

Reviewed TRL-00A implementation:
`0c79d0791cad5331a0cfb67f8a2c3c8d0c0e3e8c`.

Consumed TRL-00A release:
`786afe18bf177b4db1e808f6fedb317beac22be8`.

Active jobs: none. TRL-00A task `28311_0` and CPU publisher `28312` were
cancelled under exact user authorization before execution; both had zero
runtime, no start, and no node, and neither log exists. AF-00A GPU task
`28281_0` completed `0:0`; its original CPU
publisher `28282` failed `1:0`; recovery publisher `28291` completed `0:0`
and published the official `frozen_cem_negative` result. The earlier failed
Jacobian diagnostic from tasks `28222_0`/`28223` remains preserved.

Historical detail is preserved, not deleted:

- pre-inverse-flow chronology:
  `docs/archive/progress/2026-07-15-pre-inverse-flow-pivot.md`;
- R05A attempts A/B, CPU apparatus, and CG-00:
  `docs/archive/progress/2026-07-15-r05a-pre-execution-apparatus.md`;
- sampled-current launch A and exact cancellation:
  `docs/archive/progress/2026-07-15-r05a-sampled-current-launch-a.md`;
- sampled-current launch B, GPU diagnostic, and CPU publication failure:
  `docs/archive/progress/2026-07-15-r05a-sampled-current-launch-b.md`;
- immutable decisions: ADR-0028 through ADR-0063;
- compact terminal evidence: `evidence/r05a/`.

## Current research question

Given a Codex-frozen semantic obstacle label, does the public VLSA/AEGIS
GroundingDINO/geometry/QP safety layer prevent the collisions in the immutable
20-case frozen pi0.5 diagnostic while preserving task progress?

This is a paired collision-conditioned baseline diagnostic, not an estimate of
general SafeLIBERO benchmark performance. It does not test student
learnability or authorize probe/MLP training.

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

## Active gate: R06 / AEGIS collision-conditioned baseline diagnostic

ADR-0063 pauses R05A and activates R06 after the exact TRL-00A release was
cancelled before execution. ADR-0064 freezes the paired comparison. Because
the public GLM-4.5V key is unavailable, the user explicitly authorized
ADR-0065: capture the exact branch image, let Codex choose and freeze the
semantic obstacle label before seeing AEGIS outcomes, then run the original
GroundingDINO/filtering/MVEE/full nine-variable QP. R06 compares frozen pi0.5
against that Codex-label AEGIS arm from the same simulator state, observation,
instruction, explicit policy noise, nominal actions, and execution horizon.
It retains all 20 cases and separately reports the existing 17-case and
three-case strata, safety alone, task progress/completion, action modification,
stopping behavior, and joint safety-plus-progress.

The new gate has no paired canary or population result. The action-boundary
selection, exact pairing, 17+3 strata, metrics, fixed-denominator failures, and
canary identity are frozen. The current staged path is: one allocation-backed
canary capture and baseline reconfirmation, Codex label freeze, one paired
canary, then a separately released 20-case capture/label/population. This does
not evaluate the original GLM selector and cannot be called original
end-to-end VLSA/AEGIS.

Dependency setup job `28391` completed on `worker-2` with exit `0:0` and
installed the official GroundingDINO Swin-T checkpoint at the frozen SHA-256
`3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799`.
The matching config SHA-256 is
`172e80017f9395668a9cb5d1b8bd9d061c0e360471c6ed673c83b69bb14399f1`.
This is setup evidence only; the allocation runtime and simulator integration
remain unvalidated.

## Paused historical gate: R05A / TRL-00A reference-trajectory lift

The sealed AF-00A population diagnostic in ADR-0059 localizes the constant-
field failure. Equal-split Arm A directly supplied essentially the full target
direction (`alpha=1.0000000088`), but the changed Pi0.5 base field opposed it
(`alpha=-0.4001357184`), leaving terminal completion `0.5998642904`, far below
the necessary XYZ-RMS bound `0.9938440103`. All 520 CEM evaluations had
negative target-direction field feedback. The same evidence also shows the
fixed CEM was inefficient: only 3/520 samples beat Arm A and the first appeared
in generation 7. This rejects `Delta/t` on the canary, not state/time-dependent
flow steering or global reachability.

ADR-0060 therefore preregisters TRL-00A, the direct optimizer-free test of the
user's proposed conversion. From the exact paired zero trace it constructs six
fully specified reference states `ref_5..ref_10 = xbar + alpha*Delta`. At every
active Euler step the ordinary Pi0.5 model reevaluates its base field at the
arm's current state, and the opt-in sampler computes the residual velocity
needed to reach the next masked reference. Arm B projects each newly computed
increment to the unchanged `B/5` cap; raw Arm R measures the authority required
by the same construction. Every finite generated schedule is duplicated and
then replayed twice through the ordinary `residual_schedule` path.

The implementation is fail closed and finite-only: one scientific artifact
requires the exact 18-request ledger. A raw nonfinite value or any partial
terminal is apparatus-inconclusive and publishes no scientific result. The GPU
writes only raw evidence; a zero-GPU CPU `afterany` publisher independently
rehashes R02, AF-00A, checkpoint, normalization, source, transaction, telemetry,
and schemas, then reconstructs reference arithmetic, projection, float32
`c -> u -> dt*u`, recurrences, budgets, replays, objectives, gates, and outcome.

Local verification on 2026-07-16 passes the complete `./init.sh` gate: 753
tests with 259 declared dependency skips, plus 26 dependency-backed
producer/validator/preregistration tests and 14 HPC/publication contract tests.
The torch helper/structural subset passes 9/9 locally. Exact OpenPI policy and
sampler runtime tests are registered as a zero-skip allocation preflight under
the OpenPI interpreter; client/NumPy/HPC tests run separately under the LIBERO
interpreter. Independent scientific review reports GO with no open P0/P1. No
generated action has entered the simulator, and no probe or MLP has been
trained.

Historical AF-00A terminal context remains below for auditability.

IFT-00 passed as synthetic implementation evidence. There is now one accepted
real one-case teacher-search result, but it is `frozen_cem_negative` and no
successful transport mechanism has been validated. CFS-00A run B is a valid
published apparatus result, but it stopped before a candidate and therefore
is not a method result:

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
- ADR-0042's exact runtime-identity repair is now allocation-validated.  The
  separately released zero-GPU, non-array worker-1 job `28043` completed
  `0:0` in one second and matched both public launcher paths, direct link
  targets, fully resolved executables, and resolved-binary SHA-256 values.
  The immutable result SHA-256 is `3bda039c...b0de`; all eight repository
  bindings and the held/source/submission chain matched.  No interpreter,
  model, simulator, metric, training path, or GPU ran.
- CFS-00A run B completed on real pi0.5. It computed the zero-control baseline,
  autograd Jacobian, and all registered three-direction by two-epsilon finite-
  difference comparisons. The Jacobian gate failed before FISTA. The generic
  exception discarded the six numeric comparisons, so no Arm-B candidate,
  nonlinear replay, Arm C, or four-gate result exists. Arm A also lacks the
  duplicate, payload, canonical replay, and independent validation required
  for scientific reproduction. ADR-0047 therefore preserves the run as
  apparatus-inconclusive and authorizes only diagnostic persistence.
- The ADR-0047 diagnostic allocation reached the typed adapter terminal, but
  the unchanged WebSocket server appended its standard `server_timing` field.
  The paired client required the transported mapping itself to have only the
  reserved key and rejected it before extracting or serializing the numeric
  diagnostic. ADR-0048 classifies this as a client transport-normalization
  apparatus defect. No numeric Jacobian or finite-difference values were
  preserved in that consumed run.
- ADR-0048's strict client normalization then passed in exact corrected run
  `r05a-constrained-flow-fd-diagnostic-20260716b`. The full float32 `35 x 75`
  Jacobian and all six fixed comparisons were independently preserved and
  reconstructed. Every comparison failed, with relative errors `0.626` to
  `0.993`; the central derivatives themselves changed by `0.819`, `0.937`,
  and `1.187` relative L2 when epsilon was halved. The current autograd
  Jacobian is therefore not validated as an actual-forward local model at the
  registered scales. This retires the autograd-linearized submethod, not the
  broader time-dependent actual-forward transport hypothesis.

The terminal frozen scientific experiment remained exactly one case,
`crfs-1069f29a8d76463a`, on source host `worker-1`: one H100, eight CPUs,
exactly 64 GiB host RAM, two hours, array `0-0%1`, no requeue, plus one
zero-GPU CPU `afterany` validator. The corrected diagnostic preserved its
numeric values but stopped at the failed finite-difference gate. It did not
enter FISTA or create a candidate. No teacher- or policy-generated action was
executed in the simulator, and both checked-in configs are now fail closed.

ADR-0050 now freezes the direct replacement test. AF-00A uses only the
ordinary `residual_schedule` path: two exact equal-split repeats, 520
actual-forward CEM evaluations, selected/reversed replay controls, and 534
total policy requests on the same paired case. The objective, float32
`c -> u -> dt*u` transport, eight-ULP budget rule, final selection, and outcome
precedence are fixed before execution. Implementation and independent
publication-path review are complete. All 35 dependency-backed AF-focused tests and the
complete 673-test repository gate passed on 2026-07-16. Independent scientific
and Slurm/publication reviews found no P0/P1 blocker.

Launch A used release `6ff5d5c` and reserved held GPU task `28275_0`, but
VinUni normalized the exact task record's displayed `JobId` to the array parent
`28275` while preserving `ArrayJobId=28275 ArrayTaskId=0`. The submitter
expected the alternate `JobId=28275_0` spelling and stopped before creating a
CPU publisher or releasing the task. Exact task `28275_0` was cancelled with
elapsed `00:00:00`, no node, and no model or simulator execution. ADR-0052
classifies this as apparatus-inconclusive and permits only a narrow display-ID
normalization. The AF-00A config is again fail closed.

Launch B used release `778bfb1` and held task `28279_0`. The semantic
normalization was correct, but its inline `sed` required a space before the
record's first `JobId=` token and returned empty. The transaction again
stopped before CPU publisher creation or GPU release. Exact task `28279_0`
was cancelled with start `None`, elapsed `00:00:00`, no node, and no allocated
TRES. ADR-0054 replaces both inline extractors with one shared parser and adds
executed positive/negative tests using the real VinUni record shape. No
scientific execution or result exists, and the config remains fail closed.

Run C used behavior-tested release `bd14f97` and completed the scientific GPU
work. Exact task `28281_0` completed `0:0` on worker-1 in `00:03:18`; all 36
allocation-focused tests passed with zero skips, and the sealed payload,
tensors, 534-request ledger, host telemetry, and GPU telemetry are present.
CPU publisher `28282` failed `1:0` in `00:00:12` before creating
`results.json` because it compared the producer's recomputed binary64 norm
`3.639839842906512` with R02's reported binary64 norm
`3.6398398429065115` by exact equality. The values differ by one binary64 ULP
but both cast to the exact registered float32 budget
`3.6398398876190186` that the GPU used for every control.

Independent validation of the exact run-C payload, tensor archive, ledger,
and R02 source passes and reconstructs `frozen_cem_negative`. Arm A objective
was `2218.1335502517986`; changed Arm B improved it only to
`2121.5404274315442`; reversed Arm C was worse at
`2968.7447114541746`. Arm-B XYZ max/RMS error was
`0.7899218065691934` / `0.35177249996585547`, versus registered limits
`0.010` / `0.005`. No generated action entered the simulator. ADR-0056
preserved every original byte and ADR-0057 released one separately receipted,
CPU-only publication recovery. Job `28291` completed `0:0` on worker-0 in
`00:00:08` with two CPUs and 8 GiB requested/allocated, and no GPU. It
published schema-valid `results.json` SHA-256 `507f25bc...dae914` and recovery receipt SHA-256
`5462f875...cb632`; the failed original receipt remains unchanged at
`f8e9ef10...36d27`. The official outcome is now `frozen_cem_negative`.
ADR-0058 closes the fixed AF-00A search and both consumed release paths.

## What this experiment can conclude

| Validated terminal result | Interpretation | Required next action |
|---|---|---|
| `mechanism_pass` | Equal-split Arm A fails, but the changed selected actual-forward CEM schedule passes all four physical fidelity gates | One-case teacher-transport support only; stop and separately decide whether any efficacy gate is justified |
| `baseline_sufficient_no_incremental_support` | Equal-split Arm A already passes all four gates | The target is transportable, but AF-00A gives no evidence that CEM is needed |
| `frozen_cem_negative` | All 520 fixed evaluations and replays are valid, but selected B still misses a gate | Stop automatic expansion; this exact search is unsupported on the canary, not infeasible |
| `apparatus_inconclusive` | Pairing, Arm-A reproduction, numerical, determinism, constraint, replay, OOM, test, telemetry, schema, publication, or source-job failure | Preserve any valid component-level diagnostic, but make no teacher-transport or efficacy conclusion; repair or retire only the demonstrated component under a new decision |

Clearance and progress are not measured in IFT-00A because no generated action
is executed. They begin only in a separately authorized IFT-01.

## Current verification state

- ADR-0062 released TRL-00A from exact commit `786afe1` with source contract
  `140e7e13...93a9`, submission receipt `9ff2c0b8...6b77e`, and final
  fingerprint `2ef79dcb...4b16`.
- Exact task `28311_0` remained pending for resources with zero runtime, no
  node, and no allocated TRES. CPU publisher `28312` remained unstarted on its
  dependency. Under explicit user authorization, only those exact jobs were
  cancelled. Slurm records both terminal with zero runtime, start `None`, and
  node `None assigned`; both exact logs are absent.
- The immutable run root contains only twelve hash-recorded control-plane
  artifacts and no case directory, `results.json`, or publication receipt.
  ADR-0063 and `evidence/r05a/trl00a-cancelled-before-execution.json` consume
  the identity and supply no positive or negative TRL scientific result.
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
- ADR-0043/0044 preregistered and released only the shell identity check.
  Exact job `28043` is terminal `COMPLETED|0:0` on worker-1.  It requested one
  CPU and 256 MiB; Slurm reported two allocated logical CPUs plus 256 MiB,
  consistent with worker-1's two threads per core and live `CR_CORE_MEMORY`
  granularity.  No GPU TRES was requested or allocated.  Held, source,
  submission, result, and log SHA-256
  values are `30be8432...3b990`, `45656005...0fbc`, `bef280b1...d1d9`,
  `3bda039c...b0de`, and `a8383717...8d1`.
- The full 1,298-line VinUni H100 Server Guide was audited at SHA-256
  `acee44c...b108`.  The saved preflight `c351ec19...5221` showed an empty
  queue, healthy worker-1, live 16-CPU/256-GiB/two-GPU-equivalent ceiling,
  `/mnt/data` at 73%, and no login compute process.  Source synchronization
  was Git-only; no bulk transfer, broad scan, cleanup, cancellation, or
  service hosting occurred.
- ADR-0045 accepts only the runtime-identity apparatus pass and consumes its
  ID.  It proves the standalone validator accepts the frozen link chains but
  does not yet validate full CFS integration, gives no CFS or safety evidence,
  and gives no H100 authority.  The identity config and both CFS configs are
  fail closed pending a separately reviewed new release.
- ADR-0046 now binds that terminal evidence into the fresh CFS apparatus while
  preserving the exact scientific projection
  `7dc2c8f6...dc915`.  It also binds the complete 1,298-line VinUni guide at
  SHA-256 `acee44c...b108`, requires a fresh live preflight, keeps all compute
  inside Slurm, pins the single H100 task to worker-1, and preserves the
  zero-simulator-action boundary.
- The ADR-0046 implementation passes the complete `./init.sh` gate: 611 tests
  pass with 198 declared dependency skips, plus all 21 artifact and 18 gate
  audits.  Shell syntax, JSON parsing, Python compilation, source hashes, and
  whitespace checks pass.  Two independent reviews report GO with no material
  scientific, source-binding, publication, or HPC defect.  This is apparatus
  evidence only, not a CFS result.
- Release `118ff0a` selected run ID
  `r05a-constrained-flow-same-budget-canary-20260716a` and created held array
  job `28047`, but the submitter rejected the parent display token
  `ArrayTaskId=0%1` before registering the CPU publisher or releasing the GPU
  task.  Exact task `28047_0` had the required `ArrayTaskId=0`.  The held job
  was inspected and cancelled with zero runtime, no node, and no allocated
  TRES.  No Python, model, arm, or simulator action ran; this identity is
  apparatus-inconclusive and consumed.
- Corrected release `12a7da6` submitted immutable run ID
  `r05a-constrained-flow-same-budget-canary-20260716b` exactly once. GPU task
  `28048_0` completed `0:0` on worker-1 in `00:03:12`; CPU `afterany`
  publisher `28049` completed `0:0` on worker-0 in `00:00:02`. The immutable
  source-contract and submission SHA-256 values are `5cbc304f...f2716` and
  `ad7d9764...8e98e`; all 58 repository bindings matched.
- The CPU publisher independently validated and atomically published
  `results.json` SHA-256 `655b48f...315f` with status
  `apparatus_inconclusive`. All 91 allocation tests passed with zero skips.
  Sampled host high-water was 16,136,884,224 bytes (about 15.0 GiB) under the
  unchanged 64-GiB limit; `max`, `oom`, and `oom_kill` deltas were zero. GPU
  sampled high-water was 8,513 MiB. This excludes resource, OOM, and
  publication failure as the stopping cause.
- The exact failure was `registered finite-difference validation rejected the
  autograd Jacobian`. It occurred after all six derivative comparisons and
  before FISTA. Those numeric arrays were not persisted, so precision,
  derivative-interface mismatch, and local nonsmoothness remain unresolved.
  The transport hypothesis is untested, not refuted. ADR-0047 and
  `evidence/r05a/cfs00a-same-budget-launch-b.json` preserve the exact boundary;
  both CFS configs are fail closed and no H100 resubmission is authorized.
- The ADR-0047 diagnostic-only implementation passed its pre-release gates.
  Dependency-backed core/adapter tests passed 29/29. The NumPy-backed paired
  client, independent semantic validator, CPU publisher, and HPC contract
  passed 54/55 tests; the sole local skip was the PyTorch-only canary audit.
  Independent review found no release blocker. The complete `./init.sh` gate
  passed 625 tests with 205 declared dependency skips plus all 21 artifact and
  18 gate audits. These were implementation checks, not allocation evidence.
- Those checks proved the typed path can preserve the exact 35-by-75 float32
  Jacobian, request- and config-bound float32 budget, and all six comparisons,
  and that the CPU validator can independently rebuild them. The allocation
  exposed one uncovered integration seam: the unchanged WebSocket server adds
  `server_timing` after the adapter returns its one-key terminal mapping.
- Exact release `3d44b2c` submitted run
  `r05a-constrained-flow-fd-diagnostic-20260716a`. GPU task `28212_0`
  completed `0:0` on worker-1 in `00:03:06`; CPU publisher `28213` completed
  `0:0` on worker-0 in `00:00:10`. Source contract
  `7c8eee72...bed3b99` bound 60 repository paths, and submission
  `73eb2320...f283cd` passed.
- The CPU job atomically published result `5c843ce5...1062e0` and receipt
  `be91ec15...d32d0`, both `apparatus_inconclusive`. Raw payload
  `d423da9f...5cbb2d` records the exact client error: `terminal constrained-flow
  response must contain only its reserved key`. The adapter terminal existed,
  but its numeric Jacobian and comparisons were rejected before local
  extraction and therefore are absent from the immutable artifacts.
- All 99 allocation tests passed with zero skips. Sampled host high-water was
  16,074,977,280 bytes; sampled device high-water was 8,513 MiB; required OOM
  event deltas were zero. No FISTA, candidate, nonlinear replay, Arm C,
  four-gate evaluation, or generated simulator action ran. ADR-0048 and
  `evidence/r05a/cfs00a-fd-diagnostic-20260716a.json` consume the identity and
  authorize only strict client-side timing normalization after review.
- Exact corrected release `1e2d36d` submitted immutable run
  `r05a-constrained-flow-fd-diagnostic-20260716b`. GPU task `28222_0`
  completed `0:0` on worker-1 in `00:02:56`; CPU `afterany` publisher `28223`
  completed `0:0` on worker-0 in `00:00:06`. Source contract
  `956b4f8b...b78` bound 66 clean repository paths, and submission
  `7f36201e...e16c` passed.
- The strict client normalization worked. CPU publication atomically preserved
  raw payload `37edb7bc...6f41`, result `955d7b88...59aa`, and receipt
  `29e926ad...1200`. The independent validator reconstructed the exact
  float32 `35 x 75` Jacobian, budget, both epsilons, all directional products,
  all error matrices, and all six failed checks.
- Relative Jacobian errors were
  `[[0.716, 0.626], [0.676, 0.868], [0.703, 0.993]]`. Halving the perturbation
  changed the central derivative by `0.819`, `0.937`, and `1.187` relative L2
  in the three fixed directions. The mixed-precision `bfloat16` cast is a
  credible mechanism, but it was not isolated from all other forward
  numerical effects. No tolerance was changed.
- All 99 allocation tests passed with zero skips. Sampled host high-water was
  16,072,122,368 bytes under the unchanged 64-GiB limit; sampled device
  high-water was 8,513 MiB; and required OOM deltas were zero. No FISTA,
  candidate, nonlinear replay, Arm C, simulator efficacy, or generated action
  ran. ADR-0049 and
  `evidence/r05a/cfs00a-fd-diagnostic-20260716b.json` consume the identity,
  retire only the autograd-linearized CFS-00A route, and close both configs.
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

Finish and independently review the R06 capture runner, allocation-preflight
the now content-bound GroundingDINO assets, and release one capture-only canary for
`crfs-1069f29a8d76463a`. The capture must select boundary 20, reproduce the
baseline collision twice, and save the same-state 1024 image. Codex then
freezes one image/instruction-bound semantic label under ADR-0065. Only after
that label and the full GroundingDINO/AEGIS apparatus validate may a separate
paired canary run. The 20-case population remains a later release.

Do not call the Codex-label arm or a simulator-geometry substitute the exact
end-to-end AEGIS method. Keep the existing 17 eligible cases and three retained
late/no-witness cases as separate strata, report every failure, and distinguish
safety alone from joint safety-plus-progress.

Do not resume job `27928`, jobs `27962`/`27963`, or jobs `28021`/`28022`; do
not reuse any consumed launch ID; and do not republish launch B's observed
payload. Do not resubmit retry B, CG-00, or the consumed CPU apparatus run.
Do not resume or reuse jobs `28212_0`/`28213` or their run ID.
Do not resume or reuse jobs `28222_0`/`28223` or their run ID.
Do not resume or reuse jobs `28281_0`, `28282`, or `28291`, or the run-C ID.
Do not resume or reuse jobs `28311_0`/`28312` or the consumed TRL-00A run ID.
Also, do not launch IFT-01 from this gate.

## Non-negotiable stops

- Do not launch `r03a-analytic-kill-population-20260715a`.
- Do not resubmit TRL-00A or synthesize its absent case/result/publication
  artifacts.
- Do not modify or rerun the consumed IFT-00A/CFS-00A worker, case, target,
  checkpoint, noise, solver, iterations, learning rate, tolerances, mask,
  active times, path budget, or per-step cap. A derivative-free optimizer may
  be defined only in a separate preregistered actual-forward experiment.
- Do not append the final action delta, clip the returned action, overwrite an
  endpoint, or increase the correction budget.
- Do not interpret clearance without progress as success.
- Do not train a probe or residual-field MLP unless a separate untouched-group
  IFT-03 authorization passes.
- Do not use the 17 development groups for student training or final testing.
