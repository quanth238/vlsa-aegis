# Archived R05A pre-execution apparatus history

Archived on 2026-07-15 (Asia/Ho_Chi_Minh) when the reviewed full-lifetime
sampled-current canary became the only active R05A experiment.

This file preserves completed operational history that no longer belongs in the
short active tracker. It is provenance, not a launch instruction. The
authoritative decisions and immutable evidence remain in `docs/decisions/` and
`evidence/r05a/`; nothing was deleted or reclassified by this archive move.

## Research boundary carried through every attempt

The active question was and remains whether a paired, simulator-verified
safe-progress action target can be transported through the exact frozen pi0.5
sampler into a budgeted time-dependent residual-velocity sequence. The target,
checkpoint, normalization, case, observation, instruction, policy noise,
solver, 128-update limit, active steps 5--9, first-five-XYZ mask, path budget,
per-step cap, and fidelity thresholds were not tuned from these attempts.

None of the entries below executed a teacher-generated action in the simulator,
measured collision-avoidance efficacy, authorized IFT-01, or trained a probe or
MLP. Finite nonconvergence is a bounded-search result, not an infeasibility
certificate.

## Chronology

| Stage | Exact execution | What it established | Why it did not answer R05A |
|---|---|---|---|
| IFT-00 synthetic contract | local, `evidence/r05a/ift00-synthetic.json` | Deterministic inverse-control implementation, exact sampler recurrence, mask/time/budget enforcement, explicit failure statuses, and unchanged no-control path | Synthetic implementation evidence only; no real checkpoint |
| IFT-00A attempt A | run `r05a-inverse-flow-canary-20260715a`; GPU `27714_0`; CPU validator `27715` | The held-GPU plus CPU-`afterany` transaction worked and the control tests passed 17/17 | A toy sampler assertion stopped before pi0.5; no teacher search |
| Attempt-A diagnostic | CPU job `27722` | Error `0.0074953213` satisfied the unchanged registered fidelity gates; ADR-0030 corrected the extra assertion without changing a tolerance | Apparatus diagnosis only |
| IFT-00A retry B | run `r05a-inverse-flow-canary-20260715b`; GPU `27726_0`; CPU validator `27727` | All allocation suites passed 17/8/10/12 with zero skips; real pi0.5 ran two deterministic finite 128-update searches | Both searches were nonconverged, and missing Linux-5.15 `memory.peak` prevented a validated final `results.json` |
| ADR-0031 CPU apparatus regression | run `r05a-adr0031-apparatus-cpu-20260715a`; task `27797_0` | Repaired result validation, exact suite counts, trace-derived status, and cgroup membership mapping passed | Exact host peak remained unavailable; no checkpoint or real teacher search ran |
| CG-00 telemetry capability | run `r05a-cgroup-v2-current-capability-20260715a`; task `27820_0` | Exact-job cgroup-v2 `memory.current`, finite `memory.max`, hierarchical events, and bounded sampling were usable on worker-1 | The 6,045,696-byte high-water described only the shell task and was not pi0.5 memory or a peak |
| ADR-0036 implementation | commit `d3d51d3d5fe4d318760c89a87113ee4c3c0500d7` | Full-lifetime exact-job sampled-current telemetry and CPU-only fail-closed publication passed 65 focused tests, the 491-test harness, 21 artifact/18 gate audits, and independent science/HPC/security review | No run ID or H100 job existed at this stage |

## Exact preserved evidence

- Attempt A: `evidence/r05a/ift00a-attempt-a.json`, SHA-256
  `bc1707688a16c54a8facfedfae874d17c9c123b4c57f5f18260d493cbd2cc998`.
- Retry B: `evidence/r05a/ift00a-attempt-b.json`, SHA-256
  `850e2b9d2e11d92342095c55ffeb0e94f9ec17fbd77de1188f3a65efd84cab0b`.
- Retry-B unfinalized payload SHA-256:
  `d5721d08747cd7c8f335057f2d89f7224d8921ac0bf1cba9bd16475fd3622d2b`.
- CPU apparatus regression:
  `evidence/r05a/adr0031-apparatus-cpu-a.json`.
- CG-00:
  `evidence/r05a/cgroup-v2-current-capability-a.json`.
- Controlling decisions: ADR-0028 through ADR-0036.

## Retry-B scientific diagnostic

Retry B reached the real frozen model. Both searches returned finite status 3
after 128 updates and failed closed without applying controls. The recorded XYZ
target maximum and RMS errors were `0.8631912` and `0.3312218`, outside the
unchanged `0.010` and `0.005` gates. This is strong negative evidence for the
frozen solver on the one preservation case, but the run did not produce the
complete telemetry-bound envelope required for an accepted R05A result.

The next valid experiment is therefore not a solver change. It is one
scientifically identical H100 canary using ADR-0036's honest full-lifetime host
telemetry and independent CPU publication. If that accepted canary is again
finite nonconverged, ADR-0031/0036 require stopping this registered solver
direction without tuning it or calling the target infeasible.

## Consumed identities and forbidden reuse

The four run IDs listed above and all listed Slurm IDs are terminal provenance
and must never be reused. Do not resubmit retry B, the CPU apparatus regression,
or CG-00. Do not use this archive as authorization for IFT-01, simulator
efficacy, label collection, probe training, or an unregistered retry.
