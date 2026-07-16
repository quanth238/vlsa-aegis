# 0062 — Release the reference-trajectory lift canary

Status: execution release accepted on 2026-07-16 for one fresh immutable
TRL-00A submission only.

## Reviewed apparatus

The accepted fail-closed implementation is exact commit
`0c79d0791cad5331a0cfb67f8a2c3c8d0c0e3e8c`. It implements the
optimizer-free online reference-trajectory lift preregistered by ADR-0060.
At each active Euler step the frozen pi0.5 base field is reevaluated at the
arm's actual controlled state. The opt-in sampler then computes the integrated
increment needed for the next fully specified reference state, either projects
that increment to the unchanged `B/5` cap or records it as the raw-authority
arm, converts it to residual velocity, and uses the unchanged ordinary Euler
statement. Every complete finite schedule is duplicated and replayed twice
through the ordinary residual-schedule path.

The implementation passed the complete repository gate: 753 tests with 259
declared dependency skips. The dependency-backed scientific and publication
suite passed 48/48 locally. The OpenPI sampler and policy runtime tests are
also registered as a zero-skip allocation preflight. Independent scientific,
validation, and Slurm/release reviews found no open P0/P1 blocker.

The GPU can produce scientific raw evidence only after the exact finite
18-request ledger completes. A nonfinite value, terminal response, partial
ledger, source mismatch, replay mismatch, telemetry failure, or schema failure
is apparatus-inconclusive and cannot publish `results.json`. A separate
zero-GPU CPU `afterany` job independently reconstructs the source pairing,
reference arithmetic, projection, float32 transport, recurrences, budgets,
replays, objectives, gates, and outcome before it may atomically publish.

## Exact execution release

- Immutable run ID:
  `r05a-reference-trajectory-lift-canary-20260716a`.
- Source host: `worker-1`.
- GPU task: partition `main`, account/QOS `normal`, one H100, eight CPUs,
  65,536 MiB host RAM, time limit `00:30:00`, array `0-0%1`, no requeue.
- Publisher: CPU-only `afterany`, two CPUs, 8,192 MiB host RAM, time limit
  `00:15:00`, no GPU, no requeue.
- Single submission: `true`.
- Automatic resubmission: `false`.
- Automatic next experiment: `false`.

This direct-child release changes only the TRL-00A config and this decision.
It changes no case, state, observation, instruction, noise, checkpoint,
normalization, R02 target, float32 budget, reference path, control mask,
projection, request ledger, replay contract, objective, gate, outcome rule,
schema, resource envelope, or tolerance. A healthy but occupied `worker-1`
may leave the released task pending; waiting is permitted, but rerouting or
resubmission is not.

## Claim boundary

This is one-case mechanism evidence only. It asks whether the privileged
action delta can be represented by a deterministic state- and time-dependent
residual field under the same registered control authority, and records the
raw authority required by this canonical lift. It executes no generated
action in the simulator. No outcome by itself establishes collision safety,
task progress, population generalization, global reachability or
infeasibility, novelty, latency, deployment readiness, or MLP/probe
learnability. This release does not authorize simulator efficacy testing,
IFT-01, label collection, probe training, or residual-field MLP training.
