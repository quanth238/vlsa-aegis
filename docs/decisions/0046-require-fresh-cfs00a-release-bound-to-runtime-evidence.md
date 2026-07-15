# 0046 — Require a fresh CFS-00A release bound to runtime evidence

Status: preregistered implementation-stage release protocol on 2026-07-16.
H100 submission remains forbidden until an independently reviewed
implementation commit receives the canonical direct-child release appendix
defined below.

## Basis

CFS-00A launch A is permanently consumed and apparatus-inconclusive under
ADR-0042.  It stopped before the allocation tests, model, or Arms A/B/C because
its runner rejected the canonical OpenPI Python symlink.  ADR-0045 subsequently
accepted zero-GPU worker-1 job `28043` only as a standalone runtime-identity
apparatus pass.  That job proves the frozen production link chains satisfy the
new validator, but it did not test the validator inside the full CFS workload
and grants no H100 or scientific authority.

This protocol binds ADR-0045, its compact evidence JSON, and its preserved
VinUni preflight into a new CFS implementation.  The previous ADR-0041 release,
run ID, and jobs remain immutable history and cannot be reused or rewritten.

The operating reference is the complete 1,298-line local file
`/Users/quanth238/Library/Mobile Documents/iCloud~md~obsidian/Documents/LLM Knowledge Base/10 Raw/articles/research-infrastructure/2026-05-03 - VinUni H100 Server Guide.md`,
read at SHA-256
`acee44c535e2fc25f8986e41efe233f21683a71c7fb5fa0ae726f0dae573b108`.
The repository binds that digest and implements the applicable controls; the
login node never reads this local path during submission.

## Frozen scientific boundary

The action-to-flow question is unchanged.  The scientific projection produced
by `constrained_flow_scientific_config_hash` must remain exactly
`7dc2c8f63838ae4e22db8a927d87cae89daf0025c931b33e225c946abe8dc915`.
The case, source state and observation, instruction, policy noise, target,
checkpoint, normalization, active times, mask, source budget, per-step cap,
Arm-A Adam solver, Arm-B physical Jacobian and projected FISTA solver, Arm-C
initialization, tolerances, duplicate schedules, canonical replays, and all
four physical fidelity gates cannot change.  Generated actions remain
forbidden from simulator execution.

Only release/apparatus plumbing may change:

- bind `docs/decisions/0045-accept-runtime-identity-regression.md` and
  `evidence/r05a/runtime-identity-regression-20260716a.json` by exact bytes;
- replace the consumed ADR-0041 release path with this decision for all future
  release validation and publication;
- include the complete VinUni guide login-process audit and fail closed when
  shared storage is at least 90%;
- require an empty user queue, healthy worker-1, at least 64 GiB free host RAM,
  and live QOS capacity for the frozen request before reserving a run root or
  calling `sbatch`; observe current worker-1 H100 occupancy exactly, while
  allowing the one pinned job to wait in Slurm when all H100s are occupied;
- retain exact held-source/CPU-afterany receipts, exact source hashes, and
  one-release/no-resubmit semantics.

## Two-stage release rule

The reviewed implementation commit must remain fail closed:
`ready_to_run=false`, nonempty blockers, `execution_release=null`, and no H100
authorization.  A later release commit must be its direct child and may change
only:

- `configs/experiments/r05a_constrained_flow_canary.json`;
- `configs/experiments/r05a_constrained_flow_canary_apparatus.json`;
- this decision.

The release commit must append the canonical `Exact execution release` section
and put the same closed `execution_release` object in both configs.  It must
select one unused immutable run ID and retain the frozen resource envelope:
one worker-1 H100, eight requested CPUs, exactly 64 GiB host RAM, two-hour
short-canary project limit, array `0-0%1`, no requeue, then one zero-GPU CPU
`afterany` publisher with two CPUs, 8 GiB, and 15 minutes.  This is a bounded
experiment, not a training or service template.

Immediately before the single submission, the exact VinUni guide preflight
must be rerun and inspected.  Live Slurm/QOS/storage state overrides examples.
If worker-1 has no free H100, record that fact and allow only this one pinned
job to remain pending in Slurm; do not reroute or increase concurrency.  The
allocation-local policy server is experiment IPC only and must be terminated
by the registered exact-PID trap.

## Canonical release appendix

At the implementation commit, this decision ends here.  The direct-child
release appends exactly:

1. heading `## Exact execution release`;
2. one-submission authorization;
3. accepted implementation commit, unused run ID, worker-1, and canonical
   resource JSON;
4. `false` for automatic resubmission, automatic next experiment, simulator
   efficacy, infeasibility, and probe/MLP training authorization; and
5. the unchanged no-IFT-01/no-tuning/no-generated-simulator-action boundary.

Until that canonical appendix exists, no H100 submission is authorized.
