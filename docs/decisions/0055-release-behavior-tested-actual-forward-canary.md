# 0055 — Release the behavior-tested actual-forward canary

Status: execution release accepted on 2026-07-16 for one fresh immutable
AF-00A submission only.

## Reviewed apparatus

The accepted fail-closed implementation is exact commit
`c33607281eab661980f31d7fa9e77dfe818fbf2a`. ADR-0054 preserves launch B as
a zero-runtime apparatus failure. Both exact-task identity consumers now use
one shared shell parser that was executed against the full retained VinUni
record. It accepts the exact parent and child display forms and rejects wrong,
missing, or duplicate identity fields while independently binding the array
parent and task zero. The parser is source-contract hashed and revalidated in
the GPU wrapper. All state, resource, ReqTRES, receipt, fingerprint, and
single-release checks remain unchanged.

The tree passed all 36 dependency-backed AF-focused tests and the complete
674-test repository gate, with 221 expected dependency skips. Independent
scientific/record and Slurm/release reviews reported no blocker. The scientific
config canonically matches the accepted AF implementation after release-only
fields are removed.

## Exact execution release

- Immutable run ID: `r05a-actual-forward-cem-canary-20260716c`.
- Source host: `worker-1`.
- GPU task: partition `main`, account/QOS `normal`, one H100, eight CPUs,
  65,536 MiB host RAM, time limit `02:00:00`, array `0-0%1`, no requeue.
- Publisher: CPU-only `afterany`, two CPUs, 8,192 MiB host RAM, time limit
  `00:15:00`, no GPU, no requeue.
- Single submission: `true`.
- Automatic resubmission: `false`.
- Automatic next experiment: `false`.

This direct-child release changes only the AF-00A config and this decision.
It does not change the case, observation, instruction, policy noise, R02
target, float32 correction budget, ordinary residual-schedule path, 520-query
CEM, 534-call ledger, objective, constraints, fidelity gates, selection rule,
outcome precedence, model, action, or resource envelope.

## Claim boundary

This release tests only whether the exact one-case safe-progress target can be
transported through frozen pi0.5 actual forward evaluations by the fixed
time-dependent residual schedule search. It executes no generated action in
the simulator. It authorizes no collision-safety, task-progress, efficacy,
generalization, novelty, latency, or infeasibility claim and no automatic
IFT-01 launch, label collection, probe training, or residual-field MLP
training.
