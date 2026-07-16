# 0051 — Release the actual-forward CEM canary

Status: execution release accepted on 2026-07-16 for one immutable AF-00A
submission only.

## Reviewed implementation

The fail-closed implementation is exact commit
`257b31ffdd50759e7d05472675363d38ae386a71`. It passed all 35
dependency-backed AF-focused tests and the complete 673-test repository gate.
Independent scientific and Slurm/publication-path reviews reported no P0/P1
blocker. The implementation parent authorizes no H100 execution.

## Exact execution release

- Immutable run ID: `r05a-actual-forward-cem-canary-20260716a`.
- Source host: `worker-1`.
- GPU task: partition `main`, account/QOS `normal`, one H100, eight CPUs,
  65,536 MiB host RAM, time limit `02:00:00`, array `0-0%1`, no requeue.
- Publisher: CPU-only `afterany`, two CPUs, 8,192 MiB host RAM, time limit
  `00:15:00`, no GPU, no requeue.
- Single submission: `true`.
- Automatic resubmission: `false`.
- Automatic next experiment: `false`.

This direct-child release changes only the AF-00A config and this decision.
It does not change the frozen case, target, policy noise, correction budget,
520-query CEM, 534-call ledger, objective, constraints, fidelity gates,
selection rule, resource envelope, or outcome precedence.

## Claim boundary

This release tests only whether the exact one-case safe-progress target can be
transported through the frozen pi0.5 actual forward sampler by the fixed
time-dependent residual schedule search. It executes no generated action in
the simulator. It authorizes no collision-safety, task-progress, efficacy,
generalization, novelty, latency, or infeasibility claim; no IFT-01 launch,
label collection, probe training, or residual-field MLP training follows
automatically from any result.
