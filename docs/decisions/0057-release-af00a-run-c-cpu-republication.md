# 0057 — Release AF-00A run-C CPU republication

Status: exact one-job execution release accepted on 2026-07-16. This release
authorizes publication recovery only; it does not authorize another GPU run or
a later research gate.

## Reviewed implementation

The accepted fail-closed implementation is exact commit
`167dc022cb718eb68121dcf0aa61f7fd1f90a4b7`. It preserves every immutable
run-C artifact and the failed publisher receipt, verifies the original source
files from Git objects at release
`bd14f97eeffafd20525454db4d7a52614e1146c4`, and separately binds the clean
recovery release and its runtime closure. Source GPU task `28281_0`, original
publisher `28282`, and the new recovery publisher remain three distinct job
identities.

The publisher correction does not tune a tolerance or modify the control
budget. It distinguishes the R02-reported binary64 norm
`3.6398398429065115` from the exact-delta recomputation
`3.639839842906512`, while requiring both to cast to the already registered
float32 budget `3.6398398876190186` (`0x4068f323`) used by every GPU query.

The implementation passed 12/12 recovery-focused tests, 29/29 existing
AF-focused tests, and the full repository gate with 686 tests passed and 225
expected dependency skips. An independent adversarial review found no P0/P1
blocker and approved this fail-closed implementation, a direct-child release,
and exactly one CPU republication.

## Exact execution release

- Source run: `r05a-actual-forward-cem-canary-20260716c`.
- Source release: `bd14f97eeffafd20525454db4d7a52614e1146c4`.
- Preserved source task: `28281_0`, terminal `COMPLETED 0:0` on `worker-1`.
- Preserved failed publisher: `28282`, terminal `FAILED 1:0`.
- New publisher: one held CPU-only job with dependency `afterany:28282`,
  partition `main`, account/QOS `normal`, two CPUs, 8,192 MiB host RAM,
  `00:15:00`, no GPU, and no requeue.
- Single submission: `true`.
- Automatic resubmission: `false`.
- Automatic next experiment: `false`.

The submission transaction must first verify exact terminal records, immutable
artifact hashes, original failure-receipt hash, source and recovery Git
bindings, fresh output targets, CPU-only ReqTRES, and the held-job record. It
may then release that one CPU job. The new result and separate recovery receipt
must be validated before publication; handled receipt failures roll back the
new result and never alter original evidence.

This direct-child release changes only the recovery config and this decision.
It does not rerun the model, sampler, 520-query CEM, simulator, metrics,
rendering, or training, and it changes no case, target, noise, budget, solver,
gate, or tolerance.

## Claim boundary

Successful publication permits only the conclusion that the fixed one-case,
520-query CEM did or did not satisfy its preregistered action-target fidelity
gates. It cannot establish infeasibility, simulator collision avoidance, task
progress, safety efficacy, population generalization, novelty, or MLP
learnability. It does not authorize IFT-01, label collection, probe training,
or residual-field MLP training.
