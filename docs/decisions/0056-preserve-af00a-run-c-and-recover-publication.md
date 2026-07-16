# 0056 — Preserve AF-00A run C and recover publication on CPU

Status: accepted terminal diagnosis and CPU-only recovery preregistration on
2026-07-16. This decision does not authorize another GPU execution or a later
research gate.

## Terminal evidence

Exact release `bd14f97eeffafd20525454db4d7a52614e1146c4` submitted immutable
run `r05a-actual-forward-cem-canary-20260716c`. GPU task `28281_0`
completed `0:0` on worker-1 in `00:03:18` with one GPU, eight CPUs, and
65,536 MiB host RAM. CPU publisher `28282` failed `1:0` on worker-0 in
`00:00:12` with `AF-00A raw payload source budget changed`.

The immutable source-contract, held-GPU, submission, and final-fingerprint
SHA-256 values are respectively
`dcd725f0de3797ad24f2445bd347f5a1bcc9d6044f121dc9eeb2370bcdddb0f0`,
`ddcebd30f77a95d617b7461af0c34b652fc18bfea51d6c2e00afd6f4d290463d`,
`0008bc0e4b55604cf6e708e4300de342aa7b927e765d64f4e90432eb2703a4b0`,
and `32c9cd064ba59abc90cc279be3989894dc2e69e80de94f0beef3d9bc794e6221`.
The raw payload, tensor archive, query ledger, and host telemetry SHA-256
values are
`00433437470ca7678a236c4778d5cf68160c8d3299f128b64985b1aeb142d5d1`,
`ada53973653ba7c21dab33cdbc4b4498c7b2b1840f2fc284bd3d3c76baf1d383`,
`619f15ad46390d6e55e74365dc2530e58b383592ec998842d4bbd898c7266b92`,
and `277d80ac479f6db3115fe7eb110a7b324708831a82efc846b72fed5a1fd36804`.
The failed CPU receipt remains immutable at SHA-256
`f8e9ef108630cad9a9691dba0827a8be68d3b15e41ee3de2e4e6cfaf03736d27`.
It published no `results.json`; every claim and next-gate flag that it records
is false.

## Root cause

R02 reports the binary64 norm `3.6398398429065115`. Recomputing the norm
from its exact serialized delta gives `3.639839842906512`, exactly one
binary64 ULP higher. The GPU producer intentionally stores the recomputed
value after the already-frozen absolute `1e-12` R02 consistency check. The
publisher incorrectly required that recomputed value to equal the reported
value as Python binary64.

Both binary64 values cast to the exact registered float32 budget
`3.6398398876190186`. The sealed tensor archive contains those exact float32
bytes and the exact per-step radius `0.7279679775238037`. The float32 budget,
not the diagnostic binary64 scalar, was passed to every residual schedule and
the fixed CEM implementation. This is a publisher field-semantics defect, not
a changed control budget or a GPU method failure. No tolerance may be changed.

## Independently reconstructed diagnostic

The exact payload, ledger, tensor archive, and immutable R02 bytes pass the
independent NumPy validator. It reconstructs all 534 policy requests, the
frozen PCG64 proposal sequence and CEM updates, exact source pairing, budget
and constraint arithmetic, duplicate schedules, canonical replays, and Arm-C
reversal. It selects pool index 465 / CEM query 464 and returns
`frozen_cem_negative`:

- Arm A objective `2218.1335502517986`; XYZ max/RMS
  `0.7806643492412315` / `0.35969860072305654`.
- Changed Arm B objective `2121.5404274315442`; XYZ max/RMS
  `0.7899218065691934` / `0.35177249996585547`.
- Arm C objective `2968.7447114541746`; XYZ max/RMS
  `0.9606940421772645` / `0.416131221925226`.

The registered XYZ max/RMS limits are `0.010` / `0.005`, and the full-action
max/RMS limits are `0.050` / `0.015`; all three arms miss by a large margin.
This reconstruction is diagnostic until the preregistered CPU publication
recovery succeeds.

## Recovery decision

1. Preserve every run-C byte, including the failed receipt. Never delete,
   rename, overwrite, or retrofit an original artifact.
2. Do not rerun the GPU, model server, sampler, CEM, simulator, or training.
3. Repair only publisher semantics: require the R02 reported binary64 value
   to equal the frozen reported value; recompute the norm from the exact R02
   delta using the existing `_source_delta` check; require the payload
   binary64 value to equal that recomputation exactly; and require reported,
   recomputed, payload, and frozen budget values to have the registered
   float32 bytes where applicable.
4. Verify source-bound repository files from Git objects at release
   `bd14f97e...`, not by substituting recovery-checkout bytes. Bind the clean
   recovery release and its files separately.
5. Record source task `28281_0`, original failed publisher `28282`, and the
   new recovery publisher as three distinct identities. Never impersonate or
   repurpose `28282`.
6. Permit exactly one held CPU-only recovery job on `main`: two CPUs, 8,192
   MiB, `00:15:00`, no GPU, no requeue, dependency `afterany:28282`. Release
   it only after exact terminal-state, resource, source-hash, failure-receipt,
   fresh-target, and transaction-receipt checks pass.
7. The sole new outputs may be canonical `results.json`, a separate
   `cpu-republication-validation.json`, and the preregistered recovery
   transaction receipts. A recovery failure must leave the original evidence
   unchanged and must not reuse its recovery namespace.
8. The implementation must be fail closed, pass focused and complete local
   gates, and receive independent scientific and HPC review. A later
   direct-child release may change only the recovery config and its release
   ADR before the one CPU submission.

## Claim boundary

Until recovery publication passes, run C remains officially unpublished and
apparatus-inconclusive despite the valid raw diagnostic. After publication,
the only allowed conclusion is that this exact one-case, fixed 520-query CEM
failed the four action-target fidelity gates. It is not evidence of
infeasibility, simulator collision avoidance, task progress, safety efficacy,
population generalization, novelty, or MLP learnability. It does not authorize
IFT-01, label collection, probe training, or residual-field MLP training.
