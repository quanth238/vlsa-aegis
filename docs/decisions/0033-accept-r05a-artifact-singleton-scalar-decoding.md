# 0033 — Accept artifact-only singleton scalar decoding for R05A

Status: accepted on 2026-07-15 before the ADR-0032 CPU allocation, retry C,
or IFT-01.

## Context

Independent review replayed the exact retry-B `canary-payload.json`, SHA-256
`d5721d08747cd7c8f335057f2d89f7224d8921ac0bf1cba9bd16475fd3622d2b`.
Scalar trace leaves persisted by the real policy path have exact array shape
`[1]`, including `solver_status`, `cuda_memory_available`, and CUDA peak
counters. The live runner succeeded, but the downstream finalizer and semantic
validator decoded persisted leaves with the live-trace `_scalar` helper, which
requires shape `()`. Replaying the exact payload therefore failed at the first
CUDA-memory leaf before any result could be interpreted.

This is an artifact-decoding mismatch. It is unrelated to the teacher solver,
target, budget, compiled/eager tolerance, simulator, or efficacy outcome.

## Decision

1. Keep live sampler decoding unchanged: `_scalar` continues to require exact
   shape `()`.
2. Add a separate artifact-bound decoder. It may accept only one numeric or
   boolean value represented as shape `()` or the observed serialized shape
   `(1,)`; every other shape fails closed.
3. Use the artifact decoder only after a trace record has passed its existing
   array-record shape, dtype, finite-value, and SHA-256 validation. Use it in
   the independent semantic validator and memory finalizer, never in live
   teacher generation.
4. Bind a dependency-backed regression that reconstructs persisted `(1,)`
   scalar leaves, proves artifact decoding succeeds, proves a two-element
   value fails, and proves live `_scalar` still rejects `(1,)`.
5. Include this apparatus regression in the still-unused exact ADR-0032 CPU
   run `r05a-adr0031-apparatus-cpu-20260715a`. The run ID, worker, resources,
   registry counts, and all scientific settings remain unchanged.
6. Do not change the source case, checkpoint, policy noise, target, mask,
   budget, solver iterations, learning rate, tolerance, action horizon,
   simulator protocol, or any future retry-C resource.

## Consequences

The CPU allocation may establish that the persisted artifact representation is
readable without weakening live-trace semantics. It cannot repair retry B,
turn finite nonconvergence into infeasibility, establish transport, authorize
retry C by itself, launch IFT-01, or authorize probe/MLP training.
ADR-0034 separately governs exact trace-derived status recomputation; this
decision changes only scalar representation decoding.
