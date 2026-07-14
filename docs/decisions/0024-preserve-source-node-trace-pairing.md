# 0024 — Preserve source-node trace pairing before interpreting R03A

Status: accepted on 2026-07-15 after H100 task `27636_0` failed the historical
trace-exact gate, but before recording any per-leaf mismatch magnitude and
before requesting either strong-analytic arm.

## Context

Task `27636_0` reproduced the source case, observation, branch, geometry,
paired noise, and float64-reconstructed frozen action elements, but the fresh
midpoint sampler trace did not satisfy exact array equality to the immutable
R02 trace.  The R02
case ran on `worker-1`; the R03A retry ran on `worker-0`.  Eager BF16 execution
is already known to be hardware/kernel conditioned.  Final-action equality is
not enough to establish identical denoising paths, and the analytic field has
discontinuous OBB selection and a hard 5 mm margin stop.

The immutable R02 provenance maps 16/17 R03A cases to `worker-1` and one case,
`crfs-4db90bad9285c42f`, to `worker-2`.  No strong-analytic action, simulator
rollout, or latency outcome has yet been observed.

## Decision

1. Keep historical midpoint trace equality exact.  Do not derive or select a
   tolerance from the new diagnostic.
2. Pin the next case-0 H100 smoke to its immutable R02 source host,
   `worker-1`.  The submitter derives this host only from the hash-validated
   R02 artifact and refuses submission unless that exact node is healthy and
   has the full 128 GiB task memory available.
3. If the same-node trace is exact and the smoke passes, split the full R03A
   population by immutable source host: 16 registered indices on `worker-1`
   and index 5 on `worker-2`.  Preserve the same config, seeds, arms, case
   identities, and 9/17 kill threshold.
   Until that grouped launcher is registered, reject the ungrouped full-array
   entry point.  Retire the legacy cross-node MIG smoke as well; only the
   source-node H100 smoke is admissible.
4. If the same-node trace is still non-exact, do not weaken the historical
   pairing gate.  Freeze and rerun the existing privileged distributed-
   residual control contemporaneously in every R03A allocation, paired to the
   current frozen state, observation, noise, trace, and horizon.
5. Under that fallback, analytic SPS at or above the historical 9/17 threshold
   still rejects pure learned-clearance necessity.  Analytic below 9/17 is
   inconclusive if the contemporaneous privileged control fails to reproduce
   healthy steerability; it supports retaining the strong analytic baseline
   only when the contemporaneous privileged control is healthy.
6. ADR-0010 numerical limits may be reported as portability diagnostics only.
   They cannot relabel a non-exact historical trace as an exact comparator.
   Cross-node latency remains descriptive and confounded.

## Consequences

This decision preserves baseline-first causal pairing without selecting a
post-hoc tolerance.  It may delay the run until the exact source node has live
capacity.  That delay is preferable to interpreting an analytic intervention
whose internal baseline path is not the registered comparator.
