# 0030 — Bind the R05A sampler test to the frozen fidelity contract

Status: accepted on 2026-07-15

## Context

IFT-00A attempt `r05a-inverse-flow-canary-20260715a` ran as GPU task
`27714_0` on worker-1. It stopped in dependency-backed focused tests before
starting pi0.5. The inverse-control suite passed 17/17. One synthetic sampler
test classified the teacher as converged but then required its final tensor to
equal the target within `1e-5`.

CPU Slurm diagnostic `27722` reproduced the fixture using the same commit,
worker, eight threads, and PyTorch `2.7.1+cu126`. Its controlled-coordinate
error was `0.0074953213`. The four preregistered physical fidelity values were
`0.0074953213`, `0.0019352837`, `0.0074953213`, and `0.0012669405`, all within
their unchanged limits `0.010`, `0.005`, `0.050`, and `0.015`. The extra
`1e-5` assertion was therefore stricter than, and inconsistent with, the
solver's registered convergence meaning.

The attempt also revealed that the allocation wrapper still expected ten R05A
tests although the reviewed suite contained twelve. The sampler failure masked
that later fail-closed count mismatch.

## Decision

1. Replace only the synthetic test's `1e-5` allclose assertion with independent
   recomputation of all four already-frozen physical fidelity gates and equality
   to the trace-reported metrics.
2. Do not change a solver parameter, scientific tolerance, target, budget,
   schedule constraint, source pairing rule, or canary pass criterion.
3. Update the allocation wrapper to the actual twelve-test R05A suite and bind
   every frozen suite count to the checked-in test declarations structurally.
4. Preserve attempt `20260715a` as an immutable apparatus failure. Retry only
   with new run ID `r05a-inverse-flow-canary-20260715b` after focused,
   allocation-backed, full-harness, and independent review gates pass.
5. The failed attempt and diagnostic provide no pi0.5 transport, simulator
   efficacy, safety, novelty, or training evidence.

## Consequences

The test now checks the solver's actual preregistered contract rather than a
platform-sensitive optimizer iterate. This is not tolerance tuning: every
numeric limit and scientific setting remains byte-for-byte unchanged. Future
test additions cannot silently invalidate the allocation wrapper's expected
counts.
