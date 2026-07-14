# 0015 — Freeze reach-distance reconstruction arithmetic

Status: accepted on 2026-07-14 after the immutable R02 population and verifier,
before any R04 implementation or outcome.

## Context

The allocation's Python 3.8 runner validated every R02 case before writing it.
After the H100 population completed, an independent Python 3.12 validator
reconstructed one analytic reach distance one ULP differently. Python 3.12
changed floating-point `sum` behavior; tuple/list JSON representation was not
causal. The drift was `5.551115123125783e-17` in both the end distance and
derived progress. No action, simulator result, scientific predicate, or
allocation-side validation changed.

Using a tolerance on the whole reach annotation would unnecessarily weaken the
exact binding of simulator snapshots, identities, body motion, and gate-driving
values.

## Decision

Compute the three-coordinate squared distance with an explicit left-to-right
float accumulator, then apply `sqrt`. This reproduces the allocation-era
arithmetic across supported Python runtimes. Keep the complete reconstructed
reach annotation exact and keep all progress and displacement comparisons
unchanged.

Freeze the allocation values from the affected population case in a local
regression:

```text
end_distance_to_branch_target_m = 0x1.6d2ee216ebf6ep-2
reach_progress_m                = 0x1.f2ca29b3edf80p-8
```

Do not rewrite the immutable population artifacts or allocation summary.

## Consequences

- The original H100 case validates exactly on Python 3.12 as well as on the
  allocation runtime.
- Stored snapshots, annotation hashes, and scientific gates remain fail-closed;
  no general reach-record tolerance is introduced.
- The fix is reconstruction portability only and cannot alter the completed
  R03 result.
