# ADR-0071: Release the exact paired AEGIS canary

Date: 2026-07-17

Status: accepted execution release

## Context

ADR-0069 froze the outcome-blind canary label `red milk carton` after a valid
allocation-backed capture. ADR-0070 accepted the exact paired apparatus.
Independent re-review found no remaining P0/P1 blocker after the result
provenance and method-failure classification repairs. The complete local gate
passes 854 tests with 276 declared dependency skips, 21 artifact audits, and
19 gate audits.

The accepted implementation commit is
`1fa99f8639117dfcac5dd19cc24821829d8485b9`. It preserves the strict ancestry
from capture release `ccb8c5225517f21ca1405f7b1470fe47dab160ee` through
label-freeze commit `d9cf569d619e014c9e6423cd9ddb40f592435a71`.

## Decision

Release exactly one paired simulator canary for immutable run
`r06-aegis-paired-canary-20260717a`, manifest row 0, case
`crfs-1069f29a8d76463a`.

This release is the direct child of the accepted implementation commit and
changes only:

- `configs/experiments/r06_aegis_collision_conditioned.json`;
- `docs/decisions/0071-release-aegis-paired-canary.md`.

The GPU task is array `0-0%1`, pinned to `worker-1`, with one H100, eight CPUs,
64 GiB host memory, a 30-minute limit, and no requeue. The submitter must
register the held singleton GPU task and a two-CPU, 8 GiB, zero-GPU
`afterany` validator before releasing the GPU task exactly once. Pending for
resources is allowed.

## Authorized comparison

Run the frozen pi0.5 baseline and pi0.5 plus AEGIS from the same boundary-20
simulator state, observation, instruction, explicit policy noise, nominal
actions, and five-action horizon. The AEGIS arm uses the frozen Codex label,
the original GroundingDINO request and weights, public point filtering and
MVEE, and the literal full nine-variable public QP. It must retain perception,
geometry, and QP failures without fallback.

The result must report physical contact, minimum simulator clearance, buffer
safety, task progress, prefix task completion, action modification, stopping,
the public obstacle-displacement proxy, and joint safety-plus-progress. Every
terminal artifact is bound to the exact run, release commit, Slurm array task,
worker, and allocation-visible GPU.

## Scientific boundary

This is a one-case collision-conditioned diagnostic. It is not the 1,600
episode SafeLIBERO benchmark and cannot reproduce Table 1. Because GLM-4.5V is
unavailable, the evaluated arm is **pi0.5 plus AEGIS with a Codex-frozen
label**, not original end-to-end VLSA/AEGIS.

No outcome automatically authorizes the 20-case population, the full
SafeLIBERO benchmark, or probe/MLP training. Those require separate reviewed
protocols and releases.
