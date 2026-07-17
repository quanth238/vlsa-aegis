# ADR-0073: Release the repaired paired AEGIS canary

Date: 2026-07-17

Status: accepted execution release

## Context

ADR-0072 consumed the apparatus-inconclusive first launch and accepted a
strict runtime-only repair. The accepted implementation commit is
`257d08d487938fa75dddca5abc9bd35f10626af5`.

The repaired workload uses the installed AEGIS environment for the
LIBERO/client, GroundingDINO, MVEE, CVXPY/OSQP, and paired-runner process. It
extracts and exports the live robosuite 1.4.1 OpenGL image convention, imports
the full client/simulator/perception/solver stack, performs a real CUDA tensor
operation and synchronization, and runs the focused allocation test suites
with zero skips before policy-server startup.

The complete local gate passes 854 tests with 276 declared local dependency
skips, 21 artifact audits, and 19 gate audits. Three independent reviews report
GO with no remaining P0/P1 blocker.

## Decision

Release exactly one fresh paired simulator canary for immutable run
`r06-aegis-paired-canary-20260717b`, manifest row 0, case
`crfs-1069f29a8d76463a`.

This release must be the direct child of accepted implementation commit
`257d08d487938fa75dddca5abc9bd35f10626af5` and may change only:

- `configs/experiments/r06_aegis_collision_conditioned.json`;
- `docs/decisions/0073-release-aegis-paired-canary.md`.

The GPU task remains array `0-0%1`, pinned to `worker-1`, with one H100, eight
CPUs, 64 GiB host memory, a 30-minute limit, and no requeue. The submitter must
atomically register the held singleton GPU task and a two-CPU, 8 GiB, zero-GPU
`afterany` validator before releasing the GPU task exactly once.

## Authorized comparison

Run frozen pi0.5 and pi0.5 plus AEGIS from the exact same boundary-20
simulator state, observation, instruction, explicit policy noise, nominal
actions, and five-action horizon. Preserve the frozen Codex phrase
`red milk carton`, the original GroundingDINO request and weights, public point
filtering/MVEE, and the full nine-variable public QP. Retain perception,
geometry, QP, and safety failures without fallback.

Report physical contact, minimum simulator clearance, buffer safety, task
progress, prefix completion, action modification, stopping, the public
obstacle-displacement proxy, and joint safety-plus-progress.

## Scientific boundary

This is one collision-conditioned case, not the full SafeLIBERO benchmark.
Because GLM-4.5V remains unavailable, the arm is **pi0.5 plus AEGIS with a
Codex-frozen label**, not original end-to-end AEGIS.

No outcome automatically authorizes the 20-case population, the 1,600-state
benchmark, or probe/MLP training. Run A remains permanently consumed and
cannot be overwritten or combined with this retry.
