# ADR-0072: Preserve the failed paired AEGIS canary and repair its runtime

Date: 2026-07-17

Status: accepted implementation repair; retry execution not released

## Context

ADR-0071 released immutable run
`r06-aegis-paired-canary-20260717a` from release commit
`21a8fb11e45e2a259559fb7cfb690caa83ad88ed`. Exact worker-1 GPU task
`28447_0` terminated `FAILED|1:0` after 24 seconds, and exact zero-GPU
`afterany` validator `28448` terminated `FAILED|1:0`.

The GPU task stopped at `allocation_dependency_preflight` with
`ModuleNotFoundError: No module named 'cvxpy'` under
`/mnt/data/quanth/venvs/openpi-libero-client/bin/python`. It started no policy
server, ran no baseline simulator replay, and executed zero GroundingDINO,
MVEE, QP, or AEGIS steps. No `results.json` exists. Therefore this launch is
apparatus-inconclusive and provides no collision-avoidance result or AEGIS
method evidence.

Terminal identities, states, hashes, and the zero-execution boundary are
preserved in `evidence/r06/aegis-paired-launch-a.json`. The immutable run root
and both job identities are permanently consumed.

Read-only control-plane inspection found the installed AEGIS environment at
`/mnt/data/quanth/venvs/safety_vla/main/bin/python`. It shares the frozen
Python and simulator stack used by the failed client environment and adds the
required CVXPY, Open3D, GroundingDINO, and OSQP packages. This package
inventory is only preflight evidence; actual imports, CUDA kernel execution,
focused tests, model use, and simulator integration still require an
allocation.

Independent review also found that the paired workload did not export the
live robosuite image convention and version required by its own fail-closed
1024-pixel AEGIS renderer.

## Decision

Consume launch A without a scientific result and fail-close the experiment
configuration.

Accept only the following apparatus repair:

- run the LIBERO/AEGIS client, allocation tests, and paired runner with
  `/mnt/data/quanth/venvs/safety_vla/main/bin/python`;
- extract, verify, and export the live
  `R06_ROBOSUITE_IMAGE_CONVENTION=opengl` and
  `R06_ROBOSUITE_VERSION=1.4.1` declarations before runner execution;
- require a real allocation-backed CUDA tensor operation and synchronization,
  the complete AEGIS dependency imports, and the focused test suite with zero
  skips.

The repair does not change the case, boundary state, observation,
instruction, policy noise, nominal actions, five-action horizon, Codex label,
GroundingDINO assets, public AEGIS constants/QP, metrics, worker, or resource
request.

A new direct-child release with a fresh unused immutable run ID is required.
Run A may not be resumed, overwritten, or interpreted as an AEGIS failure.

## Scientific boundary

Whether AEGIS prevents the canary collision remains unanswered. The 20-case
population, full SafeLIBERO benchmark, and probe/MLP training remain blocked.
The future arm is still reported as **pi0.5 plus AEGIS with a Codex-frozen
label**, not original end-to-end AEGIS.
