# 0044 — Release the R05A runtime-identity regression

Status: accepted for one exact zero-GPU execution on 2026-07-16.

## Evidence reviewed

The fail-closed apparatus with accepted implementation parent
`a669ef2dd15cc59996d332ead54ffcdf06552635` passed 10/10 focused tests and
the complete local gate: 603/603 tests passed, with 198 declared dependency
skips, 21 artifacts, 18 gates, WIP at most one, and the baseline pinned.  Two
independent reviewers reported no remaining P0/P1 blocker.

Live shell-only control-plane inspection found the selected run ID unused, the
user queue empty, and `worker-1` at 157948 MiB free memory in
`MIXED+DYNAMIC_NORM`, sufficient for this 256 MiB CPU check.

## Exact execution release

- Accepted implementation commit: `a669ef2dd15cc59996d332ead54ffcdf06552635`.
- Immutable run ID: `r05a-runtime-identity-regression-20260716a`.
- Source node: `worker-1`.
- Partition/account/QOS: `main`/`normal`/`normal`.
- Resources: one CPU, exactly 256 MiB host RAM, zero GPUs, `00:02:00`, no requeue.
- Shape: one non-array job, submitted held, receipted, and released once.
- Automatic cancellation: `false`.
- Automatic resubmission: `false`.
- H100 submission authorized: `false`.
- Automatic next gate: `false`.

This release authorizes only shell inspection of the two frozen interpreter
link chains.  It does not authorize Python execution, model or checkpoint
loading, inference, CUDA, a simulator, rendering, metrics, training, a
scientific conclusion, a constrained-flow H100 retry, or any later gate.
