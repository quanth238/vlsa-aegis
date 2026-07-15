# 0039 — Preserve sampled-current launch B and repair exact-task accounting

Status: accepted on 2026-07-15 from terminal jobs `27962_0` and `27963`.

## Evidence

The second ADR-0037 release used immutable run ID
`r05a-inverse-flow-sampled-current-canary-20260715b`, accepted implementation
`7d15c2c7921d9d1201638cb68ee48cc06a94ded1`, and release commit
`06b365b5899c2cb31db12187350cce48a3a0ea20`. The source contract SHA-256 is
`b9a9e253c9a5e27815018c13c839805c97752545189085caac016db4ee81e1e6`;
all 28 repository bindings matched the clean release commit.

GPU task `27962_0` completed on worker-1 with exit `0:0` in `00:03:09` using
the registered one H100, eight CPUs, and 64 GiB. Its raw payload SHA-256 is
`4c85603c62446d74259820dc7f86451ffe53723b79e0c866d2e38d7a026622cd`.
Both frozen teacher searches ran all 128 updates, remained finite, and produced
the same registered schedule and returned-action hashes. Both missed the
frozen fidelity limits, so the raw payload diagnostically reports deterministic
finite nonconvergence. Failed controls were not applied; returned actions stayed
frozen. Policy-generated steps, teacher-generated steps, and efficacy rollouts
were all zero.

CPU `afterany` job `27963` then failed with exit `3:0` before invoking the
Python publisher. The wrapper queried parent job `27962` with `JobIDRaw` and
required the returned identity to equal `27962_0`. On this cluster the exact
singleton query demonstrates `JobID=27962_0` but `JobIDRaw=27962`; therefore
that comparison cannot succeed. No `results.json` or hidden candidate exists.
The failure receipt truthfully records `passed=false`, `published=false`, and
no result digest, but its generic fallback omitted publisher-job and exact-stage
provenance.

Compact evidence is
`evidence/r05a/ift00a-sampled-current-launch-b.json`; detailed provenance is
archived in
`docs/archive/progress/2026-07-15-r05a-sampled-current-launch-b.md`.

## Decision

1. Classify launch B as **apparatus-inconclusive**. The raw GPU payload is
   diagnostic evidence that this one frozen search produced deterministic
   finite nonconvergence, but publication failed, so no accepted IFT-00A
   transport result exists.
2. Permanently consume the run ID and preserve its root, receipts, logs,
   payload, and telemetry. Never publish or retrofit the observed payload,
   rerun only its CPU publisher, resume either job, or reuse the identity.
3. Keep the H100 apparatus unreleased: `ready_to_run: false`, a nonempty
   `blocked_on` list, no `execution_release`, and no replacement H100 run ID.
4. Repair only exact source-task accounting. Query `${SOURCE_JOB_ID}_0`, read
   display `JobID`, require exactly one matching row, poll for a bounded period,
   accept only `COMPLETED` with `0:0`, and fail closed on missing, wrong,
   duplicate, terminal-failure, nonzero, or timed-out nonterminal records.
5. Add exact publisher job, source task, observed source state/exit, and failure
   stage to wrapper-generated negative receipts. This changes provenance only;
   it cannot promote a failed or unpublished result.
6. Before any H100 retry, validate the helper on VinUni with one trivial
   singleton **zero-GPU CPU array source** and one zero-GPU CPU `afterany`
   validator. The source must finish `COMPLETED|0:0`; the validator must observe
   that exact `${array_job_id}_0` identity through the production helper and
   publish an immutable shell-only receipt. Neither job may load a checkpoint,
   run Python, serve a model, render, execute simulator steps, search a teacher,
   or train. A failed regression blocks another canary.
7. Freeze all scientific content: worker-1, case, source state and observation,
   instruction, seeds/noise, checkpoint, target, normalization, sampler,
   solver, optimizer, 128-update limit, tolerances, masks, active steps, path
   budget, per-step cap, simulator boundary, and 64-GiB H100 request.
8. Only after focused and complete local gates, independent review, a clean
   pushed/synchronized repair commit, and the terminal zero-GPU regression may
   a separate reviewed direct-child release select a new unused immutable
   IFT-00A ID. This decision does not itself authorize that H100 submission.

## Scientific boundary

Finite nonconvergence is not an infeasibility or optimality certificate. Run B
did not test collision avoidance, task progress, safety, efficacy,
generalization, or student learnability. IFT-01, label collection, probe
training, and residual-field MLP training remain forbidden.

Allowed wording:

> The raw GPU payload diagnostically reproduced deterministic finite
> nonconvergence; publication failed, so no accepted IFT-00A transport
> conclusion exists.

## Exact zero-GPU regression identity

The one live accounting regression permitted by this decision is:

- immutable run ID:
  `r05a-exact-array-task-afterany-regression-20260715a`;
- source: worker-1-pinned singleton CPU array `0-0%1`, one CPU, 256 MiB,
  `00:02:00`, no requeue, and zero GPUs;
- validator: worker-1-pinned CPU `afterany` job, one CPU, 256 MiB,
  `00:02:00`, no requeue, and zero GPUs;
- implementation:
  `scripts/hpc/submit_r05a_exact_task_status_regression.sh` and its exact bound
  source, validator, Slurm, helper, and test files;
- execution boundary: shell and `jq` only; zero Python, checkpoint, model,
  simulator, rendering, metrics, teacher search, or training;
- automatic cancellation, resubmission, H100 release, and next-experiment
  authorization: false.

The run ID was unused and the user queue empty when registered. The submitter
must still repeat those checks and bind the clean pushed/synchronized source
commit before creating the run root. A pass proves only that the production
helper can observe one real completed exact singleton task from an `afterany`
allocation.
