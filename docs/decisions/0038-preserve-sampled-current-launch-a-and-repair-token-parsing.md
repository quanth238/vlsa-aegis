# 0038 — Preserve sampled-current launch A and repair token parsing

Status: accepted on 2026-07-15 after exact task `27928_0` was cancelled with
zero runtime under explicit user authorization.

## Evidence

The one authorized ADR-0037 release used immutable run ID
`r05a-inverse-flow-sampled-current-canary-20260715a` at release commit
`223667c91b05be9ab403e4d92d0cd1a96b45246f`. Live preflight observed
153,695 MiB free on worker-1 and created the immutable launch reservation. Slurm
then returned singleton array job `27928`, task `27928_0`, in the expected
`PENDING`, `JobHeldUser`, worker-1-pinned state with the exact registered
one-H100, eight-CPU, 64-GiB, two-hour resource request.

The submitter rejected that valid held record before writing the held-job
receipt. Its compound shell pattern placed a trailing separator in the
`JobState=PENDING` fragment and a leading separator in the
`Reason=JobHeldUser` fragment. Because those fields were adjacent in the real
Slurm record, the pattern required two spaces where Slurm emitted one.

The failure happened before the source contract, CPU `afterany` submission,
atomic submission receipt, or GPU release. Exact task `27928_0` remained
user-held, received no node, and accumulated `00:00:00` runtime. After exact
inspection and explicit authorization, only `27928_0` was cancelled. `sacct`
records `CANCELLED by 1073`, start time `None`, node `None assigned`, and zero
elapsed time. The immutable run root contains only `launch-reservation.json`,
SHA-256
`bec4583c5c046c7ca9f1155debc1f389d058daede42d3c485a39398f19c36ed4`.

Compact evidence is
`evidence/r05a/ift00a-sampled-current-launch-a.json`.

## Decision

1. Classify launch A as **apparatus-inconclusive with no execution**. It says
   nothing positive or negative about inverse-flow transport, collision
   avoidance, task progress, learnability, or generalization.
2. Permanently consume the run ID and preserve its run root and Slurm record.
   Never resume job `27928`, reuse the run ID, delete the reservation, or
   synthesize the missing held/source/submission receipts.
3. Return the sampled-current apparatus config to `ready_to_run: false`, remove
   `execution_release`, and select no replacement run ID in the implementation
   repair commit.
4. Repair only Slurm record parsing. Validate `JobState=PENDING`,
   `Reason=JobHeldUser`, and `ReqNodeList=worker-1` as independent exact tokens.
   Apply the same independent-token rule to the CPU publisher's pending state
   and partition checks.
5. The transaction regression must use the real adjacent-field forms
   `JobState=PENDING Reason=JobHeldUser` and
   `JobState=PENDING Partition=main`. Wrong GPU state, reason, or node must fail
   after exactly one held submission and before every receipt, CPU submission,
   or release.
6. Freeze all scientific content: case, source host, state, observation,
   instruction, seeds/noise, checkpoint, target, normalization, sampler,
   solver, optimizer, 128-update limit, tolerances, masks, active steps, path
   budget, per-step cap, and simulator boundary.
7. After focused and complete gates plus independent review, preserve the
   repair as a clean unreleased implementation commit. A separate direct-child
   release-only commit may select one new immutable run ID under ADR-0037. It
   must retain worker-1 and the exact registered resources and may be submitted
   once. No automatic retry or next experiment is authorized.

## Scientific boundary

This decision repairs a control-plane parser only. No checkpoint was loaded,
no policy server or pi0.5 sampler ran, no inverse-flow teacher search ran, no
policy- or teacher-generated action was executed, and no simulator efficacy
was measured. IFT-01, label collection, probe training, and MLP training remain
forbidden.

## Verification

The adjacent-field and fail-closed sampled-current suite passes 38 tests. The
complete repository gate passes 505 tests with 152 declared dependency skips,
plus all 21 artifact and 18 gate audits. Independent science, HPC, and
publication-security review found no P0/P1 issue. These checks validate the
apparatus repair and no-claim boundary only; they are not evidence that the
teacher transport mechanism works.
