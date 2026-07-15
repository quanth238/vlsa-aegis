# Progress

Last updated: 2026-07-15 (Asia/Ho_Chi_Minh)

Active branch: `agent/crfs-oracle-harness`

Baseline: THU-RCSCT/VLSA-Aegis commit
`57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b`

Inverse-flow apparatus implementation commit:
`71e7457d8a1052b108da696a34eb3a4ad54abe7e`

The complete pre-pivot H00--R03A chronology is preserved verbatim in
`docs/archive/progress/2026-07-15-pre-inverse-flow-pivot.md`.

## Current research question

Can a paired simulator-verified safe-progress action target be converted through
the exact frozen pi0.5 sampler into a budgeted, time-dependent residual velocity
sequence that succeeds where the existing constant residual fails?

This is a teacher-transport question. It does not yet ask whether an MLP can
learn or generalize the controls.

## Authoritative baseline evidence

- R01 direct safe-progress witnesses passed 17/17 eligible development groups.
  Evidence: Slurm array `27306`, independent verifier `27364`, and
  `evidence/r01/r01-summary.json` SHA-256
  `715d9326f02df531e0c6d7aa7b94629569b320800b4283c36fff41335d9b8bc5`.
- R03 constant distributed privileged residual passed 9/17; frozen, equal-norm
  random, registered static analytic, and bridge passed 0/17. Evidence: Slurm
  array `27405`, verifier `27450`, and `evidence/r03/r03-summary.json` SHA-256
  `dea3e66c854faa0b659777bfed4b651b19d8d4d0710696ff7bfe52c44d4ee76e`.
- The eight constant-residual misses split into four clearance-only and four
  progress-only failures. This isolates the open problem as action-to-flow
  transport under a joint safety-and-task criterion.
- R04A task `27514_0`/validator `27516` and R04B task
  `27558_0`/validator `27559` passed real trace and exact saved-latent resume
  apparatus contracts. They contain no learned result or efficacy outcome.
- R03A source-node smoke `27639_0` is valid one-case evidence: both analytic
  arms avoided contact but failed required progress. No 17-case R03A population
  was launched and no population conclusion exists.

## Pivot and archive

- ADR-0028 intentionally retires the pending R03A population and blocks the
  historical scalar ECG direction. R03A code, decisions, and evidence remain
  preserved for provenance and future matched baselines.
- The stale root H03 handoff moved to
  `docs/archive/handoffs/2026-07-14-h03-session-handoff.md`.
- Temporary prior-art downloads moved from `tmp/pdfs/` to the ignored local
  archive `.harness/archive/2026-07-15-prior-art-pdfs/`.
- The manuscript, proposal PDF, checked-in evidence, simulator code, baseline,
  R04B resume seam, and analytic comparator remain in their original locations.

## Active gate: R05A inverse-flow teacher transport

IFT-00 passed as synthetic implementation evidence. IFT-00A, the one-case
allocation integration canary, remains active after two immutable apparatus
attempts. Retry B reached real pi0.5 and ran the teacher twice, but no accepted
`results.json` was finalized. See `EXPERIMENTS.md`, ADR-0028, ADR-0031, and
`evidence/r05a/ift00-synthetic.json`.

IFT-00 established the deterministic constrained solver, exact iterative
sampler time grid, target pairing, recurrence auditing, strict mask/budget
enforcement, explicit nonconvergence/nonfinite statuses, reverse-time control,
outer-`no_grad` operation, and graph-free results. Seventeen isolated PyTorch
tests and the complete 413-test harness gate passed with 152 declared
dependency skips. This remains synthetic
implementation evidence only.

IFT-00A will connect that solver to one real frozen pi0.5 source case without
executing a teacher action in the simulator. Its 64 GiB host-memory request is
an initial bounded canary request, not an assertion of consumption; measured
host/GPU peaks determine any later request.

The real IFT-01 smoke is frozen to three pre-existing R03 strata, all on their
immutable `worker-1` source host:

1. `crfs-1069f29a8d76463a` — preserve a constant-residual success;
2. `crfs-7eddaafffb4f9474` — repair a clearance-only miss;
3. `crfs-bd7b0adf95145623` — repair a progress-only miss.

The teacher receives no simulator or geometry feedback. Its ten-row schedule is
zero for sampler steps 0--4 and controls only the first-five XYZ coordinates at
the same active steps 5--9 as the R03 baseline. Total latent path is no larger
than the R03 target correction and every active step is capped at one fifth of
that budget. Its controls are frozen before efficacy rollout.

IFT-01 is GO only at 3/3 teacher Safe-Progress Success after exact pairing,
fresh direct-target reconfirmation, target fidelity, duplicate inference,
budget validation, and duplicate simulator replay. No MLP training follows a
smoke pass.

IFT-02 may run all 17 development groups only after IFT-01 passes. Its GO rule
is at least 14/17, preservation of all nine constant successes, and rescue of
at least five of the eight constant misses. The historical constant arm must
first reproduce 9/17.

## Current verification

- Before the pivot edits, `./init.sh` passed 362 tests with 117 dependency-only
  skips on 2026-07-15.
- Read-only VinUni preflight at `2026-07-15T02:51:44Z` found an empty user
  queue, no login-node compute/service process, healthy mixed `worker-0` through
  `worker-2`, drained/nonresponding `worker-3`, healthy mixed
  `worker-mig-3g40gb-0`, drained/nonresponding `worker-mig-3g40gb-1`, and 24 TB
  free on `/mnt/data`. This is capacity telemetry only; no job was submitted.
- The solver and canary received independent adversarial review. The immutable
  IFT-00A decision, schema, and config hashes are respectively
  `d2a00b1b049e92bb1ec8f11d60fa447e5e8bd5109cb8cf3f3fbb08f89498656f`,
  `e1681f865f81f2986945d10fb14073fe4cd9e78cbc5026f6749ab321b9cfb7a7`,
  and `c31401867f3cdce2b3f443ad021c39dfb812f573b570e1e7434e1f149f79abfb`.
  The apparatus passed an independent launch audit with no P0/P1 findings,
  then the complete local gate passed 413 tests with 152 declared dependency
  skips. After the ADR-0029 pending-submission correction, the current complete
  gate passed 415 tests with the same 152 declared skips, and the R05A suite
  separately passed 12/12 with zero skips, including a zero-free-GPU fake-Slurm
  atomic transaction. Apparatus commit
  `71e7457d8a1052b108da696a34eb3a4ad54abe7e` is pushed.
- The final tracking tree was pushed and synchronized by fast-forward to a
  clean local/remote commit `02ca22b4cb40a6d02b3dfb18c998b0308dfa0e33`.
  Live preflight at `2026-07-15T04:52:46Z` found the exact run ID unused, the
  user queue empty, no login-node compute process, and worker-1 healthy with
  `194687 MiB` FreeMem. Its Slurm record reported eight configured and eight
  allocated H100s, so the launcher was intentionally not invoked. This is a
  capacity wait, not an apparatus or research outcome.
- ADR-0029 records the correction to that operational policy. An independent
  review found no scientific protection in requiring an H100 to be idle before
  submission, and a control-plane-only `sbatch --test-only` accepted the exact
  worker-1-pinned request without creating a job. The launcher may therefore
  submit once and let Slurm hold it `PENDING`; all scientific hashes, resources,
  source pinning, held-job receipts, and the CPU `afterany` validator remain
  unchanged.
- IFT-00A attempt `r05a-inverse-flow-canary-20260715a` submitted exact GPU task
  `27714_0` and CPU `afterany` validator `27715`. Task `27714_0` ran on worker-1
  for 31 seconds and failed at `dependency_backed_focused_tests`; the validator
  then failed closed because no `results.json` existed. The control suite passed
  17/17, but one of eight sampler tests rejected a converged toy output only at
  an extra `1e-5` assertion. No pi0.5 server, inference, or simulator efficacy
  ran. Failure and validator artifact hashes are recorded in
  `evidence/r05a/ift00a-attempt-a.json`, SHA-256
  `bc1707688a16c54a8facfedfae874d17c9c123b4c57f5f18260d493cbd2cc998`.
- CPU Slurm diagnostic `27722` reproduced the fixture on worker-1 at the exact
  source commit. Error `0.0074953213` passed all unchanged frozen fidelity gates
  (`0.010`, `0.005`, `0.050`, `0.015`). ADR-0030 therefore binds the test to
  independent recomputation of those gates without changing a tolerance. The
  attempt also exposed a masked allocation-wrapper count mismatch: 10 expected
  versus 12 reviewed R05A tests. Attempt A is retired as apparatus failure and
  provides no inverse-flow research outcome.
- IFT-00A retry B `r05a-inverse-flow-canary-20260715b` used exact GPU task
  `27726_0` on worker-1 and CPU `afterany` validator `27727`. All four focused
  allocation suites passed 17/8/10/12 with zero skips, pi0.5 served the real
  case, and two deterministic teacher searches completed 128 finite updates.
  Both searches returned status 3 and failed closed without applying controls;
  their target errors (`0.8631912` XYZ maximum, `0.3312218` XYZ RMS) were far
  outside the unchanged fidelity gates. This is a bounded-search diagnostic,
  not infeasibility and not simulator efficacy.
- Retry B still failed IFT-00A. GPU task `27726_0` exited `6:0` after the
  wrapper could not read a live Slurm cgroup host-memory peak, so memory
  finalization and exact GPU validation never wrote `results.json`; validator
  `27727` correctly failed closed. GPU telemetry peaked at 8501 MiB (about
  8.30 GiB). No OOM exit was observed, but host peak use remains unknown. The
  terminal audit also found a
  stale finalizer count of 10 versus 12 tests and a cross-path byte-equality
  gate inconsistent with ADR-0011. Compiled and eager paths were each exactly
  stable before/after, source eager action/trace pairing was exact, and their
  numerical seam passed all unchanged ADR-0011 physical limits. ADR-0031
  freezes apparatus-only repairs and forbids solver tuning. Compact evidence:
  `evidence/r05a/ift00a-attempt-b.json`, SHA-256
  `850e2b9d2e11d92342095c55ffeb0e94f9ec17fbd77de1188f3a65efd84cab0b`.
- After recording retry B, the terminal-interpretation tree passed `./init.sh`:
  415 tests, 152 declared dependency skips, and all 21 artifact/18 gate audits.
  This does not repair or pass IFT-00A.
- The three ADR-0031 repairs are implemented without changing the frozen
  solver or science: the live cgroup resolver maps actual membership through
  mountinfo and preserves path provenance; one strict JSON registry feeds the
  shared H100/CPU suite runner and Python finalizer at 17/8/10/12; and the
  compiled/eager seam is numerical under unchanged ADR-0011 limits while
  exact eager/source and within-path pairing remain mandatory. With the
  ADR-0033/0034 artifact trust-path regressions, the current tree passed
  `./init.sh` with 436 tests, the same 152 declared dependency skips, and all
  21 artifact/18 gate audits; the dependency-backed R05A canary suite also
  passed 12/12 with zero skips. This includes the independently callable
  CPU-result validator, its eight tamper cases, and the held-job
  failure-transaction fixture. This is local implementation evidence only.
- ADR-0032 preregistered the exact CPU-only integration run
  `r05a-adr0031-apparatus-cpu-20260715a`: worker-1, `0-0%1`, two CPUs, 8 GiB,
  20 minutes, no GPU, checkpoint, policy server, real pi0.5 teacher search or
  teacher observation, simulator step, efficacy claim, or training
  authorization. Its required unit suites contain synthetic inverse-solver
  calls only. A separately callable, receipt-hash-bound validator checks the
  unpublished result candidate and has dependency-free tamper coverage. This
  was its frozen pre-submission contract; the terminal task is recorded below.
- Independent replay of the exact retry-B payload, SHA-256
  `d5721d08747cd7c8f335057f2d89f7224d8921ac0bf1cba9bd16475fd3622d2b`,
  exposed persisted scalar leaves with shape `(1,)`. ADR-0033 accepts a
  separate artifact-only singleton decoder and keeps live sampler `_scalar`
  strict. Further coupled-tamper review exposed stored teacher/replay labels
  that were not all derived from their raw timing and schedule metadata.
  ADR-0034 now requires exact teacher truth patterns, independently recomputed
  zero/canonical replay metadata, the converged teacher action/final bindings,
  branch-aware canonical top-level/policy-call shape, and one exact
  nonconvergence sentinel. The exact adapted retry-B payload then recomputed as
  `completed_nonconverged` with zero semantic-validator errors; this is
  diagnostic re-interpretation, not a repaired result or efficacy evidence.
  Independent scientific and HPC reviews found no remaining P0/P1 blocker for
  the preregistered CPU-only apparatus run. They remain explicit NO-GO for
  retry C, IFT-01, simulator efficacy, and probe/MLP training.
- The exact ADR-0032 CPU apparatus run
  `r05a-adr0031-apparatus-cpu-20260715a` is terminal and consumed. Job
  `27797_0` ran on worker-1 from commit
  `00dba0ad27169abd1344a97ae2f02b323e6a52ae` with two CPUs, 8 GiB, no GPU,
  and exited `3:0` after 20 seconds. All registered suites passed 17/8/10/12
  with zero skips. The remaining failure is live cgroup measurement: the
  mapper reconstructed the exact v2 membership leaf ending in
  `job_27797/step_batch/user/task_0`, where `memory.peak` was unreadable.
  Worker-1 is Linux `5.15.0-130-generic`; the upstream 5.15 cgroup-v2
  interface has no `memory.peak`, and Slurm `JobAcctGatherType` is null with no
  MaxRSS/MaxVMSize. Thus measured host peak remains unknown and no exact
  fallback exists on this allocation. No result candidate, checkpoint load,
  policy server, real teacher observation, simulator step, efficacy claim, or
  training occurred. Synthetic unit-test solver/model calls did run. Evidence:
  `evidence/r05a/adr0031-apparatus-cpu-a.json`. This blocks retry C and requires
  a bounded leaf-versus-allocation-ancestor diagnostic; never reuse the run ID.
- ADR-0035 preregisters the exact shell-only capability run
  `r05a-cgroup-v2-current-capability-20260715a`: worker-1, `0-0%1`, one CPU,
  256 MiB, two minutes, no GPU, no requeue, and no Python, model, simulator, or
  training. It reads only the task-to-exact-job cgroup chain. A positive native
  peak or a positive finite job-scope `memory.current`/`memory.max` trace with
  hierarchical zero limit/OOM event deltas is apparatus support only. The
  selected mount must not use `memory_localevents`. Sampled high-water remains
  a lower bound, never an exact peak. The gate has not been submitted and
  cannot authorize retry C by itself.
- The reviewed CG-00 tree passes 11/11 focused capability tests, 6/6 R05A
  tracker-contract tests, and the complete `./init.sh` gate: 447 tests with 152
  declared dependency skips plus all 21 artifact/18 gate audits. Independent
  HPC, semantic, and adversarial-test reviews report GO for CG-00 only and
  retain NO-GO for retry C, IFT-01, efficacy, and training.

## Exact next action

Do not resubmit retry B or reuse the consumed CPU run ID; do not launch IFT-01.
Finish independent review and the complete local gate for ADR-0035, commit and
push the exact tree, synchronize VinUni to that clean commit, verify the exact
run ID is unused, the user queue is empty, and worker-1 is healthy with at least
256 MiB free. Then invoke exactly once:

```bash
RUN_ID=r05a-cgroup-v2-current-capability-20260715a \
  scripts/hpc/submit_r05a_cgroup_v2_current_capability.sh
```

Interpret only that exact shell-only task, immutable receipt, bounded TSV, and
atomic result. Do not run Python or experiments on the login node. A supported
capability permits a separate reviewed telemetry implementation; it does not
authorize retry C, IFT-01, efficacy claims, or training. An unsupported outcome
stops this telemetry route. Never relabel sampled high-water as an exact peak.

## Non-negotiable stops

- Do not launch `r03a-analytic-kill-population-20260715a`.
- Do not train a probe or residual-field MLP unless a separate untouched-group
  IFT-03 authorization passes. A failed R05A stops this student direction.
- Do not use the 17 development groups for student training or final testing.
- Do not tune solver settings from simulator outcomes.
- Do not count clipping, final-action replacement, a final-step overwrite, a
  larger path budget, missing cases, or clearance without progress as success.
- Keep `D_opt` separate from `D_sim`; only simulator substep clearance/contact
  and the frozen progress/motion gates determine Safe-Progress Success.
