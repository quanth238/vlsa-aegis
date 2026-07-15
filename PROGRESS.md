# Progress

Last updated: 2026-07-15 (Asia/Ho_Chi_Minh)

Active branch: `agent/crfs-oracle-harness`

Baseline: THU-RCSCT/VLSA-Aegis commit
`57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b`

Current reviewed repository commit before this working change:
`12f8762d1a74dbfd39105a0aac93b45399a5ef46`

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
allocation integration canary, is now active with an immutable config and
submission path. See `EXPERIMENTS.md`, ADR-0028, and
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
  The working changes are not yet committed, pushed, or synchronized.
- No R05A job has been submitted and no inverse-flow research outcome exists.

## Exact next action

Pass the complete local gate, commit/push/synchronize the reviewed tree, repeat
live control-plane preflight, and invoke exactly once:

```bash
RUN_ID=r05a-inverse-flow-canary-20260715a scripts/hpc/submit_r05a_canary.sh manifests/r05a_inverse_flow_teacher_smoke.jsonl configs/experiments/r05a_inverse_flow_canary.json
```

This command registers the worker-1-pinned 64 GiB H100 canary on hold, registers
its CPU `afterany` validator, records both exact IDs, and only then releases the
canary. It must not be invoked until the local and remote trees are the same
clean reviewed commit and the exact run ID is unused.

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
