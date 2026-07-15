# Controlled experiment ledger

Last updated: 2026-07-15 (Asia/Ho_Chi_Minh)

This is the short operational record. Detailed historical evidence before the
inverse-flow pivot remains in
`docs/archive/progress/2026-07-15-pre-inverse-flow-pivot.md`.

## Fixed research hypothesis

A simulator-verified safe-progress action correction is not generally preserved
when it is repeated as a constant flow residual. A time-dependent residual
velocity sequence found through the exact frozen pi0.5 sampler can transport
the same target under the R03 correction budget, and those controls may later
serve as labels for a deployable residual-field student.

The hypothesis has two separate parts. R05A tests teacher transport. Student
learnability is not tested, and even a passing R05A authorizes only IFT-03 on
untouched groups. No MLP may be trained unless that separate authorization
passes; a failed R05A stops this student direction.

## Existing baseline evidence

| ID | Question | Evidence | Result | Role |
|---|---|---|---|---|
| B00 | Does a safe-progress five-action witness physically exist? | R01, jobs `27306`/`27364` | 17/17 eligible groups | Direct upper bound |
| B01 | Does the existing constant privileged flow residual transport it? | R03, jobs `27405`/`27450` | 9/17 | Baseline to beat |
| B02 | Does the stronger analytic smoke solve safety and progress together? | R03A task `27639_0` | 0/1; safe but insufficient progress | Historical diagnostic only |

## R05A experiment sequence

### IFT-00 — Local inverse-control contract

Status: **passed locally on 2026-07-15**

Purpose: verify the mathematics and implementation without pi0.5, a simulator,
or a research claim.

Required checks:

1. A small deterministic nonlinear vector field exposes all ten Euler steps.
2. The solver receives a terminal target and returns a ten-row schedule with
   controls active only at steps 5--9.
3. Controls are exactly zero at steps 0--4 and outside the first-five XYZ mask.
4. `P <= B` and every step displacement is at most `B/5`.
5. The audited recurrence exactly reconstructs the reported terminal state.
6. Reversing control time order is represented as a distinct arm.
7. Structural target-pairing failure and finite-search nonconvergence are
   explicit, distinct failures; finite search alone is never called an
   infeasibility proof.
8. The no-control sampler path remains byte-for-byte unchanged.
9. The supplied target is exactly the recomputed frozen terminal action outside
   first-five XYZ, including padded coordinates and signed zero.

The solver is a constrained feasibility search, not an optimizer certificate.
Energy is recorded and used only to choose among fidelity-feasible iterates;
neither a local nor global minimum will be claimed.

Pass: every focused test and the complete `./init.sh` gate pass. This is
implementation evidence only.

Evidence: `evidence/r05a/ift00-synthetic.json`. Seventeen dependency-backed
PyTorch tests passed in the isolated local environment; the complete harness
passed 413 tests with 152 declared dependency skips. No pi0.5 or simulator was
executed.

### IFT-00A — One-case allocation integration canary

Status: **active; attempts A and B retired as incomplete apparatus runs**

Purpose: connect the passed solver to the real frozen pi0.5 sampler on only
`crfs-1069f29a8d76463a`, without executing a teacher-generated action in the
simulator and without receiving efficacy credit.

The canary must run on the immutable source host `worker-1` with one H100,
eight CPUs, 64 GiB host RAM, `%1`, and no requeue. It must measure actual host
and GPU peaks rather than assume a 128 GiB requirement. A separately registered
CPU `afterany` validator must inspect the final artifact.

ADR-0029 permits this exact source-pinned request to wait in Slurm when no H100
is immediately free. The launcher must record the observed capacity truthfully;
queue delay is operational state and cannot count as research evidence.

Attempt `r05a-inverse-flow-canary-20260715a` used exact GPU task `27714_0` and
CPU validator `27715`. It stopped before pi0.5 startup because a synthetic
sampler test required `1e-5` target equality even though all four unchanged
frozen fidelity gates passed. CPU diagnostic `27722` established the exact
mismatch, and ADR-0030 freezes the non-tuning repair. A masked wrapper count
mismatch (10 expected versus 12 reviewed tests) is repaired and structurally
bound before immutable retry B. Attempt A contains no transport or efficacy
outcome; see `evidence/r05a/ift00a-attempt-a.json`.

Retry B `r05a-inverse-flow-canary-20260715b` used GPU task `27726_0` on the
required worker-1 source host and CPU validator `27727`. It passed all focused
allocation suites with exact zero-skip counts 17/8/10/12, started real pi0.5,
and executed two deterministic 128-update teacher searches. Both searches were
finite but nonconverged and failed closed to the frozen action; the XYZ target
maximum/RMS errors were `0.8631912`/`0.3312218`. No teacher action entered the
simulator. The GPU job then failed before `results.json` because live cgroup
host-memory peak discovery was unavailable; the CPU validator correctly
rejected the missing result. The unfinalized payload also exposed a stale
finalizer test count and an ADR-0011 path-seam semantic mismatch. See ADR-0031
and `evidence/r05a/ift00a-attempt-b.json`. This diagnostic neither confirms nor
refutes target transport formally, but it is strong negative evidence for the
currently frozen solver on the preservation case.

IFT-00A pass requires exact fresh source pairing; target equality outside
first-five XYZ; exact compiled-before/after and eager-before/after zero-control
stability; the unchanged ADR-0011 numerical compiled/eager seam limits;
duplicate teacher schedules; an independent explicit-schedule replay equal to
the teacher result; exact recurrence, mask, time, budget and fidelity
validation; all frozen-model parameter gradients remaining `None`; and
complete host/GPU memory telemetry. The runtime budget is the immutable source
R02 float64 norm cast once to float32; the source direction norm, its float32
norm, and the realized rounded target-difference norm are reported separately.
No sampled policy action may be passed to `env.step`, and no simulator efficacy
outcome may be produced.

#### ADR-0031 CPU apparatus regression

Status: **terminal apparatus failure; immutable run ID consumed**

Exact run ID: `r05a-adr0031-apparatus-cpu-20260715a`.

This is a worker-1, CPU-only, two-CPU, 8 GiB, `0-0%1` Slurm check of the
ADR-0031 apparatus repairs plus the ADR-0033/0034 artifact trust-path repairs.
It runs the authoritative 17/8/10/12 registry through the exact shared suite
helper with zero skips, independently reparses the log, and resolves the live
allocation cgroup peak through the actual membership and mount paths. It also
checks artifact-only `(1,)` scalar decoding while keeping live scalars strict,
and derives teacher, zero-replay, and canonical-replay status from exact raw
trace metadata instead of trusting stored pass labels. It loads no checkpoint,
starts no policy server, runs no real pi0.5 teacher search, produces no
checkpoint teacher observation, executes no simulator action, and cannot
support a scientific claim. The unit suites do exercise synthetic inverse
solvers as implementation tests; those are not real checkpoint teacher
searches. See ADR-0032 through ADR-0034. A pass is only one prerequisite for a
separately reviewed retry-C decision.
The submission receipt also binds a separately callable strict validator. It
recomputes receipt/source hashes, resource identity, the 17/8/10/12 log, and
the live cgroup sidecar against a hidden candidate before publication.

Exact task `27797_0` ran on worker-1 from commit
`00dba0ad27169abd1344a97ae2f02b323e6a52ae` with the registered two CPUs,
8 GiB, 20-minute, no-GPU contract. All four suites passed 17/8/10/12 with zero
skips. The job then exited `3:0` at `live_cgroup_memory_peak`: the resolver
mapped the exact cgroup-v2 membership through mount root `/` to the task leaf
ending in `job_27797/step_batch/user/task_0`, but that leaf's `memory.peak`
was unreadable. `sacct` recorded neither MaxRSS nor MaxVMSize, so host peak use
remains unknown. Worker-1 reports Linux `5.15.0-130-generic`; the upstream 5.15
cgroup-v2 interface documents `memory.current` and `memory.max` but no
`memory.peak`, while the current interface documents `memory.peak`. Slurm's
`JobAcctGatherType` is also null. Therefore no exact peak fallback exists on
this allocation. No result candidate or `results.json` was written. See
`evidence/r05a/adr0031-apparatus-cpu-a.json`. This provides no inverse-flow
scientific outcome, and the run ID must not be reused.

#### CG-00 — Job-owned cgroup telemetry capability

Status: **preregistered and locally verified; not submitted**

Exact run ID: `r05a-cgroup-v2-current-capability-20260715a`.

Question: can worker-1 expose an allocation-owned memory interface that lets a
future H100 canary report host-memory evidence honestly, without pretending a
sampled value is an exact peak?

Procedure: one shell-only `0-0%1` task on worker-1, one CPU, 256 MiB, two
minutes, no GPU and no requeue. It maps only the executing task-to-exact-job
cgroup chain, samples job-scope `memory.current` 20 times at 100 ms, records a
positive finite `memory.max`, and checks hierarchical `memory.events` before
and after. It checks `memory.peak` once at the exact job scope and never searches
a shared parent. See ADR-0035.

Capability pass: either a positive native `memory.peak` is available, or all 20
sampled-current observations are valid with a positive high-water, the largest
gap is at most 500 ms, `memory.max` is positive and finite, and
`max`/`oom`/`oom_kill` event deltas are zero. The selected mount must not use
`memory_localevents`, which would make those counters local-only. The sampled
high-water is always labeled a lower bound, never an exact peak. Any unsafe
mapping fails closed; a missing or inadequate interface is a completed
unsupported result.

Scientific role: apparatus evidence only. It cannot validate inverse-flow
transport, repair retry B, authorize retry C by itself, launch IFT-01, or train
a probe/MLP. A supported result permits only a separately reviewed full-run
telemetry change before one frozen H100 canary.

Local acceptance: 11/11 capability tests, 6/6 R05A tracker-contract tests,
and the complete 447-test harness passed with 152 declared dependency skips.
Independent HPC, semantic, and adversarial-test reviews found no remaining
P0/P1 issue for CG-00 only.

### IFT-01 — Three-case real transport smoke

Status: **blocked on IFT-00A and a reviewed immutable efficacy config**

Fixed cases:

Immutable manifest SHA-256:
`bdb8ccbba01ebf500e0f1bd0fe4a4043054f922a273f9e90eb3860cfe753a633`.

| Case | Pre-existing stratum | What must happen |
|---|---|---|
| `crfs-1069f29a8d76463a` | Constant residual succeeded | Teacher preserves Safe-Progress Success |
| `crfs-7eddaafffb4f9474` | Clearance-only constant miss | Teacher reaches `D_sim >= 5 mm` and keeps progress |
| `crfs-bd7b0adf95145623` | Progress-only constant miss | Teacher preserves clearance and restores progress |

Fixed paired arms:

1. Frozen pi0.5.
2. Fresh direct paired safe-progress target.
3. Existing constant residual with path budget `B`.
4. Inverse-flow teacher active at steps 5--9 with `P <= B` and per-step
   displacement `<= B/5`.
5. The same teacher controls in reverse flow-step order.

All arms use the same branch, observation, instruction, explicit noise, source
host, and five executed actions. The teacher optimizer receives no simulator
or geometry feedback. Controls are frozen before any efficacy rollout.

Primary outcome: joint Safe-Progress Success, not clearance alone.

GO: all contracts validate, the fresh direct target reconfirms, and the teacher
passes 3/3. Any missing or invalid case is a failure. No MLP training follows
this smoke.

### IFT-02 — Seventeen-case development population

Status: **blocked on IFT-01**

GO requires all of the following:

- exactly 17 valid fixed-denominator artifacts;
- the constant arm reproduces the accepted 9/17 result;
- teacher success is at least 14/17;
- all nine constant successes are preserved;
- at least five of the eight constant misses are rescued;
- every counted result obeys the control budget and per-step cap.

The reverse-order diagnostic distinguishes time-specific transport from a
control multiset that works in any order. These 17 groups remain development
only regardless of the result.

### IFT-03 — New-state student authorization

Status: **blocked on IFT-02 and untouched official groups**

This future gate must repeat baseline, direct feasibility, and teacher transfer
on genuinely new source groups before any label collection or MLP training.
It must freeze episode/state-group splits and include naturally safe examples
with zero-control labels. No current case may enter this dataset.

## Current stop conditions

- Do not launch the retired R03A population.
- Do not train a scalar ECG probe or residual-field MLP.
- Do not accept a final-step overwrite, clipping, or a larger correction budget
  as inverse-flow success.
- Do not tune solver settings from simulator outcomes.
- Do not interpret local or synthetic tests as research evidence.
- Do not launch IFT-01 or an unchanged IFT-00A retry after retry B.
