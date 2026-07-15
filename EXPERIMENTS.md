# Controlled experiment ledger

Last updated: 2026-07-16 (Asia/Ho_Chi_Minh)

This is the sole active experiment protocol. Completed operational history is
archived in:

- `docs/archive/progress/2026-07-15-pre-inverse-flow-pivot.md`;
- `docs/archive/progress/2026-07-15-r05a-pre-execution-apparatus.md`;
- `docs/archive/progress/2026-07-15-r05a-sampled-current-launch-a.md`;
- `docs/archive/progress/2026-07-15-r05a-sampled-current-launch-b.md`.

ADRs and evidence JSON remain authoritative for immutable decisions and
terminal runs.

## Fixed research hypothesis

A simulator-verified safe-progress action correction is not generally
preserved when repeated as a constant flow residual. A time-dependent residual
velocity sequence found through the exact frozen pi0.5 sampler may transport
the same target under the same correction budget.

The hypothesis has two separate parts:

1. R05A asks whether a valid teacher control sequence exists and works.
2. A later student study would ask whether an MLP can predict such controls on
   untouched state groups.

No current experiment tests the second part. A failed R05A stops this student
direction; a passing R05A still does not automatically authorize training.

## Existing baseline evidence

| ID | Question | Evidence | Result | Role |
|---|---|---|---|---|
| B00 | Does a safe-progress five-action target physically exist? | R01 jobs `27306`/`27364` | 17/17 eligible groups | Direct upper bound |
| B01 | Does a constant privileged flow residual transport it? | R03 jobs `27405`/`27450` | 9/17 | Baseline to beat |
| B02 | Does stronger analytic geometry steering preserve safety and progress? | R03A task `27639_0` | 0/1 joint success | Historical diagnostic; safe but insufficient progress |

The eight B01 misses comprise four clearance-only failures and four
progress-only failures. Therefore every efficacy experiment uses joint
Safe-Progress Success; clearance alone is not success.

## R05A experiment sequence

### IFT-00 — Local inverse-control contract

Status: **passed as implementation evidence**

IFT-00 verifies the ten-step control interface, exact Euler recurrence,
target-pairing rules, zero/active masks, path and per-step budgets, deterministic
duplicate search, explicit nonconvergence/nonfinite statuses, and byte-unchanged
no-control sampling on a synthetic nonlinear field. It makes no pi0.5,
simulator, safety, or learnability claim.

Evidence: `evidence/r05a/ift00-synthetic.json` and ADR-0028.

### IFT-00A — One-case real mechanism canary

Status: **active gate; this ledger records terminal history but grants no
execution authority. The current apparatus is unreleased; any future H100
canary requires a new separately reviewed release decision and identity.**

Launch A used exact release commit `223667c91b05be9ab403e4d92d0cd1a96b45246f`,
run `r05a-inverse-flow-sampled-current-canary-20260715a`, and held task
`27928_0`. A shell token parser rejected the valid adjacent pending/held fields
before any receipt, CPU validator, GPU release, allocation, or scientific
computation. The task was cancelled with zero runtime and no node. ADR-0038 and
`evidence/r05a/ift00a-sampled-current-launch-a.json` classify it as
apparatus-inconclusive; its identity is consumed and supplies no transport
outcome.

Launch B used exact release commit `06b365b5899c2cb31db12187350cce48a3a0ea20`,
run `r05a-inverse-flow-sampled-current-canary-20260715b`, GPU task `27962_0`,
and CPU `afterany` job `27963`. The GPU task completed and the raw payload
diagnostically reproduced two deterministic finite 128-update misses. The CPU
wrapper then failed before Python publication because it used parent
`JobIDRaw=27962` as though it were exact task `27962_0`. No candidate or
`results.json` exists. ADR-0039 classifies the launch as
apparatus-inconclusive, permanently consumes its identity, freezes the raw
outcome as diagnostic only, and requires a zero-GPU source-plus-`afterany`
regression before any separately reviewed H100 release.

That regression passed from clean commit `5e595a3`: source task `27975_0` and
validator `27976` both completed `0:0` on worker-1, the production helper
observed exact display task `27975_0`, and all nine source bindings matched.
Both jobs were shell-only, requested one CPU and 256 MiB each, and received no
GPU. This validates the accounting repair only; it does not accept launch B's
raw payload, evaluate teacher transport, or authorize another H100 canary.

Question: can the exact frozen pi0.5 sampler realize one paired safe-progress
target using the registered time-dependent residual velocity sequence?

#### Exact paired source

- case: `crfs-1069f29a8d76463a`;
- group: `safelibero_spatial:II:0:46`;
- source host: `worker-1`;
- environment seed: `924805038`;
- policy seed: `1179198633`;
- exact R02 source SHA-256:
  `055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593`;
- checkpoint SHA-256:
  `988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed`;
- frozen scientific config SHA-256:
  `c31401867f3cdce2b3f443ad021c39dfb812f573b570e1e7434e1f149f79abfb`.

The simulator branch, reset state, observation, instruction, checkpoint,
normalization, and policy noise must match the paired R02 source. Physical
displacement uses checkpoint scale only; normalization mean is never subtracted
from a displacement.

#### Target and vector-field control

1. Recompute the fresh frozen normalized terminal action `x_frozen`.
2. Form `x_target = x_frozen + DeltaA*_model` only in the first-five XYZ
   coordinates. Every other coordinate must remain exactly paired.
3. Cast the immutable R02 correction budget once to float32:
   `B = 3.6398398876190186`.
4. Invert through the exact frozen ten-step pi0.5 sampler to obtain `u_k`.
5. Require `u_0` through `u_4` to be exactly zero. Only steps 5--9 and only
   first-five XYZ may be active.
6. Require total path `P <= B` and every active-step displacement `<= B/5`.
7. Forbid clipping, endpoint overwrite, final-action replacement, and appending
   `DeltaA*` after sampling.

The teacher runs exactly two deterministic 128-update searches. A converged
schedule must also pass an independent explicit-schedule canonical replay. Zero
controls before/after, frozen/eager pairing, recurrence, masks, timing, budgets,
fidelity, parameter gradients, and returned action bytes are independently
validated.

#### Simulator and resource boundary

The environment may perform only its registered reset plus 20 dummy settle
control steps. Policy-generated and teacher-generated action steps are both
zero. No collision or progress efficacy rollout occurs.

The exact allocation is one singleton task `0-0%1` on worker-1: one H100,
eight CPUs, exactly 64 GiB host RAM, two hours, and no requeue. A separate
zero-GPU CPU `afterany` validator uses two CPUs and 8 GiB.

Host telemetry is ADR-0036's exact-job-scope full-lifetime sampled-current
trace. Its high-water is an observed lower bound, not a native peak. Acceptance
also requires an unchanged 64 GiB job hard limit, zero hierarchical
`max`/`oom`/`oom_kill` deltas, lifecycle coverage, bounded sampling gaps, and a
separate nullable native-peak record. The H100 task cannot publish
`results.json`; the CPU validator is the sole publisher.

#### IFT-00A outcomes

| Validated outcome | Research meaning | Decision |
|---|---|---|
| `completed_converged` | One-case teacher-transport mechanism pass | Stop; separately decide whether to preregister IFT-01 |
| finite deterministic `completed_nonconverged` after 128 updates | Negative result for the frozen registered solver | Stop IFT-01 and this solver direction; never claim infeasibility |
| pairing, determinism, nonfinite, OOM, allocation test, telemetry, schema, publication, source-job, or artifact failure | Apparatus-inconclusive | No transport conclusion; repair only the evidenced defect under a new decision |

Attempts A/B, the CPU apparatus regression, and CG-00 are preserved in the
R05A apparatus archive. Sampled-current launches A/B are preserved separately
in the launch archives. No consumed run ID may be reused, and launch B's raw
payload may not be retrofitted into an accepted result.

### CFS-00A — Same-budget constrained-flow diagnostic

Status: **launch A is terminal apparatus-inconclusive; the checked-in apparatus
is fail closed. No real CFS outcome exists yet.**

ADR-0040 replaces the draft's broad planner-to-student proposal with one
causal diagnostic of the observed Run-B miss. It preserves the exact case,
source observation, noise, target, checkpoint, five active flow steps,
first-five-XYZ mask, source budget `B = 3.6398398876190186`, per-step `B/5`
cap, historical Adam solver, and all four physical action-fidelity gates. It
adds no decoded-action budget and performs no new planner or geometry query.

The three paired arms are:

1. **A — historical Run B:** initialize each active integrated increment as
   `Delta_model / 5`, project it, and run the exact historical 128-update Adam
   search. The registered schedule/action hashes and metrics must reproduce or
   the whole comparison is inconclusive.
2. **B — constrained linearized candidate:** differentiate the 35 physical
   first-five seven-channel outputs with respect to the 75 active integrated
   increments at zero control. Solve the fixed product-ball weighted
   least-squares diagnostic with 4,096 deterministic CPU-float64 projected
   FISTA updates, then replay the exact float32 schedule through the nonlinear
   frozen sampler.
3. **C — linearized initialization plus historical Adam:** change only Arm A's
   initial integrated increments to Arm B's validated float32 candidate and
   retain the exact historical optimizer, objective, constraints, and
   selection rule.

Every arm is invoked twice and canonically replayed. The allocation and CPU
publisher independently reconstruct pairing, the physical Jacobian checks,
linear algebra, float32 schedule execution, constraints, recurrence, metrics,
and returned-action bytes. The experiment has only three terminal scientific
classes:

| Validated CFS-00A class | Meaning |
|---|---|
| `mechanism_pass` | Arm A reproduced and Arm B's nonlinear replay or Arm C's selected replay passed all four physical gates; one-case transport support only |
| `frozen_method_negative` | Arm A reproduced and deterministic finite B/C both failed the gates; this fixed method is unsupported on the canary, not infeasible |
| `apparatus_inconclusive` | Any source, reproduction, numeric, determinism, constraint, replay, telemetry, allocation, schema, or publication invariant failed |

CFS-00A executes zero generated actions in the simulator. It cannot support a
collision, progress, safety, generalization, learnability, or infeasibility
claim. It does not authorize a trust-region retry, a changed interface,
IFT-01, a population, label collection, a probe, or an MLP. Any continuation
requires a separate decision after this canary is interpreted.

Exact release `77bf9f6` submitted launch A as GPU task `28021_0` with CPU
validator `28022`. The GPU allocation failed at `allocation_contract` because
the blanket no-symlink input rule rejected the canonical OpenPI virtual-
environment Python launcher. Both registered interpreter paths are intentional
symlinks to existing executable binaries. No allocation test, model, arm,
telemetry, payload, or generated simulator action ran; the CPU validator
correctly refused publication. ADR-0042 permanently consumes the run ID and
permits only a separately reviewed runtime-identity repair and zero-GPU
shell-only validation before any new release.

ADR-0043/0044 then preregistered and released only that shell identity check.
Exact zero-GPU job `28043` completed `0:0` on worker-1 and matched both public
launcher paths, their direct link targets, fully resolved regular executable
files, and frozen resolved-binary SHA-256 values. It requested one CPU and
256 MiB; Slurm allocated two logical CPUs and 256 MiB, with no GPU TRES. No
interpreter, model, metric, simulator, arm, or training path ran. ADR-0045
accepts this as apparatus evidence that the standalone validator recognizes
the frozen production symlink chains. Its integration into the full CFS
workload remains untested; it provides no CFS or safety evidence and does not
itself authorize an H100 submission. Both CFS configs remain fail closed until
this evidence is bound into a separately tested and reviewed release with a
new unused run ID and fresh VinUni-guide preflight.

ADR-0046 implements that binding without changing the CFS scientific
projection. The full local gate passes 611 tests with 198 declared dependency
skips, and two independent reviews found no material scientific, source-
contract, publication, or HPC defect. The implementation remains fail closed;
only a separate three-file direct-child commit may select the next immutable
run ID and authorize one worker-1 submission. These checks are apparatus
evidence and do not yet say whether Arms B or C can transport the target.

The first ADR-0046 release (`118ff0a`, run ID ending `20260716a`) created held
array job `28047` but stopped before release because the receipt checker read
the parent token `ArrayTaskId=0%1` instead of exact task `28047_0`'s
`ArrayTaskId=0`. The inspected held job was cancelled with zero runtime, no
node, and no allocated TRES. No CPU publisher, allocation test, Python, model,
arm, telemetry payload, or simulator action ran. This release is consumed and
apparatus-inconclusive; the only permitted repair is exact-task inspection for
a new immutable run ID.

Corrected release `12a7da6` submitted run ID ending `20260716b` once as GPU
task `28048_0`, pinned to worker-1, with CPU `afterany` validator `28049`.
The GPU task is currently pending for resources; this is an expected queue
state, not a method outcome. Its 58-path source contract and atomic submission
receipt are immutable at SHA-256 `5cbc304f...f2716` and
`ad7d9764...8e98e`. No arm or scientific computation has run yet.

### IFT-01 — Three-case real transport smoke

Status: **blocked on a validated converged IFT-00A and a separate immutable
efficacy decision**

Fixed cases:

| Case | Existing stratum | Required teacher behavior |
|---|---|---|
| `crfs-1069f29a8d76463a` | Constant residual success | Preserve Safe-Progress Success |
| `crfs-7eddaafffb4f9474` | Clearance-only miss | Restore clearance while preserving progress |
| `crfs-bd7b0adf95145623` | Progress-only miss | Preserve clearance while restoring progress |

Every arm must share the exact simulator state, observation, instruction,
policy noise, source host, and executed horizon:

1. frozen pi0.5;
2. fresh direct paired safe-progress target;
3. existing constant residual with path budget `B`;
4. inverse-flow teacher schedule with the same `B` and `B/5` cap;
5. the same teacher controls in reverse flow-step order.

Primary outcome: joint Safe-Progress Success using simulator substep clearance,
contact, progress, and no-pushing measurements. GO requires 3/3 valid teacher
successes. Missing or invalid cases fail the smoke. No MLP training follows a
smoke pass.

### IFT-02 — Seventeen-case development population

Status: **blocked on IFT-01**

GO requires exactly 17 valid fixed-denominator artifacts, reproduction of the
constant 9/17 result, teacher success at least 14/17, preservation of all nine
constant successes, rescue of at least five of eight constant misses, and all
budget constraints. These 17 groups remain development only and cannot be used
for student training or final testing.

### IFT-03 — New-state student authorization

Status: **blocked on IFT-02 and untouched official groups**

This future gate must repeat baseline, direct feasibility, and teacher transfer
on genuinely new source groups, freeze group-preserving train/validation/test
splits, include naturally safe zero-control examples, and separately establish
support coverage before any label collection or learning.

## Current stop conditions

- Do not launch the retired R03A population or reuse any consumed R05A run ID.
- Do not launch IFT-00A before an exact committed execution-release decision.
- Do not change source worker, case, target, checkpoint, noise, solver,
  iterations, optimizer, tolerance, control mask/times, or correction budget.
- Do not launch IFT-01, an efficacy rollout, or a population from IFT-00A.
- Do not train a scalar ECG probe or residual-field MLP.
- Do not accept clipping, action overwrite, a final-step patch, increased
  budget, or clearance without progress as success.
- Do not interpret local/synthetic tests or telemetry capability as research
  efficacy evidence.
