# 0060 — Preregister the optimizer-free reference-trajectory lift canary

Status: accepted for implementation on 2026-07-16.  This decision does not
authorize an H100 submission.  A clean reviewed implementation and a separate
direct-child execution release are required before one immutable run may be
submitted.

## Question

On the exact AF-00A canary, can the privileged normalized action delta be
converted into a genuine state- and time-dependent pi0.5 residual field?  If
so, does the resulting field satisfy the unchanged `B` path and `B/5`
per-step control-authority contract?

This is `TRL-00A`, a new optimizer-free mechanism canary.  It is not an
AF-00A retry or solver adjustment.  It uses no CEM, empirical AF population
fit, autograd, Jacobian, finite difference, Adam, FISTA, geometry query,
planner, or simulator feedback.  The retrospective two-times-budget surrogate
from ADR-0059 is not an input to either arm.

## Frozen source and fidelity contract

Freeze the complete AF-00A row-zero source:

- case `crfs-1069f29a8d76463a`, group
  `safelibero_spatial:II:0:46`, environment seed `924805038`, policy seed
  `1179198633`, and source node `worker-1`;
- exact observation, instruction, explicit float32 policy noise, checkpoint,
  normalization asset, baseline revision, immutable R02 source, and source
  hashes already bound by ADR-0050;
- `Delta32`, the elementwise float32 cast of the immutable R02 delta; the
  fresh normalized target equals the paired frozen final plus `Delta32` only
  on first-five XYZ, with every outside-mask byte copied from the frozen
  final;
- `dt32 = float32(-0.1)`, ten Euler steps, intervention step 5, active steps
  5 through 9, and the exact first-five-XYZ control mask;
- `B32 = float32(3.6398398876190186)` and
  `R32 = float32(B32 / float32(5))`;
- the AF-00A weighted physical objective and the four unchanged fidelity
  limits: XYZ max/RMS `0.010` / `0.005`, and first-five seven-channel max/RMS
  `0.050` / `0.015`.

Physical displacements use checkpoint scale only and never subtract a
normalization mean.  Before interpreting a lift, the zero schedule and
equal-split Arm A must reproduce the AF-00A requested/applied schedule, final
normalized action, returned action, objective, and failed gates.  A mismatch
is apparatus-inconclusive.

## Frozen reference trajectory

Let `xbar_j32`, `j = 0,...,10`, be the exact float32 states of the fresh paired
ordinary zero-residual recurrence: the explicit noise followed by its ten
recorded next states.  Define the six active reference fractions using only
the registered float32 operations:

```
alpha_5 = float32(+0.0)
alpha_j = float32(float32(j - 5) / float32(5)), j = 6,...,10.
```

The active bytes are therefore `(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)` in float32.
Construct the reference by copying `xbar_j32`, then updating only masked
coordinates whose weighted delta is nonzero:

```
weighted_j32 = float32(alpha_j32 * Delta32)
ref_j32 = copy(xbar_j32)
ref_j32[mask and weighted_j32 != 0] =
    float32(xbar_j32 + weighted_j32)[mask and weighted_j32 != 0].
```

This preserves every outside-mask byte and every unchanged signed-zero byte.
Require `ref_5 == xbar_5` and `ref_10[mask] == target32[mask]` byte-for-byte.
The policy control supplies these six fully constructed `ref_5..ref_10`
states, not the unshifted `xbar` states.  The sampler consumes each supplied
`ref_(k+1)` directly and must not add `Delta32` a second time.  The runner and
independent CPU validator reconstruct the supplied references from a fresh
zero trace and require that reconstruction byte-for-byte.  The live controlled
recurrence must enter step 5 byte-exactly at `ref_5 == xbar_5` before any lift
is applied.

## Shared online lift arithmetic

Each lift starts from the explicit paired noise and executes ordinary steps 0
through 4 with exact positive-zero control.  At active transition `k -> k+1`
it uses its own current controlled state:

1. Evaluate the frozen ordinary model once to obtain `v_base_k32` through the
   unchanged mixed-precision denoise path.
2. Form the ordinary uncontrolled proposal in production order:
   `base_next_k32 = float32(x_k32 + float32(dt32 * v_base_k32))`.
3. Start from a full positive-zero float32 tensor and set only the mask to
   `raw_c_k32 = float32(ref_(k+1)32 - base_next_k32)`.  This is an integrated
   residual, not a velocity or terminal action edit.
4. The arm-specific rule selects `requested_c_k32`.  Convert only on the mask
   as `requested_u_k32 = float32(requested_c_k32 / dt32)`.  The authoritative
   increment is the sampler trace value
   `executed_c_k32 = float32(dt32 * requested_u_k32)`.
5. Add the residual velocity and execute the existing ordinary Euler
   statement.  Record the actual next state and tracking error.  Never apply
   an iterative rounding correction, line search, action clipping, endpoint
   patch, post-sampling overwrite, or simulator action.

Persist at every step the current state, base velocity, uncontrolled proposal,
desired next reference, raw and requested increments, requested velocity,
authoritative executed increment, total velocity, actual next state, tracking
error, raw norm, applied norm, projection scale, and projection decision.
All tensors are finite float32; norms/scales used by the projection are
recorded with their declared dtype and operation order.

## Arms

### A — frozen equal split

Execute the exact AF-00A Arm-A ordinary `residual_schedule` twice with `B32`.
Both calls must reproduce each other and the immutable AF-00A Arm-A evidence.

### B — same-budget reference lift

At every active step, recompute `raw_c` from this arm's current state.  For the
15 active coordinates, compute `n64` by casting each coordinate to float64,
then accumulating in strict C order as
`sum64 = float64(sum64 + float64(x64 * x64))`, followed by one float64 square
root. Apply the exact product-ball rule:

```
projected64 = raw64                                  if n64 <= float64(R32)
projected64 = raw64 * (float64(R32) / n64)           otherwise.
```

The first branch includes `n64 == 0` and preserves the raw float64 signed-zero
bytes when cast back to float32. It does not canonicalize a zero block.

Cast once to `requested_c32`, then use the shared float32 transport.  After
five steps, validate the authoritative executed increments with the unchanged
eight-nextafter-ULP `B32`/`R32` schedule validator.  There is no remaining-
budget redistribution, adaptive gain, tolerance adjustment, or retry.

Generate B twice from scratch.  Its requested velocity schedule must be
byte-identical across generations.  Replay the first schedule twice through
the existing ordinary `residual_schedule` path with exact `B32`.  Both
replays must match each other and the generated recurrence in every shared
non-timing trace leaf, normalized final, and returned action byte.  B is not
accepted unless it is also an ordinary fixed-schedule witness.

### R — raw reference lift

Set `requested_c32 = raw_c32` without projection and use the same production
transport.  Generate R twice from scratch.  Its only numerical guard is
float32 representability: every reference, state, base velocity, proposal,
control, total velocity, executed increment, and next state must be finite at
the exact point it is formed.  A prospective nonfinite value aborts the
apparatus before applying that step.  No magnitude threshold, clipping,
replacement, partial action, continuation, or accepted scientific terminal
is permitted.  Such a run is `apparatus_inconclusive`, publishes no scientific
`results.json`, and must be repaired and rerun under a separately reviewed
release.  Both raw calls must therefore return the same finite schedule for a
TRL-00A scientific result to exist.

For a finite raw schedule, first test the unchanged validator with exact
`B32`.  If it passes, replay with `B32` and allow the schedule to serve as a
same-budget witness.  If it fails, derive a replay-only diagnostic envelope
without changing the schedule:

```
seed64 = max(float64(path32),
             5 * float64(max_step_norm32),
             float64(B32))
B_replay32 = the smallest finite float32 value not below seed64.
```

The unchanged eight-ULP validator must accept that envelope or the apparatus
fails closed.  Replay the frozen raw velocity schedule twice through ordinary
`residual_schedule` and require the same shared recurrence/final/action byte
equalities as B.  `B_replay32` is validation metadata only; it is never
reported as same-budget support.  Record `path32/B32` and
`max_step_norm32/R32` descriptively.

If every raw pre-projection norm satisfies the exact no-op condition
`n64 <= float64(R32)`, the raw and budgeted generated schedules and scientific
outputs must match byte-for-byte.  Passing the slack-bearing executed-
schedule validator alone does not imply projection was a byte no-op.

## Exact request ledger

A complete finite run issues exactly 18 ordinary policy requests in this
zero-based order:

1. `0`: compiled frozen pre;
2. `1`: eager source trace pre;
3. `2`: eager normalized-final pre;
4. `3`: zero schedule pre/reference;
5. `4..5`: Arm-A ordinary duplicates;
6. `6..7`: budgeted-lift generation duplicates;
7. `8..9`: budgeted ordinary fixed-schedule replays;
8. `10..11`: raw-lift generation duplicates;
9. `12..13`: raw ordinary fixed-schedule replays;
10. `14..17`: zero, eager normalized final, eager source trace, and compiled
    frozen post.

There is no shorter accepted ledger.  A budgeted failure, raw nonfinite value,
or any other raw failure stops immediately as apparatus-inconclusive and
publishes no scientific `results.json`.  Missing, extra, reordered,
deduplicated, or post-failure requests invalidate the run.  Timing metadata is
recorded but excluded from byte determinism.

## Objective, outcomes, and claims

For every complete action arm, independently construct physical error
`returned[:5,:7] - target_physical[:5,:7]`, recompute the AF objective, and
evaluate all four gates.  `xyz_pass` means the first two gates; `full_pass`
means all four.  Neither a budget ratio nor the replay envelope changes a
gate.

Any source/hash/pairing/Arm-A/reference/dtype/equation/determinism/ordinary-
replay/constraint/test/telemetry/schema/publication failure has highest
precedence and yields `apparatus_inconclusive`.

Otherwise classify in this exact order:

1. **`same_budget_lift_pass`:** B, or a raw schedule accepted and replayed
   under exact `B32`, differs from A and passes all four gates.  This is a
   one-case ordinary same-budget witness; it establishes that AF-00A did not
   prove family unreachability and supports the new lift mechanism only.
2. **`canonical_budget_bottleneck`:** B misses, raw passes all four gates and
   exact replay, and the unchanged `B32` validator rejects raw.  This shows
   that this canonical lift represents the target at its measured authority
   while the registered authority blocks this construction.  It does not
   prove that no other same-budget schedule exists.
3. **`canonical_mask_coupling`:** neither full-pass outcome above applies and
   at least one valid finite arm, B or R, passes both XYZ gates but fails a
   full-action gate.  The canonical mask reaches its controlled target but
   does not preserve the required uncontrolled outputs on that path.  Report
   each finite arm and raw authority separately; do not claim global mask
   infeasibility.
4. **`canonical_lift_negative`:** B and finite raw R are valid, neither passes
   both XYZ gates, and no prior outcome applies.  This rejects only the fixed
   linear-reference lift, not global flow reachability.

A raw full pass means only that one deterministic, privileged, state- and
time-dependent five-step residual can be replayed as an ordinary fixed
schedule and reproduce the action target at its recorded authority.  If it
exceeds `B32` or `R32`, it is not evidence for a same-budget method, safety,
progress, generalization, deployment, or student learnability.  Its authority
is an upper bound for this construction, not a minimum required by all fields.

No outcome automatically authorizes simulator execution, IFT-01, a
population, label collection, probe training, or residual-field MLP training.
A same-budget mechanism pass permits only drafting and reviewing a separate
simulator efficacy canary.

## Artifacts and independent publication

The GPU writes only an immutable raw JSON payload, fixed-shape no-pickle NumPy
tensor archive, exact request ledger, allocation test log, and full-lifetime
host/GPU telemetry.  Persist every source/reference/control/trace leaf,
target, alpha byte, raw replay-envelope derivation, objective, gate, and digest
needed to reconstruct the result.

A separate zero-GPU CPU `afterany` job is the sole `results.json` publisher.
It independently rehashes the source, reconstructs the target, reference,
alpha, projection, float32 transport, recurrences, constraints, replay
envelope, objectives, gates, and outcome from raw tensors.  It must not trust
GPU pass Booleans.

## Baseline compatibility and release boundary

Add one reserved opt-in `reference_trajectory_lift` mode to the ordinary
policy server.  Keep its helper isolated.  Do not alter the historical inverse
control, CFS adapter, AF-00A runner, default sampler route, ordinary
`residual_schedule`, or the ordinary Euler statement.  Add regressions for
default byte parity, residual-schedule parity, strict control validation,
reference anchoring, raw and projected arithmetic, projection no-op,
float32 guard, duplicate generation, ordinary replay, the exact finite ledger,
outcomes, schema, and publication.

The draft config starts with `ready_to_run: false`, a nonempty `blocked_on`,
no immutable run ID, and `execution_release: null`.  After the complete local
gate and independent scientific/code/HPC review pass on a clean implementation
commit, one direct-child release may change only the config and release ADR,
bind one unused run ID and every source hash, register one held GPU plus CPU
`afterany` transaction, and release the exact source task.

The provisional envelope is one H100 on `worker-1`, eight CPUs, 64 GiB host
RAM, 30 minutes, array `0-0%1`, and no requeue.  The CPU publisher uses two
CPUs, 8 GiB, 15 minutes, and zero GPUs.  The login node remains control plane
only.  Do not reroute, retry, change resources/method/budget/tolerances, or
launch a next experiment automatically.
