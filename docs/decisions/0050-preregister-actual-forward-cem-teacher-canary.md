# 0050 — Preregister the actual-forward CEM teacher canary

Status: accepted for implementation on 2026-07-16. This decision does not
authorize an H100 submission. A clean reviewed implementation and a separate
direct-child execution release are required before one immutable run may be
submitted.

## Question

Can the exact paired safe-progress action correction be represented by a
budgeted, time-dependent residual velocity schedule when the teacher searches
only against outputs of the frozen pi0.5 sampler's actual forward path?

This is the direct continuation of ADR-0049. It does not repair, approximate,
or bypass the rejected autograd Jacobian. It does not test collision avoidance
or task progress in the simulator.

## Frozen paired source

AF-00A uses only manifest row zero:

- case `crfs-1069f29a8d76463a` and group
  `safelibero_spatial:II:0:46`;
- environment seed `924805038` and policy seed `1179198633`;
- exact R02 source SHA-256
  `055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593`;
- checkpoint SHA-256
  `988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed`;
- exact fresh observation, instruction, float32 policy noise, frozen baseline,
  checkpoint normalization, and zero-preserving float32 target construction;
- source-reported float64 budget `3.6398398429065115`, cast exactly once to
  float32 `3.6398398876190186`.

Physical displacement uses checkpoint scale only. The normalization mean is
never subtracted from a displacement.

## Control variable and ordinary model path

The scientific variable is the integrated increment `c_k = dt * u_k`, with
`dt32 = float32(-0.1)`. `Delta32` is the elementwise float32 cast of the
immutable R02 delta, not a subtraction reconstructed from the target. `B32`
is the registered float32 budget, `R32 = float32(B32 / float32(5))`, and the
search radius is `R64 = float64(R32)`. The variable has exactly 75 active
coordinates: first-five XYZ at flow steps 5 through 9. Steps 0 through 4 and
every other coordinate are exact zeros. Each active 15-vector has L2 norm at
most `R32`, and the sum of the five norms is at most `B32` under the unchanged
validation rule below.

The ordinary `residual_schedule` request expects velocity `u`, so every
candidate follows the exact order `c32 = float32(project(c64))`, then
`u32 = float32(c32 / dt32)`. The authoritative executed increment is the
sampler trace's `float32(dt32 * u32)`, not the pre-conversion `c64` or `c32`.
Every ledger row records all three representations. The existing schedule
validator remains authoritative: it computes float32 per-step norms and their
float32 sum from `dt32*u32`, and permits exactly eight successive float32
`nextafter(+infinity)` ULPs above `R32` and `B32`. No returned action is
post-clipped, appended, or overwritten. The preregistered radial projection is
the only candidate modification before transport.

The implementation must use the ordinary policy server and existing
`residual_schedule` sampler path. It must not call `inverse_flow_teacher`,
autograd, the retired CFS adapter, the Jacobian, FISTA, or the historical Adam
teacher. No OpenPI sampler or policy implementation change is authorized.

## Arms

- **Z, paired zero:** exact zero schedule before and after the experiment. It
  is an apparatus reference only.
- **A, equal split:** construct `cA32 = float32(Delta32 / float32(5))` for
  every active flow step, then apply the same `c32 -> u32 -> dt32*u32`
  production round trip and unchanged eight-ULP schedule validation. It is not
  historical Adam.
- **B, actual-forward CEM:** the selected schedule from the fixed search below.
- **C, reversed B:** reverse B's five exact requested float32 velocity rows
  (`u32[5:10]`) byte-for-byte and replay them twice. Do not reverse executed
  increments and reconvert them. This is a timing-order diagnostic and may
  never replace B after the result is observed.

There is deliberately no constraint that `sum_k c_k = Delta_model` for Arm B.
Imposing it would reintroduce an endpoint-style premise and prevent the
nonlinear vector field from transporting the target through its own dynamics.

## Fixed actual-forward CEM

The optimizer is a diagonal cross-entropy method over compact shape `(5, 15)`.
All search state and random samples are float64 NumPy values. Every model
request is float32.

1. Search seed: integer `20260716`; bit generator: NumPy `PCG64`. Record the
   runtime NumPy version and every raw standard-normal sample so validation
   never assumes cross-version random-stream portability.
2. Run exactly eight generations of 65 candidates: the current center followed
   by 32 antithetic Gaussian pairs in pair order. This is exactly 520 search
   rollouts. There is no early stopping.
3. Initialize the center from Arm A's actually executed float32 increments,
   converted to float64. Initialize every diagonal standard deviation to
   `sigma0 = R64 / sqrt(float64(15))`.
4. In generation `g`, candidate zero is exactly `mean_g`. For pair
   `p = 0..31`, draw and store one `z[g,p]` of shape `(5,15)`, then emit
   `mean_g + sigma_g*z[g,p]` followed by `mean_g - sigma_g*z[g,p]`. Every
   operation in this paragraph is float64. Project each 15-vector `v` as
   exact zero if its norm is zero, unchanged if `||v||_2 <= R64`, and
   `v * (R64/||v||_2)` otherwise. Then use the exact `c32 -> u32 -> dt32*u32`
   order above and update all scientific records from the sampler's actually
   executed float32 increment.
5. For returned physical action `a`, let `a*` be the independently constructed
   physical target and compute in float64
   `e = a[:5,:7] - a*[:5,:7]`. Define
   `q = mean(concat((e[:,:3]/0.005).ravel(),
   (e[:,3:7]/0.015).ravel())**2)` over exactly 35 values. Rank candidates for
   the distribution update by `(q, global_policy_request_index)`. Exact
   float64 equality defines a tie. The XYZ RMS gate uses exactly 15 values and
   the seven-channel RMS gate exactly 35 values.
6. Use the first 13 ranked candidates as unweighted elites. The next mean is
   the float64 mean of their executed increments, projected blockwise by the
   rule above. For every coordinate compute
   `variance = mean((elite_executed - projected_mean)^2)` (`ddof=0`), then
   `sigma = sqrt(variance)` and clip to `[sigma0/16, sigma0]`. There is no
   smoothing.
7. Store the raw normals, raw proposals, projected float32 increments,
   transported velocities, executed increments, final normalized actions,
   returned physical actions, objective, all four fidelity metrics, elite
   indices, means, variances, query timings, and exact hashes.

The four unchanged physical fidelity gates are:

- XYZ maximum absolute error at most `0.010`;
- XYZ RMS error at most `0.005`;
- first-five seven-channel maximum absolute error at most `0.050`;
- first-five seven-channel RMS error at most `0.015`.

Elite ranking does not decide the final teacher. The final pool contains one
canonical Arm-A record plus all 520 search evaluations; Arm A is not an extra
CEM elite or population member. Define energy as
`E = sum(float64(executed_increment)**2)` over exactly 75 active values. Pool
index zero is Arm A; pool indices `1..520` are CEM search queries `0..519` in
request order. If any pool member passes all four gates, select
lexicographically by `(E, q, global_policy_request_index)`. If none passes,
select by `(q, E, global_policy_request_index)`. Float ties are exact. Arm A's
canonical global request index is 4 and CEM queries use `6..525`. Arm A remains eligible as the
unchanged incumbent, so a selected finite miss cannot be silently worse than
the baseline objective. “B changed” means both B's exact requested and
trace-applied float32 velocity schedule bytes differ from Arm A.

## Exact request and repeatability ledger

A complete nonterminal search uses exactly 534 policy requests:

1. four paired pre-search requests: compiled frozen, eager source trace, eager
   normalized final, and zero schedule;
2. two byte-identical Arm-A repeatability requests;
3. 520 CEM search requests;
4. two selected-B canonical replays;
5. two reversed-B replays;
6. four paired post-search requests: zero schedule, eager normalized final,
   eager source trace, and compiled frozen.

Before CEM search query 0, both Arm-A replies must match in requested/applied schedule,
initial noise, deterministic recurrence trace, normalized final, and returned
action bytes. Failure stops before CEM as apparatus-inconclusive. If an
identical executed schedule occurs more than once during search, its
scientific output must be byte-identical.

Both selected-B replays must match the selected pool evaluation and each
other in schedule, recurrence, normalized final, and returned actions. Both
reversed-B replays must match each other. The post-search paired references
must match their pre-search counterparts. Global policy request indices are
zero-based: pre-search calls are `0..3`, Arm A is `4..5`, CEM is `6..525`, B
is `526..527`, C is `528..529`, and post-search is `530..533`. Even when B or
C duplicates A, all registered replay requests still execute. Every
non-apparatus outcome requires exactly 534 requests. A caught apparatus fault
must issue no later requests, but it need not publish a partial request ledger;
an OOM, signal, or process death may preserve only the allocation wrapper,
client/server log, telemetry, and failure-stage evidence. Such a run publishes
no scientific result. Missing, extra, reordered, deduplicated, or post-failure
calls in a complete run are invalid.

## Raw artifact and independent publication

The GPU allocation writes only immutable raw evidence:

- a small JSON payload and query ledger;
- an atomically sealed fixed-shape NumPy tensor archive for the 520-query
  arrays and replay arrays;
- allocation tests and full-lifetime host/GPU telemetry.

A separate zero-GPU CPU `afterany` job is the sole `results.json` publisher.
It must rehash every raw file, reconstruct the paired target and budget,
regenerate the CEM proposals from the stored normal samples, recompute every
projection, objective, gate, elite update, final selection, schedule
constraint, repeatability comparison, and outcome. It must not trust GPU pass
Booleans.

## Outcomes and stop rules

- **`mechanism_pass`:** Arm A fails, Arm B is changed and passes all four
  gates, all 534 calls and independent validations pass, and both B replays are
  exact. This is one-case teacher-transport support only. It permits drafting
  and reviewing IFT-01, not launching it.
- **`baseline_sufficient_no_incremental_support`:** Arm A itself passes. The
  canary gives no evidence that CEM is needed.
- **`frozen_cem_negative`:** the full valid search completes but B misses at
  least one gate. This rejects only this fixed 520-query CEM on this case; it
  is not an infeasibility certificate.
- **`apparatus_inconclusive`:** pairing, repeatability, nonfinite values, query
  accounting, constraints, reconstruction, replay, regression, OOM, tests,
  telemetry, schema, publication, or source identity fails.

A reverse-only pass is diagnostic and does not authorize IFT-01. A B pass with
an equal or constant active schedule supports residual transport but not a
time-dependent advantage. Timing necessity requires an order-sensitive result
and remains separate from transport feasibility.

Outcome precedence is fixed. Any apparatus fault dominates. Otherwise, after
all 534 requests, an Arm-A pass yields
`baseline_sufficient_no_incremental_support`; if A fails, a changed Arm-B pass
yields `mechanism_pass`; every other finite complete result yields
`frozen_cem_negative`.

No AF-00A outcome supports a simulator safety, collision, task progress,
generalization, latency, novelty, probe, or MLP claim. No generated action is
executed in the simulator. IFT-01, population execution, label collection, and
training remain blocked.

AF-00A therefore tests only the necessary actual-forward teacher-transport
link. It does not by itself validate the proposed learned safety method.

## Resource and release boundary

Retain the proven singleton worker-1 envelope: one H100, eight CPUs, exactly
64 GiB host RAM, array `0-0%1`, no requeue, and a two-hour upper time limit.
The CPU publisher uses two CPUs, 8 GiB, zero GPUs, and `afterany`. Do not raise
host memory to 128 GiB.

Implementation must land first with the new config fail closed. A separate
direct-child release may change only that config and its release ADR,
select one unused immutable run ID, bind every source hash, register the held
GPU plus CPU-afterany jobs, and release the exact source task only after all
receipts validate. No automatic retry, reroute, resource change, or next gate
is permitted.

## Exact next action

Implement the pure-NumPy CEM, direct ordinary-server canary, independent CPU
validator/publisher, allocation tests, telemetry wrappers, and held-release
transaction. Keep both configs fail closed until a complete local gate and
independent implementation/release review pass.
