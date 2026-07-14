# 0026 — Validate R03A protocol scalars in their recorded dtype

Status: accepted on 2026-07-15 after source-node H100 task `27638_0`
failed its early-arm active-horizon audit, but before any analytic action,
per-arm status, simulator repeat, clearance, contact, SPS result, or latency
sample was persisted or exposed for inspection.

## Context

Task `27638_0` ran case 0 on its immutable R02 source host, `worker-1`, at
clean commit `9e168855238043eccb8911321e288d7d6245b812`.  Slurm records `FAILED`,
exit `1:0`, after 90 seconds.  It passed exact source pairing and advanced past
the midpoint iteration before requesting the early arm.  Both analytic modes
were attempted; the early arm completed four policy replies (two warmups and
two measured replies, with the measured pair exact), then the runner aborted before its rollout with
`analytic trace active horizon differs`.

The sampler defines the early active horizon mathematically as `(10 - 1) / 10`
and records it in a required scalar `float32` trace leaf.  Its native recorded
value is therefore `float32(0.9)`, or `0.8999999761581421` when promoted to a
Python float.  The validator instead compared that value with the binary64
expression `0.9` using an absolute allowance of `2e-8`; their representation
difference is `2.384185793236071e-8`.  The midpoint value `float32(0.5)` is
exactly representable, which is why the same check did not fail earlier.

The immutable failure artifact SHA-256 is
`658be97eecad63e8e4c4b2d2f1b462c411efcb8379d31bcd90dd9a9af268cea3`.
The client log containing the traceback has SHA-256
`2636a6feab93c621120fd7b50375ead592117ac5022cbfc08a7358248413d6a0`;
the Slurm log has SHA-256
`a9c1a84fb200f6685607f7cf95017c2e573a624ebf63d924e709018dc75df25f`.
The policy-server log, which retains one persistent connection and no inference
exception, has SHA-256
`2d1c302a11ee154eebd05359bc5d0b1f2e0bc55cdd9f4ad64f95bd6566487c78`.
No final R03A result exists.  Control flow proves that both analytic policy
arms were attempted, but the unwritten midpoint status and whether it took zero
or two simulator repeats cannot be recovered.  The early arm completed its
four replies and did not reach simulator rollout.  Nothing from this failed
run can enter an efficacy or latency result.

## Decision

1. Do not increase or tune any tolerance.  For deterministic protocol scalars
   that the sampler records as `float32`, construct the expected value in
   `float32` and require exact `np.array_equal` equality.  Apply this to `dt`,
   `active_horizon`, `safety_margin_m`, and `softplus_tau_m`.
2. Keep the registered numerical audits for source-derived path budget,
   velocity gain, Euler recurrence, and physical transforms unchanged.  They
   are not implicated by this failure.
3. Add an early-arm regression proving that exact `float32(0.9)` passes and
   either adjacent float32 value fails.  Retain the midpoint test.
4. Retry only case 0 with a new immutable run identifier on `worker-1`, with
   unchanged config, arms, seeds, geometry, budget, checkpoint, action gate,
   and kill threshold.
5. Keep the grouped population and probe training blocked until that retry
   produces and passes review of a final schema-valid artifact.

## Consequences

The replacement removes all fixed-scalar tolerances.  For the early horizon it
repairs a binary64-versus-float32 domain mismatch by accepting only the one
contract-correct float32 representation; for the other fixed scalars it
narrows the accepted set to their exact recorded values.  It changes no sampler
value or scientific intervention.  Task `27638_0` remains an apparatus failure
and supplies no analytic safety or latency result.
