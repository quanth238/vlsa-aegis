# 0027 — Register the source-node-grouped R03A population

Status: accepted on 2026-07-15 after exact source-node smoke task `27639_0`
completed, but before observing any of the other 16 registered case outcomes.

## Context

Task `27639_0` ran case 0 on its immutable R02 source host, `worker-1`, at
clean commit `fd8871962e1424aa8d43a1d69a52e7663bdb4acf`. Slurm records
`COMPLETED 0:0` after 91 seconds. The final artifact SHA-256 is
`c9fd2f362b487429cc6e896fce827a72d4ab67231f0e2fe018d686ca4943919d`.
Its client, policy-server, and Slurm log SHA-256 values are respectively
`0eff1190867d683af9826b4961921a9194993c9cc7a447f148d652f5beeb89cc`,
`5fab4cfa270e5e8514bae92270e2f15d1bcf1df06877d661a1d212ccb7122acd`,
and `cf99292abedd7b46dbdfb6d3462663d6409ffeae995053af15bbdafbdd0c225b`.
An independently copied artifact had the same hash and produced zero semantic
validation errors and zero Draft-2020-12 schema errors.

All current/source bindings passed exactly, including observation, branch
snapshot, rotated-OBB geometry, noise, frozen actions, and the complete frozen
trace. Both analytic arms completed two warmups, two measured inferences, and
two simulator repeats. Measured actions and science traces were exact across
replies, both simulator repeats were exact, and both pre-intervention traces
were exact to their frozen reference. Neither arm clipped, saturated,
exceeded its R02 path budget, or encountered a zero or nonfinite gradient.

Case 0 is a valid paired negative, not an apparatus failure. Frozen clearance
was -4.272 mm with 36.169 mm progress. The midpoint arm reached 16.596 mm
clearance without contact but only 25.489 mm progress; the early arm reached
14.749 mm without contact but only 26.713 mm progress. Both therefore failed
only the frozen 29.897 mm minimum-progress gate. The observed count is 0/1
for each analytic arm; either arm can still reach the registered 9/17 kill
threshold, so the population decision remains completely unresolved.

ADR-0024 requires the full population to preserve immutable source-host
pairing. Sixteen source artifacts record `worker-1`; manifest index 5 alone
records `worker-2`. Two arrays necessarily produce two Slurm array job IDs,
while the original summarizer accepted only one. Also, the allocation worker
returns nonzero for schema-valid nominal-reconfirmation or policy failures,
although those fixed-denominator outcomes must remain visible to the
population summarizer.

## Decision

1. Accept task `27639_0` as the allocation-backed apparatus smoke only. Its
   one paired outcome may be reported but cannot decide the population kill
   test or authorize probe training.
2. Before reading any source host, hash-validate all 17 immutable R02 artifacts
   against the accepted R03 result set. Require the exact disjoint grouping
   `worker-1: 0-4,6-16` and `worker-2: 5`.
3. Submit two held H100 arrays under one new immutable run ID: worker-1 indices
   `0-4,6-16%1` and worker-2 index `5%1`. Validate both jobs are actually
   pending on explicit user holds with throttle one, write an append-only
   reservation and launch receipt, register the CPU population verifier with
   the exact dual-`afterany` dependency, and write its immutable receipt.
   Release the two exact source job IDs only after that receipt exists. A
   partial submission remains held, visible, and consumes its run ID; it is
   never silently retried or relabeled.
4. Pass the expected source node into every allocation and compare it with the
   allocation hostname before creating a case directory, loading a policy, or
   running Python.
5. Bind the population summary to its own exact Slurm job ID and the two exact
   source job IDs by immutable source
   host, global manifest task index, `main` partition, and the grouped launch
   and summary-submission receipts. Register its dependency before release as
   `afterany:<worker-1-job>:<worker-2-job>`. This permits schema-valid fixed-
   denominator failure artifacts to be summarized; missing, invalid, wrongly
   hosted, wrongly indexed, or wrongly committed artifacts still fail closed.
6. Keep the manifest, config, arms, seeds, geometry, R02 per-case budget,
   9/17 threshold, schema, endpoint-free SPS gate, and descriptive-only timing
   protocol unchanged. Keep the ungrouped array rejected and keep probe
   training blocked through population interpretation.

## Consequences

The next claim-bearing action is the complete 17-case grouped population, not
another case-0 smoke and not probe training. Case-0 latency remains two-sample
descriptive telemetry only. The grouped result can determine whether either
strong analytic arm reaches 9/17; it still cannot establish learned-probe
efficacy, transport, novelty, or deployment latency.
