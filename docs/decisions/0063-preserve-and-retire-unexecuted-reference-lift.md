# 0063 — Preserve and retire the unexecuted reference-lift release

Status: accepted on 2026-07-17 after explicit user authorization to cancel
only exact pending jobs `28311_0` and `28312`, and after terminal Slurm
inspection established that neither job executed.

## Terminal evidence

ADR-0062 released one immutable TRL-00A run,
`r05a-reference-trajectory-lift-canary-20260716a`, from exact clean release
commit `786afe18bf177b4db1e808f6fedb317beac22be8`.  Its source contract,
submission receipt, and final release fingerprint have SHA-256 values
`140e7e13...93a9`, `9ff2c0b8...6b77e`, and `2ef79dcb...4b16`, respectively.

The GPU singleton array task `28311_0` remained `PENDING` for `Resources` with
runtime `00:00:00`, no assigned node, and no allocated TRES immediately before
cancellation.  Slurm records its array parent `28311` as
`CANCELLED by 1073|0:0`, start `None`, elapsed `00:00:00`, node
`None assigned`, and end `2026-07-17T02:22:13`.  CPU `afterany` publisher
`28312` is likewise `CANCELLED by 1073|0:0`, start `None`, elapsed
`00:00:00`, node `None assigned`, and the same terminal time.  Both exact log
paths are absent because neither workload started.

The immutable run root retains twelve control-plane artifacts.  It has no case
directory, scientific `results.json`, publication receipt, policy or model
request, generated simulator step, or allocation output.  The complete path
and SHA-256 inventory plus exact job facts are recorded in
`evidence/r05a/trl00a-cancelled-before-execution.json`.

## Decision

1. Classify this release as
   **apparatus-inconclusive, intentionally cancelled before execution**.  It
   supports neither a positive nor negative conclusion about reference lifts,
   inverse-flow transport, control authority, action fidelity, collision
   avoidance, task progress, generalization, or learnability.
2. Permanently consume run ID
   `r05a-reference-trajectory-lift-canary-20260716a` and jobs `28311_0` and
   `28312`.  Never resume, reuse, resubmit, republish, or synthesize missing
   scientific artifacts for them.  Preserve the immutable run root and exact
   absence of logs/results.
3. Return the live TRL-00A config to fail closed: `ready_to_run=false`, a
   nonempty retirement blocker, no H100 submission authority, and
   `execution_release=null`.  ADR-0062 and the terminal evidence retain the
   historical release identity.
4. Pause R05A without changing or retracting its earlier component evidence.
   No TRL-00A outcome exists.  AF-00A remains the accepted one-case
   `frozen_cem_negative`; the broader flow-transport hypothesis remains
   unresolved rather than rejected.
5. Make the exact collision-conditioned VLSA/AEGIS baseline diagnostic the
   sole active gate, R06, and authorize its additive implementation.  This
   transition grants no job authority by itself. H100 execution remains
   blocked until the fail-closed apparatus is reviewed and the original
   GLM-4.5V credential plus content-bound GroundingDINO config/weights are
   present. A paired canary and complete 20-case population require their own
   frozen protocol, validation, and execution releases. A frozen-label,
   privileged-geometry, or other AEGIS-core substitute is not equivalent or
   authorized as original AEGIS.
6. Continue to forbid probe or residual-field MLP training.  Do not claim
   AEGIS prevents the frozen collisions until paired simulator artifacts are
   independently validated.

## Consequence

The next work is baseline evaluation, not another TRL submission.  The
reference-lift implementation, preregistration, release commit, and immutable
control-plane artifacts remain available for audit, but no active monitor or
launch path should treat them as executable authority.
