# 0048 — Preserve the failed-derivative diagnostic and normalize terminal transport

Status: accepted from terminal jobs `28212_0` and `28213` on 2026-07-16;
one client-side apparatus repair is authorized, but no new run identity or
execution release is authorized yet.

## Evidence

Immutable run `r05a-constrained-flow-fd-diagnostic-20260716a` used exact
release `3d44b2c5a4779725101d672ed29658a841c85541`, source contract
`7c8eee72c63c17a3922162a6f78eff6467f6be4301e9e42ca6df07377bed3b99`,
and submission receipt
`73eb2320a6e7498bb4c7ab968bd5f90293ed804f3a9ae6b5dbf8f4174cf283cd`.
The source contract bound 60 repository paths to a clean release tree.

GPU task `28212_0` completed `0:0` on worker-1 in `00:03:06`. CPU
`afterany` publisher `28213` completed `0:0` on worker-0 in `00:00:10`. The
CPU job independently validated and atomically published result
`5c843ce5ed4f0e836b43dd6e0b42fec3ae120fa61cb1c56350eac046001062e0`
with status `apparatus_inconclusive`; its publication receipt is
`be91ec15ea08431af63af85298faf3a4015bd527e5eb26d3ea28ae2bd21d32d0`.
The raw payload hash is
`d423da9f1434c7be2ed0c894c7252f764d9a9939e88cae5955c01e1a9a5cbb2d`.
The exact GPU and CPU Slurm log hashes are
`2c35bbb192f61512a9999e85dfacaaeec737cf0f7ae78de3468b347a401b712b`
and `d43d7d220f42008c712df98d1d6903d7aede6d429d3176cd6d7bee2ce97ffacf`.

All 99 registered allocation tests passed with zero skips. Sampled host
high-water was 16,074,977,280 bytes under the unchanged 64-GiB hard limit;
`max`, `oom`, and `oom_kill` event deltas were zero. Sampled device high-water
was 8,513 MiB. Both jobs, publication, source binding, telemetry, and cleanup
were healthy. No generated policy or teacher action was executed in the
simulator.

The real pi0.5 request reached the opt-in adapter's typed
finite-difference-rejection path. The adapter returned exactly one outer key,
`__crfs_terminal__`, containing the diagnostic. The unchanged baseline
WebSocket server then performed its ordinary behavior and appended
`server_timing` to the reply. The paired client received the two-key mapping
but required the transported mapping itself to contain only the reserved key.
It therefore failed with:

`ConstrainedFlowCanaryError: terminal constrained-flow response must contain only its reserved key`

This rejection occurred before the client extracted or serialized the
terminal diagnostic. Consequently no numeric Jacobian or finite-difference
comparison values were preserved. No FISTA solve, Arm-B candidate, nonlinear
replay, Arm-C refinement, four-gate evaluation, or scientific Arm-A
reproduction exists.

Compact evidence is
`evidence/r05a/cfs00a-fd-diagnostic-20260716a.json`.

## Decision

1. Classify this run as **apparatus-inconclusive**. It is neither
   `mechanism_pass` nor `frozen_method_negative`. The teacher-transport
   hypothesis remains untested, not refuted.
2. Permanently consume run `r05a-constrained-flow-fd-diagnostic-20260716a`
   and jobs `28212_0`/`28213`. Never resume them, reuse the run ID, or retrofit
   numeric evidence into their immutable artifacts.
3. Identify the demonstrated root cause narrowly: the policy adapter's strict
   one-key terminal reply crosses an unchanged WebSocket transport that adds
   standard `server_timing`, while the paired client validates the transported
   mapping as if no transport metadata could exist. This is an apparatus
   normalization defect. It does not identify the cause or size of the
   underlying autograd-versus-finite-difference disagreement.
4. Preserve the baseline server and opt-in adapter behavior. Do not remove,
   suppress, or special-case `server_timing` in the global WebSocket server,
   and do not broaden the adapter terminal payload.
5. Authorize only a strict client-side normalization at the transport
   boundary. A terminal reply may contain either exactly `__crfs_terminal__`
   or exactly `__crfs_terminal__` plus `server_timing`. When timing is present,
   the client must require only `infer_ms` or `infer_ms` plus `prev_total_ms`,
   with each value a finite nonnegative built-in float. It may then construct a
   new one-key mapping for the unchanged terminal parser. It must not mutate
   the received mapping, accept any other outer or timing key, coerce values,
   or weaken the existing diagnostic validation.
6. Prove with regressions that the normal baseline response is unchanged;
   exact one-key terminal replies still work; standard timing is normalized;
   extra, missing, malformed, nonfinite, negative, Boolean, integer, or
   otherwise mistyped metadata fails closed; and rejection still stops before
   append, duplicate request, FISTA, candidate creation, replay, Arm C, and
   simulator execution.
7. Freeze every scientific choice from ADR-0040 and ADR-0047. The scientific
   projection remains
   `7dc2c8f63838ae4e22db8a927d87cae89daf0025c931b33e225c946abe8dc915`.
   Do not change the case, source, observation, instruction, noise, checkpoint,
   target, mask, active flow steps, budget, cap, directions, epsilons,
   tolerances, solver, updates, or fidelity gates.
8. Return the checked-in execution apparatus to fail closed. The repair must
   pass focused tests, the complete local gate, dependency-backed allocation
   tests, and independent review. Only afterward may a separate direct-child
   release select one new immutable diagnostic run ID and authorize one
   worker-1 submission.
9. Do not choose that run ID, create an execution release, submit a job, start
   IFT-01, or train a probe/MLP under this decision.

## Scientific boundary

This run confirms only that the typed diagnostic reached the WebSocket
transport and was lost at a strict client-envelope check. It contains no
numeric derivative evidence and cannot establish teacher transport, method
failure, infeasibility, collision avoidance, progress, safety,
generalization, novelty, or learnability.

No tolerance, epsilon, budget, solver setting, target, case, model, or noise
may be tuned using this run. A repaired diagnostic retry must stop at the same
registered finite-difference rejection before FISTA.

## Exact next action

Implement and review only the strict client-side `server_timing`
normalization above. Keep both CFS configs fail closed. After all gates and an
independent review pass, create a separate immutable diagnostic release; do
not select its run ID or authorize execution in this ADR.
