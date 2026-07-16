# 0047 — Preserve CFS-00A run B and record failed Jacobian diagnostics

Status: accepted from terminal jobs `28048_0` and `28049` on 2026-07-16;
diagnostic-only implementation is authorized, but another H100 submission is
not authorized until a new implementation is reviewed and released.

## Evidence

Immutable run `r05a-constrained-flow-same-budget-canary-20260716b` used exact
release `12a7da69d3d34050709d08b9ca903a99f9d30862`, source contract
`5cbc304ffda2841e824da9ed0dd926f8f4e912f2aba9cbcd64d6f243fccf2716`,
and submission receipt
`ad7d9764897c1a4297fe10c22573d51f35939d36a00b9ad0b1039b76be18e98e`.
All 58 repository bindings matched the clean release.

GPU task `28048_0` completed `0:0` on worker-1 in `00:03:12`.  CPU
`afterany` publisher `28049` completed `0:0` on worker-0 in `00:00:02`.  The
CPU job independently validated and atomically published result
`655b48f421af3e639a472d6bef8f20c41e2b2ed60dbb33a0a643bb4aa3e2315f`
with status `apparatus_inconclusive`; its publication receipt is
`b8bee6bc497ee70b5d621ef0c13281a6ebfbc0fbeeb9a2f0373fae413fd0b769`.
The exact GPU and CPU log hashes are
`0a94967ad198b46d85239d7aa58e8039867e5155af0220f67204fb97b085e1`
and `27be3c8fa37959821feed933f9abaf68a79ea61cb4cf697e3a0d766e0403ca99`.

The real pi0.5 path reached the zero-control baseline, constructed the
35-by-75 autograd Jacobian, and completed the registered three directions by
two epsilon central-difference comparisons.  The global finite-difference
gate rejected the Jacobian at
`openpi/src/openpi/models_pytorch/crfs_linearized_control.py:596` before
projected FISTA.  The generic exception retained only the rejection message,
not the already-computed numeric arrays.

One ordinary historical request necessarily returned to the paired client
before its comparison request failed.  It was not persisted into a legacy
payload, duplicated, canonically replayed, or independently validated.
Therefore Arm A was not scientifically reproduced.  Arm B produced no FISTA
candidate and no nonlinear replay.  Arm C and the four physical fidelity
gates did not run.

The allocation was healthy: all 91 registered allocation tests passed with
zero skips; sampled host high-water was 16,136,884,224 bytes under an unchanged
64-GiB hard limit; all `max`, `oom`, and `oom_kill` event deltas were zero; and
sampled device high-water was 8,513 MiB.  No generated policy or teacher action
was executed in the simulator.  Compact evidence is
`evidence/r05a/cfs00a-same-budget-launch-b.json`.

## Decision

1. Classify run B as **apparatus-inconclusive**.  It is neither
   `mechanism_pass` nor `frozen_method_negative`.  The transport hypothesis
   remains untested, not refuted.
2. Permanently consume the run ID and jobs.  Never resume them, reuse the ID,
   rerun only a component, or retrofit new numeric evidence into their files.
3. Return both CFS configs to fail closed: `ready_to_run=false`, a nonempty
   blocker list, `execution_release=null`, and no H100 authorization.
4. Identify the demonstrated apparatus defect narrowly: a failed registered
   finite-difference computation loses its already-computed numeric evidence
   at the WebSocket exception boundary.  The root cause of the derivative
   mismatch itself is not yet known.  Precision/quantization, an autograd or
   finite-difference interface mismatch, and genuine local nonsmoothness
   remain separate unresolved explanations.
5. Permit one diagnostic-only repair.  A typed server-local rejection may
   carry detached CPU clones of the exact float32 Jacobian, exact float32
   budget, and existing finite-difference diagnostics.  Only the opt-in CFS
   adapter may convert that rejection to a strict data-only response under
   `__crfs_terminal__`.  The baseline policy and global WebSocket server remain
   unchanged.
6. The paired client must validate exact keys, shapes, dtypes, finiteness,
   fixed reason, and execution boundary, then stop immediately.  It must not
   issue a duplicate comparison, FISTA solve, candidate replay, Arm-C
   refinement, or any generated simulator action after rejection.
7. Persist a new raw variant
   `terminal_finite_difference_rejection`, always classified
   `apparatus_inconclusive`.  The CPU validator must independently reconstruct
   the registered directions and epsilons, `Jd`, central derivatives,
   absolute and relative errors, threshold map, per-direction decisions, and
   failed global decision.  Missing, extra, nonfinite, reshaped, retyped, or
   inconsistent evidence fails closed.
8. Freeze the scientific experiment: case, source state and observation,
   instruction, policy noise, checkpoint, normalization, target, active
   steps, masks, source budget and per-step cap, directions, epsilon fractions,
   relative and absolute tolerances, Arm-A solver, Arm-B FISTA method, Arm-C
   method, and four fidelity gates cannot change.  The scientific projection
   remains
   `7dc2c8f63838ae4e22db8a927d87cae89daf0025c931b33e225c946abe8dc915`.
9. The implementation commit must be fail closed and pass focused tests, the
   full `./init.sh` gate, allocation-dependent tests, and independent review.
   A later direct-child release may change only the two CFS configs and this
   ADR, select one unused immutable run ID, and append an exact one-submission
   release.  Until then, no H100 submission is authorized.

## Scientific boundary

The diagnostic can reveal which direction and epsilon disagreed and by how
much.  It cannot establish teacher transport, method failure, infeasibility,
collision avoidance, progress, safety, generalization, novelty, or
learnability.  It does not authorize IFT-01, a population run, label
collection, probe training, or residual-field MLP training.

No tolerance, epsilon, budget, solver update count, model, target, case, or
noise may be tuned using run B.  Any later derivative-interface repair must be
separately preregistered only after the raw diagnostic is interpreted.

## Exact next action

Implement and validate only the structured failed-diagnostic path above,
preserve all frozen scientific choices, obtain independent review, then create
a new direct-child single-run release.  Do not launch FISTA, IFT-01, or
training automatically.

## Exact execution release

Execution authorization: one preregistered CFS-00A canary submission only.

- Accepted implementation commit: `768ab1c93824c1c8d089c65793fa55ee57257ecc`.
- Immutable run ID: `r05a-constrained-flow-fd-diagnostic-20260716a`.
- Source host: `worker-1`.
- Resources (canonical JSON): `{"account":"normal","array":"0-0%1","cpus_per_task":8,"gpus":1,"host_memory_mib":65536,"partition":"main","qos":"normal","requeue":false,"time_limit":"02:00:00","validator_account":"normal","validator_cpus":2,"validator_dependency":"afterany","validator_gpus":0,"validator_host_memory_mib":8192,"validator_partition":"main","validator_qos":"normal","validator_time_limit":"00:15:00"}`.
- Single submission: `true`.
- Automatic resubmission: `false`.
- Automatic next experiment: `false`.
- Simulator efficacy claim authorized: `false`.
- Infeasibility claim authorized: `false`.
- Probe or MLP training authorized: `false`.

This appendix authorizes only the frozen one-case failed-Jacobian diagnostic canary, which must stop before FISTA after a registered finite-difference rejection. It does not authorize IFT-01, solver or tolerance tuning, simulator execution of generated actions, an efficacy or infeasibility claim, label collection, probe training, or MLP training.
