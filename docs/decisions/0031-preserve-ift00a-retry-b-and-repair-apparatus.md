# 0031 — Preserve IFT-00A retry B and repair only its apparatus

Status: accepted on 2026-07-15 after jobs `27726_0` and `27727` were terminal,
and before any retry C, solver change, efficacy rollout, or IFT-01 launch.

## Context

IFT-00A retry B used immutable run
`r05a-inverse-flow-canary-20260715b` at clean source commit
`0cc9cfb1070de8a06e23a229e556dc27aa8c600a`. GPU task `27726_0` ran on the
required source host `worker-1` for 5 minutes 8 seconds and exited `6:0`. Its
CPU `afterany` validator `27727` then failed closed because `results.json` did
not exist.

This was not an out-of-memory exit. The exact GPU log says only
`live Slurm cgroup memory peak is unavailable`. The wrapper reached that check
after the real pi0.5 client completed and wrote a 7 MB canary payload, but
before atomic memory finalization. The payload and GPU samples report a peak of
about 8.5 GiB GPU memory. Host peak use remains unknown.

The unfinalized payload is useful diagnostic evidence but is not an accepted
canary artifact. It binds the immutable R02 source action and trace exactly,
constructs the float32 target under the source budget, starts real pi0.5, and
runs the inverse-flow teacher twice. Both deterministic 128-update searches
return finite status code 3 and fail closed to the frozen action. Their target
errors are far outside the frozen fidelity limits: first-five XYZ maximum
`0.8631912` and RMS `0.3312218`, with full-action maximum `0.8631912` and RMS
`0.2169745`. This is evidence of a bounded search miss, not an infeasibility
certificate and not a simulator safety or task-progress result.

Three apparatus defects prevent an accepted interpretation:

1. The wrapper assumes fixed cgroup mount paths and could not read the live
   Slurm host-memory peak. Its failure receipt also retained the stale stage
   name `r05a_canary_payload` even though payload generation had completed.
2. The allocation wrapper correctly ran 12 R05A canary tests, but
   `ALLOCATION_TEST_COUNTS` in the finalizer still expected 10. This mismatch
   would have rejected finalization after the memory reader was repaired.
3. The canary classified compiled-versus-eager byte inequality as a zero-path
   failure. Both compiled calls were byte-stable before/after, both eager calls
   were byte-stable before/after, and eager source action/trace pairing was
   exact. The cross-path difference passed the unchanged ADR-0011 physical
   limits: first-five XYZ maximum/RMS `0.0044150`/`0.0020751` and full-action
   maximum/RMS `0.0086891`/`0.0025280`. ADR-0011 already established before
   R02 outcomes that compiled/eager byte equality is diagnostic while these
   numerical limits are the acceptance rule.

## Decision

1. Preserve retry B, its exact jobs, logs, payload, failure receipt, and CPU
   receipt as an incomplete apparatus run. It does not pass IFT-00A and cannot
   authorize IFT-01 or training.
2. Treat the two real teacher calls only as diagnostic evidence of finite
   nonconvergence. Do not call the miss infeasible, do not claim collision or
   progress efficacy, and do not tune a solver setting or tolerance from it.
3. Repair host-memory collection by resolving the allocation's actual cgroup
   mount and membership path rather than assuming `/sys/fs/cgroup` layout.
   Keep the required measurement as a positive live Slurm cgroup peak; do not
   replace it with requested RAM, `nvidia-smi`, a fabricated value, or an
   unscoped process RSS. Set the failure stage before attempting the read and
   record enough path provenance to diagnose another failure.
4. Change the finalizer's R05A suite count from 10 to 12 and add a regression
   that compares the shell runner, finalizer registry, and declared test
   methods. No skipped suite can satisfy this gate.
5. Restore the already accepted ADR-0011 path semantics in R05A. Continue to
   require exact eager current/source action and native-trace pairing, exact
   zero-schedule/frozen equality, exact compiled before/after stability, and
   exact eager before/after stability. Retain compiled/eager `array_equal` as a
   diagnostic, but gate that seam by independently recomputed unchanged
   ADR-0011 physical maximum/RMS limits. Do not introduce or tune a tolerance.
6. Do not change the frozen target, source case, noise, path budget, per-step
   cap, active mask/time, 128-update solver, learning rate, checkpoint,
   simulator protocol, or 64 GiB request for the apparatus retry.
7. A new immutable retry may run only after focused tests, the complete local
   gate, an allocation-backed CPU regression, clean commit synchronization,
   and independent review pass. An unchanged rerun of retry B is forbidden.
8. If the corrected accepted artifact reports the same finite nonconvergence,
   stop IFT-01 and the currently registered inverse-flow teacher direction.
   Any later solver-method pivot must be a new preregistered experiment, not a
   tuned retry on this observed case.

## Consequences

The current research direction is neither confirmed nor formally refuted by
retry B. It has, however, received a strong negative diagnostic on its easiest
preservation stratum: the registered solver did not transport the target for a
case where the historical constant residual succeeded. The next experiment is
therefore a strict apparatus-only adjudication, not an opportunity to improve
the solver after seeing its miss.

The compact immutable interpretation is
`evidence/r05a/ift00a-attempt-b.json`, SHA-256
`83a09684d474dcc3b38004facd3901fe274b7fcad3b9133827f2ba5bf82d37a4`.
No teacher-generated action has been executed in the simulator, and no probe
or residual-field MLP may be trained.
