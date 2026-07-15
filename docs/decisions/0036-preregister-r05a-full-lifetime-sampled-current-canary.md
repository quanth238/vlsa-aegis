# 0036 — Preregister the R05A full-lifetime sampled-current canary

Status: accepted on 2026-07-15 after terminal CG-00 evidence and before any
full-lifetime implementation commit, immutable H100 run ID, retry C submission,
IFT-01 launch, simulator efficacy rollout, or training.

Pre-execution clarification: implementation review requires the H100 allocation
to rehash and strictly parse its sealed trace after all model work has stopped,
without writing a result candidate.  The CPU validator then independently
rehashes and parses the same raw trace during candidate construction and again
before publication.  Publication review additionally requires the CPU validator
to bind the candidate's exact regular-file identity and bytes across the atomic
rename, reload and schema-validate the published bytes, and roll back the result
and receipt on any mismatch.  The H100 receipt paths are canonical run-root
paths and cannot be inherited from the environment.  These clarifications were
made before any immutable run ID or H100 submission and change no scientific
input or solver setting.

## Context

IFT-00A retry B reached the real frozen pi0.5 sampler and completed two
deterministic 128-update inverse-flow teacher searches, but it could not publish
an accepted result because `worker-1` runs Linux 5.15 and has no cgroup-v2
`memory.peak` interface.  Exact CPU apparatus task `27797_0` reproduced that
interface limitation.  Shell-only CG-00 task `27820_0` then established that the
exact job scope exposes a usable `memory.current`, a positive finite
`memory.max`, and hierarchical `memory.events` counters.  CG-00's
6,045,696-byte sampled high-water is only a lower bound for that tiny shell task;
it is not a pi0.5 memory estimate and is not an exact peak.

ADR-0028 required actual host and GPU memory peaks for the one-case allocation
canary.  No exact host peak can be recovered from the available Linux 5.15 and
Slurm accounting interfaces.  Reusing a sampled `memory.current` maximum in the
old `host_cgroup_peak_bytes` field would be false.  Repeatedly launching the H100
job while retaining that impossible finalization requirement would waste compute
without answering the teacher-transport question.

The research method remains unchanged:

```text
privileged safe-progress Delta A star
  -> paired normalized target action
  -> deterministic inverse solve through frozen pi0.5
  -> budgeted time-dependent residual velocity schedule u_k
```

This decision changes only how the allocation canary records host-memory safety.

## Decision

1. Preserve the existing scientific canary inputs and live science path byte
   exactly.  The following SHA-256 bindings remain frozen:

   - scientific config file:
     `c31401867f3cdce2b3f443ad021c39dfb812f573b570e1e7434e1f149f79abfb`;
   - scientific config projection:
     `9e2ff74cda8ac3b5d4098942a82cc3352cebea3f4b8e1fc1d326c2dc24d2887d`;
   - manifest:
     `bdb8ccbba01ebf500e0f1bd0fe4a4043054f922a273f9e90eb3860cfe753a633`;
   - ADR-0028:
     `d2a00b1b049e92bb1ec8f11d60fa447e5e8bd5109cb8cf3f3fbb08f89498656f`;
   - historical scientific result schema:
     `e1681f865f81f2986945d10fb14073fe4cd9e78cbc5026f6749ab321b9cfb7a7`;
   - exact R02 source:
     `055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593`;
   - R02 config:
     `c5be1b4759487f4d6541e49110b90fcf5de404bdaed5f389358db0abe4388f4e`;
   - R03 summary:
     `dea3e66c854faa0b659777bfed4b651b19d8d4d0710696ff7bfe52c44d4ee76e`;
   - R03 ordered result digest:
     `fa79a132f2fecbedab5917111ef71f022e52c933d8e535aaca118add5f5b7895`;
   - checkpoint:
     `988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed`;
   - normalization asset:
     `b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84`;
   - baseline revision:
     `57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b`;
   - `main/crfs_oracle/r05a_canary.py`:
     `9d4402ccb92835bc1af2a3e04fe54c0b94839f41beb2971022216f73ec67fede`;
   - `main/run_crfs_r05a_canary.py`:
     `247bd20e48ffe2228d2633d2f667a2791371163cca1c43ad7d278f4fbfd7439a`;
   - inverse-control implementation:
     `965082822466774e0a86eeb6f2d178c9e5f3d6d02bca5090c4e1fcc77bc52aa8`;
   - pi0.5 sampler:
     `80366dcc7b2ddc598717d4c71c0e68e46312a1e5ffd3fc04479d5599433f4c55`;
   - policy boundary:
     `d16767ff2073d5c177cdfcc06dc05dcbf7cdb9ef2a2a0150023f7953b03508b9`.

2. Keep the exact scientific case and procedure frozen: case
   `crfs-1069f29a8d76463a`, group `safelibero_spatial:II:0:46`, environment seed
   `924805038`, policy seed `1179198633`, source host `worker-1`, one H100, eight
   CPUs, 64 GiB host request, `0-0%1`, no requeue, and a CPU `afterany`
   validator.  The target, noise, checkpoint, scale-only displacement
   conversion, active mask and times, budget, per-step cap, solver method,
   initialization, 128 updates, learning rate, Adam settings, constraint slack,
   fidelity limits, policy call sequence, and zero simulator teacher-action
   steps cannot change.  No action-space delta may be appended after sampling.

3. Do not modify the historical config or schema to make sampled current look
   like an exact peak.  Add a separately named apparatus config, telemetry
   helper, H100 wrapper, envelope schema, finalizer, CPU validator, submitter,
   and Slurm entries.  The new envelope binds the unchanged scientific payload
   path and SHA-256 plus the separately reviewed telemetry contract and all
   source hashes.  The historical `host_cgroup_peak_bytes` field is never
   populated from sampled current.

4. Resolve the process's unique cgroup-v2 membership through the unique
   longest-prefix cgroup2 mount and stop at the exact
   `job_$SLURM_ARRAY_JOB_ID` boundary.  Never enumerate siblings, descendants,
   or a shared ancestor above the exact job.  The selected mount must have
   hierarchical `memory.events`; a `memory_localevents` mount cannot satisfy
   this gate.

5. Start a distinct host sampler after source/hash/allocation verification but
   before any setup Python, focused tests, policy server, checkpoint load, or
   model inference.  It must atomically signal ready only after recording
   pre-workload `memory.max`, complete hierarchical events, and a first positive
   `memory.current` sample.  Request a 100 ms interval, preserve every canonical
   integer sample index, monotonic timestamp, and byte value, and require
   strictly increasing timestamps with a maximum adjacent gap no larger than
   500,000,000 ns.

6. Preserve lifecycle markers for monitor-ready, policy-server launch, full
   policy-server cleanup, GPU-monitor cleanup, workload-cleanup-complete, and
   monitor-stop observation.  The policy server must use an executable PID
   boundary so `kill` plus `wait` accounts for the real server process.  Stop
   and await the server first, then the GPU monitor, then record workload cleanup,
   then ask the host sampler to take its final sample and seal.  Acceptance
   requires the first sample no later than policy launch and the last sample no
   earlier than workload cleanup.  A failure path may seal diagnostic telemetry,
   but it cannot publish an accepted canary result.

7. At seal time, record `memory.max` again and the complete hierarchical
   `memory.events` map again.  `memory.max` must be a positive finite job-scope
   hard limit, unchanged before/after; it is not usage and is not claimed to be
   the tightest hierarchy limit.  Independently recompute nonnegative
   `max`, `oom`, and `oom_kill` deltas and require all three to be zero.  Record
   `memory.events.local` only as a diagnostic.  Record native `memory.peak`
   capability and value separately; missing native peak is allowed, while a
   present value must be a canonical positive integer.

8. Name the observed maximum
   `host_cgroup_sampled_current_high_water_bytes` and require the explicit flag
   `host_cgroup_sampled_current_is_lower_bound_not_peak: true`.  Recompute it
   from raw samples using integer arithmetic and require
   `0 < high_water <= memory.max`.  Do not use shell floating-point arithmetic
   to validate monotonic nanoseconds.  The H100 allocation-side checker and CPU
   validator must independently rehash and parse canonical integers, sample
   ordering, gaps, summaries, lifecycle coverage, memory limit, and event
   deltas.  CPU candidate construction and prepublication revalidation must
   each re-read the raw trace.

9. The GPU allocation's result-relevant outputs are its atomic scientific
   payload, focused-test log, GPU samples, sealed host trace, and failure receipt
   when applicable.  Ordinary policy/client logs, setup config, and transient
   lifecycle marker files are diagnostic only and cannot enter a result.  The
   GPU allocation never writes a result candidate or final `results.json`.  The CPU `afterany`
   validator is the sole result creator and publisher: it must re-read all raw
   files and immutable receipts, including the post-contract atomic submission
   receipt that names the actual CPU job and dependency, create an atomic hidden
   candidate, validate that candidate again, then atomically rename it to the
   final path.  A missing
   payload, raw-file mutation, validator failure, or nonzero source-job failure
   leaves no final result.  Bind the validator to an externally supplied
   SHA-256 of a pre-CPU source-contract receipt so coupled result/receipt
   tampering cannot redefine the expected job, paths, hashes, or resources.  The
   envelope and final publication receipt must also bind the exact atomic
   submission receipt path and SHA-256, and all source hashes must be checked
   again immediately before publication.

10. Independently validate the unchanged scientific result contained in the
    payload with the existing R05A semantic validator.  A new wrapper may exempt
    only the exact, preregistered legacy errors caused by the deliberately absent
    exact-peak record; any other source, pairing, target, trace, solver,
    determinism, fidelity, status, or no-simulator-use error fails closed.  The
    implementation and tests must freeze that allowlist exactly before any H100
    run.

11. Interpret a completely published envelope in exactly three ways:

    - validated `completed_converged`: IFT-00A mechanism pass only; it does not
      establish collision avoidance, task progress, simulator efficacy, or
      student learnability, and IFT-01 still requires a separate decision;
    - validated `completed_nonconverged`: accepted negative result for this
      frozen teacher; per ADR-0031 stop IFT-01 and the registered solver
      direction, without calling the bounded miss infeasible;
    - `completed_apparatus_failure`, invalid/missing telemetry, pairing or test
      failure, nondeterminism, nonfinite output, OOM, schema failure, source-job
      failure, or missing artifact: apparatus-inconclusive, with no transport
      conclusion.

12. Before choosing an immutable H100 run ID or submitting a job, require
    dependency-free success and adversarial tests for resolver boundaries,
    ready/stop/seal lifecycle, failure cleanup, monitor death, missing/unlimited/
    changed `memory.max`, duplicate/reordered timestamps or indices, excessive
    gaps, forged summaries, new limit/OOM events, lifecycle-marker tampering,
    native-peak separation, raw mutation after candidate creation, external
    receipt binding, CPU-only publication, and schema rejection of any peak
    relabeling.  Then require the complete local gate and independent scientific,
    HPC, and adversarial-test reviews.  This ADR alone does not authorize a GPU
    submission.

## Consequences

The sampled-current contract is weaker than an exact kernel peak but is honest
and allocation-safe: it gives a full-lifetime observed lower bound, a hard
job-scope limit, and evidence that the hierarchy recorded no limit or OOM event.
It cannot reconstruct retry B's missing host peak or prove continuous peak
usage.  Its only purpose is to let one otherwise frozen H100 canary reach an
accepted positive, negative, or apparatus-inconclusive interpretation.

No IFT-01 case, teacher action rollout, simulator efficacy claim, probe, or MLP
is authorized by this decision.  If the corrected canary reproduces retry B's
finite nonconvergence, the controlled conclusion is to stop this solver
direction, not to tune the observed case.
