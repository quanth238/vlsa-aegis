# 0035 — Preregister the R05A sampled-current capability gate

Status: accepted on 2026-07-15 before the capability job, retry C, IFT-01, or
training.

## Context

Exact CPU apparatus task `27797_0` ran on `worker-1` from commit
`00dba0ad27169abd1344a97ae2f02b323e6a52ae`. Its mapper reconstructed the
cgroup-v2 membership ending in `job_27797/step_batch/user/task_0`, but the
selected `memory.peak` file was unavailable. The worker reports Linux
`5.15.0-130-generic`, Slurm reports `CgroupPlugin=cgroup/v2`, and
`JobAcctGatherType=(null)` supplied no `MaxRSS` or `MaxVMSize` fallback.

This is a kernel-interface mismatch, not evidence of high host-memory use.
The authoritative [Linux 5.15 cgroup-v2 documentation](https://www.kernel.org/doc/html/v5.15/admin-guide/cgroup-v2.html)
documents `memory.current`, `memory.max`, `memory.events`, and
`memory.events.local`, but contains no `memory.peak` interface. The
[current upstream documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html)
does document `memory.peak`. Searching allocation ancestors for the same
unimplemented 5.15 interface cannot recover an exact peak.

An alternative contract is scientifically weaker but potentially sufficient
for apparatus safety: poll allocation-owned `memory.current`, record the
observed `memory.max` hard ceiling, and compare hierarchical
   `memory.events` counters before and after the workload. It must never relabel a
sampled maximum as a kernel peak.

## Decision

1. Reserve the immutable run ID
   `r05a-cgroup-v2-current-capability-20260715a` for one shell-only capability
   task. Submit a held `0-0%1` array to `main/normal/normal`, pinned to
   `worker-1`, with one CPU, 256 MiB, two minutes, no GPU, and no requeue.
   Release only after exact held-job validation and an atomic receipt bound to
   a clean, pushed, synchronized commit and source hashes.
2. The allocation reads only its own v2 membership line and the unique
   longest-prefix cgroup2 mount mapping. It constructs the membership chain
   from task leaf to the exact `job_$SLURM_ARRAY_JOB_ID` boundary. It never
   enumerates a cgroup tree and never reads a sibling, descendant, or shared
   ancestor above that job boundary.
3. At each constructed allocation-owned level, record directory availability
   and the availability/value of `memory.current`, `memory.max`,
   `memory.events`, and `memory.events.local`. Check `memory.peak` once at the
   job scope only as a capability bit; a positive integer is required for the
   native-peak outcome, and the probe does not search for it at parents. Record
   `/proc/sys/kernel/osrelease`, the exact membership line, and the selected
   mountinfo line.
4. At the job scope, sample `memory.current` 20 times at a requested 100 ms
   interval. Preserve every monotonic timestamp and value and the maximum
   observed gap. The maximum sampled value is named
   `sampled_memory_current_high_water_bytes` and is explicitly a lower bound
   on the unobserved true peak. In production, timestamps must be strictly
   increasing and the maximum adjacent gap must not exceed 500,000,000 ns.
5. Read hierarchical `memory.events` immediately before and after sampling.
   The selected cgroup2 mount must not carry the `memory_localevents` option;
   if it does, record a completed unsupported outcome because those counters
   are local-only rather than hierarchical.
   The `max`, `oom`, and `oom_kill` counters must be canonical, nondecreasing,
   and have zero deltas for the sampled-current contract to be supported.
   `memory.events.local` is recorded but cannot replace hierarchical events,
   because it excludes descendant events. A positive numeric `memory.max` is
   required and recorded as the configured job-scope hard limit, not as usage
   and not necessarily as byte identity with the scheduler request. It is not
   called the tight effective ceiling because an uninspected ancestor or a
   descendant can impose a stricter limit. The sampled trace
   must also contain a positive `memory.current` observation; zero-only traces
   are completed but unsupported.
6. Publish the bounded diagnostic even when interfaces are absent; absence is
   an outcome, not an execution failure. Fail closed only for an unsafe or
   uninterpretable mapping, wrong allocation contract, malformed counter, or
   non-atomic artifact path. The result permits exactly three outcomes:
   `sampled_current_contract_supported`, `native_memory_peak_available`, or
   `sampled_current_contract_unsupported`.
7. This task loads no checkpoint, starts no policy server, runs no Python,
   model inference, teacher search, simulator action, research-metric
   computation, or training. It is apparatus evidence only. It cannot repair task `27797_0`,
   estimate retry-B host peak, establish transport or efficacy, authorize
   retry C by itself, launch IFT-01, or authorize a probe/MLP.

## Consequences

If the job-owned current/max/events surface is unavailable, if `memory.max` is
unlimited, or if required counters/samples are invalid, the sampled-current
route stops. If a native `memory.peak` appears, retain exact-peak semantics
instead of silently choosing the weaker fallback. If the sampled contract is
supported, a separate reviewed change must instrument the entire future H100
allocation from before model start through process cleanup. That later trace
must retain raw samples, interval gaps, hard ceiling, and event deltas and must
continue to call the sampled high-water a lower bound, never a peak.
