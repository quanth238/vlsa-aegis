# 0032 — Preregister the R05A CPU apparatus regression

Status: accepted on 2026-07-15 before submission, retry C, or IFT-01.

## Context

ADR-0031 permits only three apparatus repairs after IFT-00A retry B: portable
live-cgroup peak discovery, one authoritative allocation-suite count registry,
and the already accepted ADR-0011 numerical compiled/eager seam. Independent
review then accepted the artifact-only singleton decoder in ADR-0033 and exact
trace-derived status recomputation in ADR-0034. A real H100 retry is not an
appropriate first test of these repaired paths. They must first run together
in a worker-1 Slurm allocation without loading pi0.5 or producing another
teacher observation.

## Decision

1. Use the single immutable run ID
   `r05a-adr0031-apparatus-cpu-20260715a` exactly once.
2. Submit one held `0-0%1` CPU array task to `main/normal/normal`, pinned to
   `worker-1`, with two CPUs, 8 GiB host RAM, 20 minutes, no GPU, and no
   requeue. The user queue must be empty. Release occurs only after an atomic
   submission receipt binds the exact clean, pushed, synchronized commit and
   hashes of the registry, shared suite runner, cgroup helper, allocation
   runner, separately callable result validator, and Slurm file.
3. Run the same shared shell helper used by the H100 canary. Its sole count
   source is `main/crfs_oracle/r05a_allocation_tests.json`. Require exactly
   17 inverse-control, 8 sampler, 10 policy, and 12 canary tests, with zero
   skips and exactly one successful unittest completion per suite. The canary
   suite includes the ADR-0033 singleton-decoding and ADR-0034 exact
   trace/status regressions without changing its registered count.
4. Independently reparse the persisted combined log with
   `_parse_allocation_test_log`; observed and expected maps must equal the
   Python loader's view of the same registry.
5. Exercise the repaired cgroup resolver against the allocation's actual
   `/proc/self/cgroup` and `/proc/self/mountinfo`. Require a positive peak no
   larger than the exact 8 GiB request. Independently parse the sidecar and
   require its version, membership, mount root, mount point, metric path, hash,
   and peak to be internally consistent.
6. Build a hidden result candidate, validate it with the separately callable
   dependency-free validator bound in the submission receipt, then apply the
   final shell guard. Atomically publish `results.json` only after every check
   passes. It must
   state that no GPU was allocated, no checkpoint was loaded, no policy server
   started, no real pi0.5/checkpoint teacher search ran or teacher observation
   was produced, no simulator step ran, no efficacy was measured, no scientific
   claim is allowed, and probe training remains unauthorized. Synthetic
   inverse-solver calls inside the required unit suites are implementation
   tests, not real teacher searches.
   A stage-specific failure receipt replaces a result on any failure; a failed
   validator or guard must never leave a published `results.json`.
7. This run is apparatus evidence only. Even a pass does not repair retry B,
   establish inverse-flow transport, authorize retry C by itself, authorize
   IFT-01, or permit MLP/probe training. Terminal logs, Slurm accounting,
   receipts, sidecars, and the result must receive independent review first.
8. Do not change any frozen R05A source case, checkpoint, noise, target,
   budget, mask, solver iteration, learning rate, tolerance, simulator
   protocol, or the future H100 canary's 64 GiB request.

## Consequences

The test adjudicates only whether the ADR-0031 apparatus repairs and the
ADR-0033/0034 artifact trust-path repairs are executable together on the exact
worker layout that failed retry B. The preregistered run ID is retained even
though its scope now records those accepted pre-submission reviews. A failure
stops before retry C and is diagnosed from the immutable receipt and
stage-specific artifacts. A pass satisfies one prerequisite for a separately
reviewed retry-C decision; it carries no safety, progress, transport, novelty,
or efficacy credit.
