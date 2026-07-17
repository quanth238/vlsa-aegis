# ADR-0070: Accept the exact paired AEGIS canary apparatus

Date: 2026-07-17

Status: accepted implementation; execution not yet released

## Context

ADR-0069 accepted the allocation-backed canary capture and froze the
outcome-blind Codex phrase `red milk carton`. It did not execute AEGIS or
answer whether AEGIS prevents the collision.

The paired-canary implementation now compares the frozen pi0.5 baseline with
pi0.5 plus the public GroundingDINO, filtering/MVEE, and full nine-variable
AEGIS QP from the exact same boundary-20 simulator state. It preserves the
registered observation, instruction, explicit policy noise, nominal actions,
five-action horizon, checkpoint, normalization, and captured RGB/depth bytes.
The original unavailable GLM-4.5V selector remains replaced only by the
previously frozen Codex label.

The implementation repairs six fail-closed apparatus defects found before
execution:

- the frozen-label resolver no longer receives an unsupported constructor
  argument;
- unexpected constructor/programming `TypeError` is apparatus-invalid rather
  than a scientific method failure;
- the old immutable capture is validated against Git blobs at its recorded
  release commit, not against later live source bytes;
- the capture release, label freeze, accepted paired implementation, and
  paired execution release must form strict Git ancestry, with the label bytes
  verified at their freeze commit;
- every paired terminal result must carry the exact run, release, Slurm array
  parent/task, worker, and allocation-visible GPU identity, which the CPU
  validator cross-binds to the source and submission receipts;
- only explicitly allowlisted, dependency-complete public method failures may
  become scientific method failures; untyped or unknown exceptions remain
  apparatus-inconclusive.

The GPU workload is one held singleton array task pinned to `worker-1`, with
one H100, eight CPUs, 64 GiB host memory, a 30-minute limit, and no requeue. A
zero-GPU CPU `afterany` validator is registered before the GPU task is
released. Source, submission, model, normalization, R02 pair, capture, label,
GroundingDINO assets, config, and release ancestry are content-bound.

The complete local gate passes 854 tests with 276 declared local dependency
skips, 21 artifact audits, and 19 gate audits. Allocation preflight requires
the real CUDA, GroundingDINO, Open3D, CVXPY, OSQP, SciPy, robosuite, and
focused R06 test stack with zero skips.

## Decision

Accept the exact paired-canary apparatus and permit one separate direct-child
execution release after independent review. The immutable candidate run ID is
`r06-aegis-paired-canary-20260717a`; manifest row 0 is
`crfs-1069f29a8d76463a`.

A dependency-complete perception or QP failure is retained as a method
failure. A programming, provenance, pairing, dependency, or partial-output
failure remains apparatus-inconclusive. Neither outcome may be silently
excluded.

## Scientific boundary

This decision is implementation evidence only. No paired simulator result
exists yet, so it does not establish that AEGIS prevents the canary collision.
Even a successful canary would authorize only interpretation of this one
collision-conditioned case and preparation of a separately reviewed
population protocol.

The arm must be reported as **pi0.5 plus AEGIS with a Codex-frozen label**, not
as original end-to-end VLSA/AEGIS. It cannot estimate the general SafeLIBERO
rates in Table 1. Automatic population launch and probe or MLP training remain
forbidden.
