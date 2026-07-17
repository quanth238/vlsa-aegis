# ADR-0066: Release the exact AEGIS Codex-label capture canary

Date: 2026-07-17

Status: accepted execution release

## Context

ADR-0064 preregistered the collision-conditioned comparison of frozen
$\pi_{0.5}$ against the public AEGIS control core. ADR-0065 replaced only the
unavailable GLM-4.5V semantic selector with a Codex-frozen label. The label
must be chosen from the exact same-state image before any AEGIS outcome is
observed.

The capture apparatus now preserves the immutable row-0 case, restores the
registered settle boundary, checks the full simulator and read-only controller
identity, reproduces the same baseline collision twice with exact policy
noise and returned-action bytes, and writes losslessly verified 1024-pixel RGB
and depth assets. It cannot invoke GroundingDINO, MVEE, the AEGIS QP, or any
training path. Independent adversarial review found no remaining local
scientific or HPC blocker. The complete local gate passed 830 tests with 274
declared dependency skips, and the dependency-backed focused R06 suite passed
77/77 tests.

## Decision

Release exactly one capture-only Slurm array task for immutable run
`r06-aegis-label-capture-canary-20260717a` from accepted implementation commit
`4bac44e84a208ed4e594ab36492d054d8c3cf05e`.

The release is the direct child of that implementation commit and may change
only:

- `configs/experiments/r06_aegis_collision_conditioned.json`;
- `docs/decisions/0066-release-aegis-label-capture-canary.md`.

The exact task is manifest row 0, case `crfs-1069f29a8d76463a`, pinned to
`worker-1` with one H100, eight CPUs, 64 GiB host RAM, a 30-minute limit,
array `0-0%1`, and no requeue. Pending for resources is allowed. The submitter
must reserve the task held, validate the exact task and receipts, and release
it once. Broad cancellation and automatic resubmission are forbidden.

## Scientific boundary

This release authorizes only baseline reproduction and image/depth capture.
It explicitly forbids:

- AEGIS execution;
- GroundingDINO inference, filtering, or MVEE construction;
- any QP solve;
- semantic labeling inside the allocation;
- population execution;
- probe or MLP training;
- any claim that AEGIS prevents a collision.

After a valid terminal capture, Codex may inspect the captured agent-view
image and freeze exactly one label from ADR-0065's public vocabulary. A
separate reviewed release is required before the paired
$\pi_{0.5}$-versus-$\pi_{0.5}$+AEGIS canary can run.

## Interpretation

A valid capture is apparatus evidence only. It does not evaluate original
end-to-end VLSA/AEGIS because GLM-4.5V is not used, and it does not provide an
AEGIS safety result. A failed or partial capture remains apparatus-inconclusive
and must not be promoted into a method result.
