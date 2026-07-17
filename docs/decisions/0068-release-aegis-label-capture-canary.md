# ADR-0068: Release the repaired AEGIS Codex-label capture canary

Date: 2026-07-17

Status: accepted execution release

## Context

ADR-0067 permanently consumed failed capture run
`r06-aegis-label-capture-canary-20260717a` and exact task `28409_0`. That run
published no `capture.json` and produced no AEGIS, GroundingDINO, MVEE, QP,
semantic-label, or training result.

The accepted implementation now validates the exact pinned controller
contract at the selected boundary: `SingleArm` / `MountedPanda`, one abstract
gripper value at boundary zero, two actuator commands after a complete control
boundary, and the exact registered controller-array fields. A NumPy-backed
generator-to-validator and full policy-pairing round trip cover that repair.
The complete local gate passed 834 tests with 275 declared dependency skips,
21 artifact audits, and 19 gate audits. Independent code and HPC reviews found
no material issue.

## Decision

Release exactly one repaired capture-only Slurm array task for immutable run
`r06-aegis-label-capture-canary-20260717b` from accepted implementation commit
`6c428d246ab810e9079181d96e6801ea00f68338`.

This release is the direct child of that implementation commit and changes
only:

- `configs/experiments/r06_aegis_collision_conditioned.json`;
- `docs/decisions/0068-release-aegis-label-capture-canary.md`.

The exact task is manifest row 0, case `crfs-1069f29a8d76463a`, pinned to
`worker-1` with one H100, eight CPUs, 64 GiB host memory, a 30-minute limit,
array `0-0%1`, and no requeue. The submitter must reserve the task held,
validate the exact task and immutable receipts, and release it exactly once.
Pending for resources is allowed. Automatic resubmission, rerouting, and broad
cancellation are forbidden.

## Scientific boundary

This release authorizes only:

- reconstruction of the registered boundary-20 branch;
- exact frozen pi0.5 policy replay;
- two baseline simulator replays that must reproduce the collision;
- lossless same-state RGB and depth capture;
- atomic publication and independent validation of the capture artifact.

It forbids AEGIS execution, GroundingDINO inference, filtering, MVEE, every QP
solve, semantic labeling inside the allocation, population execution, and
probe or MLP training.

Only a valid terminal `capture.json` may be shown to Codex to freeze one
semantic label from the six registered phrases. A separate reviewed release
is required before the paired pi0.5-versus-pi0.5+AEGIS canary.

## Interpretation

A valid result is capture-apparatus evidence only. It does not answer whether
AEGIS prevents the collision and cannot support an original end-to-end
VLSA/AEGIS claim because GLM-4.5V is unavailable. Any failed or partial retry
remains apparatus-inconclusive and cannot authorize a label or later
experiment.
