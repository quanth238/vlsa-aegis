# ADR-0067: Preserve the failed AEGIS capture and repair only the controller-state contract

Date: 2026-07-17

Status: accepted apparatus decision

## Context

ADR-0066 released one capture-only canary for row-zero case
`crfs-1069f29a8d76463a`. Exact task `28409_0` ran on `worker-1` from release
commit `cb57d52cab7fee8e25de74e86b6d76fa876da787`. It failed after 1 minute
43 seconds with exit `1:0`. The source contract, held-submission receipt, logs,
failure receipt, and partial image/depth files remain immutable under run
`r06-aegis-label-capture-canary-20260717a`.

The runner reached its internal post-replay validation and reported exactly
one error:

```text
capture policy/source binding is invalid
```

No `capture.json`, validation log, or scientific result was published.
GroundingDINO, MVEE, the AEGIS QP, semantic labeling, and training did not
execute. The run is therefore apparatus-inconclusive and provides no evidence
for or against AEGIS collision avoidance.

## Root cause

The failed run did not persist its in-memory controller record. A deterministic
postmortem of the exact pinned runtime sources nevertheless exposes two wrong
validator assumptions; the fresh capture remains the runtime confirmation.

First, all 20 frozen cases are `safelibero_spatial` task zero. Its
`Libero_Tabletop_Manipulation` environment rewrites the requested `Panda` to
`MountedPanda`; robosuite then constructs class `SingleArm` and stores the
passed robot type as its name. The exact pinned sources therefore require the
next live record to identify `SingleArm` / `MountedPanda`, not
`SingleArm` / `Panda`.

Second, `PandaGripper.dof` is the one-dimensional abstract gripper action, but
`PandaGripper.format_action` broadcasts it into the two actuator commands and
stores those two values in `current_action`. Boundary zero has no controller
call and legitimately retains shape `[1]`. Every later complete settle
boundary has at least one dummy control and therefore has shape `[2]`.
The pinned runtime therefore predicts shape `[2]` at boundary 20, while the
failed validator required `[1]`; the fresh capture must confirm that live
record.

The exact allocation runtime source hashes and terminal artifact hashes are
recorded in
`evidence/r06/aegis-label-capture-canary-20260717a.json`.

## Decision

Permanently consume run `r06-aegis-label-capture-canary-20260717a` and exact
task `28409_0`. Do not freeze a Codex label from its partial image files and do
not retrofit a capture artifact.

Repair only the demonstrated controller-state validation contract:

- require the frozen diagnostic identity `SingleArm` / `MountedPanda`;
- require one gripper-state value at boundary zero and two after any complete
  dummy control;
- bind the validator to the selected boundary instead of accepting either
  shape without context;
- require the exact registered controller-array field names;
- add dependency-light shape/identity rejection tests and a NumPy-backed
  generator-to-validator plus full policy-pairing round trip.

The simulator state, observation, instruction, policy noise, nominal actions,
five-action horizon, settle boundaries, safety margin, metrics, resources,
worker pin, and no-AEGIS capture boundary remain unchanged.

## Next release boundary

After the complete local gate and independent repair review pass, a direct
child release may select one fresh immutable run ID and submit one new
capture-only row-zero task. That release must still forbid GroundingDINO,
MVEE, QP, semantic labeling, population execution, and probe/MLP training.

Only a valid newly published `capture.json` can authorize Codex image review.
A separately reviewed paired-canary release remains required after the label
is frozen.
