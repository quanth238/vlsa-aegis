# ADR-0069: Accept the valid AEGIS capture and freeze the canary label

Date: 2026-07-17

Status: accepted apparatus and label decision

## Context

ADR-0068 released one repaired capture-only retry for immutable run
`r06-aegis-label-capture-canary-20260717b`. Exact task `28428_0` completed
`0:0` on `worker-1` in 1 minute 52 seconds from clean release commit
`ccb8c5225517f21ca1405f7b1470fe47dab160ee`.

The run atomically published `capture.json`, SHA-256
`f2d32024ac6fac5aa5d133b5138df72501f1590b8cb0927b732edde69dabaaac`.
The allocation validator and an independent local revalidation both report
zero errors. The live record confirms the pinned-source repair:
`SingleArm` / `MountedPanda` with a two-actuator float64 gripper action at
boundary 20.

The capture selected the registered boundary 20, preserved exact state,
observation, instruction, policy noise, returned actions, and five-action
horizon, and matched the accepted R02 observation/action bytes. Both baseline
replays were exact and reproduced the frozen collision with 126 measurements
each. Their raw simulator minimum clearance was
`-0.0070070243639765994` m; the registered conservative `D_sim` was
`-0.004272075333382801` m. MuJoCo reported no physical contact, so this case is
a negative-clearance collision rather than a contact event.

No AEGIS, GroundingDINO, filtering, MVEE, QP, semantic-label, population, or
training step ran inside the allocation.

Two independent read-only reviews found no P0/P1/P2 issue. One reviewer
revalidated the terminal Slurm identity, release ancestry, exact R02 bytes,
boundary/state binding, both 126-sample baseline replays, live controller
record, and lossless image assets. A second reviewer revalidated the closed
label schema, image-array digest, vocabulary, chronology, and fail-closed
post-capture configuration. The complete local gate passes 838 tests with 275
declared dependency skips, 21 artifact audits, and 19 gate audits.

## Label decision

After terminal validation, Codex inspected only the exact lossless agent-view
PNG, SHA-256
`f43a8c9816858fedd263358ac432f0fe490839441441125de44d028f427074c5`,
whose C-contiguous image-array digest is
`5000894309b66b8eec07155949e25a8f03d0363522280836e3e144213c55cfa8`.
The image visibly contains a tall red dairy carton directly in the robot's
path. From the six preregistered public phrases, freeze:

```text
red milk carton
```

The immutable one-row ledger is
`manifests/r06_codex_obstacle_labels_canary.jsonl`, SHA-256
`6a22b6d2f3705c008e338be6b217ce947fabfd4f56f70d3a8f442467694d996f`.
Its review timestamp is `2026-07-17T07:03:29Z`, after capture completion at
`2026-07-17T06:55:59Z`. The record contains no simulator object name or
privileged geometry, and it cannot be relabeled after an AEGIS outcome.

## Decision

Accept retry B as capture-apparatus evidence and permanently consume its run
ID and task. Close the capture execution release and bind the validated canary
label ledger into the draft R06 protocol.

This decision does **not** establish that AEGIS prevents the collision. It
authorizes preparation and independent review of a separate paired-canary
implementation and release only. That future canary must compare the same
frozen pi0.5 baseline against pi0.5 plus the public GroundingDINO,
filtering/MVEE, and full nine-variable AEGIS QP from the exact captured branch.
It must report safety, progress, completion, action modification, and stopping.

The 20-case population remains blocked until that paired canary is terminally
valid. Probe and MLP training remain forbidden.
