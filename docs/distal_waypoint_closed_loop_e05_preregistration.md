# Receding Cartesian waypoint Best-of-N E05 diagnostic

## Question

Can a short coordinated Cartesian waypoint chunk recover from the empty
single-action safe set observed by exact-candidate job `37294`, while keeping
all L5--L7 slabs and the released AEGIS EE proxy safe and preserving native
task completion?

This is a privileged simulator-oracle diagnostic. It is not a QP, learned
model, deployable perception method, or formal whole-body guarantee.

## Frozen intervention

- Start from the immutable successful released-AEGIS Table-1 action sequence.
- Before every execution, exactly roll out the nominal action and its next
  action through cloned OSC and every internal MuJoCo substep.
- If nominal is safe, execute its first action unchanged.
- If nominal is unsafe, build a four-action Cartesian candidate library:
  constant `{-1,0,1}^3` commands, two-phase stop/cardinal turns, and one- or
  two-step lattice diversions followed by the nominal suffix.
- Preserve every nominal orientation and gripper channel. Only normalized XYZ
  translation is changed, always within `[-1,1]`.
- Reject a chunk unless all eight exact proxy margins remain at least one
  micrometre, raw L5--L7 contact is absent, and obstacle motion remains below
  0.1 mm per control step throughout the complete chunk.
- Among safe chunks, prefer native task completion, then the smallest terminal
  EE error to the future waypoint from a parallel read-only replay of the
  immutable successful action sequence, then the smallest chunk correction.
- Freshly rerun the selected chunk, execute only its first action, require exact
  clone/execution next-state equality, and recompute from the new state.
- Stop without an unverified fallback if the registered library is empty.

The parallel trajectory is privileged task-direction information. It isolates
whether coordinated multi-step control—not live VLA competence—can repair the
local action-family failure. A later live-policy experiment is required for a
deployable task proposal mechanism.

## Decision

GO requires native E05 task success with every executed internal substep
nonnegative on all eight proxies, zero protected-link contact, zero paper CAR,
and exact cloned/executed state agreement. NO-GO rejects only this registered
four-action waypoint library and future-EE ranking; it does not rule out live
pi0.5 Best-of-N chunks, longer horizons, or joint-space recovery.

No neural training is authorized by this single-case diagnostic.
