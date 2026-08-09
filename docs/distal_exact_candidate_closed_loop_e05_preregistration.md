# Exact-candidate receding E05 diagnostic

## Question

Can the one exactly safe grid action observed at E05 action 186 support a
collision-free continuation to native task completion when the exact
cloned-OSC safety filter is recomputed after every executed action?

This is a privileged simulator-oracle diagnostic. It is not an affine QP,
learned policy, deployable perception method, or formal whole-body guarantee.

## Frozen intervention

- Replay the immutable 237-action successful released-AEGIS Table-1 plan.
- Before every execution, clone the complete simulator/controller state and
  measure every internal substep over the nominal action and its next action.
- Keep the seven accepted L5--L7 proxy rows and unchanged released-AEGIS EE
  proxy as eight simultaneous outcome constraints.
- If nominal is safe, execute its first action unchanged.
- If nominal is unsafe, evaluate the same lexicographic 8-by-8-by-8 XYZ grid
  in the existing normalized L-infinity 0.5 trust region.
- Retain only actions with raw protected-contact freedom, the obstacle-motion
  gate, and all eight exact minimum margins at least 1 micrometre.
- Select minimum Euclidean XYZ correction, tie-broken by grid index. Do not fit
  an affine certificate and do not call a QP.
- Freshly roll out the selected two-action horizon, execute only its first
  action, require exact clone/execution next-state equality, then recompute.
- Stop without fallback if no candidate passes.

The source config SHA-256 is
`44a0e9448a0b17a77b75fdd6060a4cd1b141a11b775a045c77b770070af28ac8`.
The prior V2 result, payload, and validation hashes are frozen in that config.

## Decision

GO requires native E05 task success with every executed internal substep
nonnegative on all eight proxies, zero raw L5--L7 contact, zero paper CAR, and
exact clone/execution state agreement. This would show that an exact sampled
receding controller can solve this case and provide a target for later
approximation. It would not show QP or neural success.

NO-GO means this registered two-step, first-action-only, 512-action family
cannot complete E05. It does not rule out a longer horizon, adaptive/local
search, or a larger recovery action family.

Scientific execution disables image observables to prevent the previously
observed multi-context OSMesa corruption. Any result gets a separate
single-context, state-hash-verified visual replay.
