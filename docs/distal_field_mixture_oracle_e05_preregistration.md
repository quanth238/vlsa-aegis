# Archived E05 normal/tangent field oracle

This post-hoc oracle starts at the exact archived state before action 185 and
uses immutable AEGIS actions 185--186. The nominal accepted seven-slab margin
must reproduce approximately `-9.313104 mm` before any search result is
accepted.

All arms correct the six XYZ values of those two actions, preserve rotation
and gripper, use the same `[-1,1]` action bounds and total L2 correction limit
of `1.0`, and are verified through complete cloned OSC transitions:

1. unrestricted deterministic directions in all six dimensions;
2. nonnegative mixtures of the seven pulled-back clearance normals;
3. the same normal mixtures plus the nominal end-effector progress gradient
   projected tangent to currently active safety rows.

Each arm performs four correction/relinearization rounds with step sizes
`0.05/0.10/0.15/0.25`. Every proposed correction is evaluated by a fresh
two-action clone. A valid candidate requires nonnegative seven-row clearance,
no raw L5--L7 contact, no paper CAR from episode start, and at least 50% of
the nominal two-action end-effector progress. Final directions are also
evaluated at matched total correction norms `0.25/0.50/0.75/1.0`.

The unrestricted arm measures whether any bounded two-action repair exists.
The structured gate passes when either field arm finds an exact safe,
task-progressing repair. Only that result would motivate learning field
mixture weights. This is a privileged simulator oracle, not an online method,
task-completion result, or population claim.
