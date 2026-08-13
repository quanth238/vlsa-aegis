# Clean task-successful action-risk experiment

## Question

Can a learned predictive safety filter distinguish safe from unsafe Cartesian
actions in episodes where released AEGIS already completes the native
SafeLIBERO task but L5--L7 later causes a protected contact and paper CAR
failure?

The experiment deliberately excludes settled collisions, gripper/hand
contacts, grasped-task-object contacts, unstable other objects, known invalid
perception, and initially unsafe states.  E05/E10 remain diagnostic only;
untouched E42/E42/E44 episodes form the final prediction test.

## Gate A: exact clean labels

For each selected episode, reconstruct the released AEGIS trajectory and take
four states 20, 15, 10, and 5 actions before the first protected L5/L6/L7
contact.  Every state must begin with all seven proxy gaps at least +1 mm and
zero protected MuJoCo contact.

Evaluate exactly 26 first actions:

- the archived nominal AEGIS action;
- zero-motion hold;
- positive/negative world XYZ;
- positive/negative local obstacle normal and two tangents;
- magnitudes 0.5 and 1.0 for every nonzero backup direction.

After obtaining each proposed candidate's successor geometry, the fixed
ledger-independent backup is recomputed there. It evaluates hold,
world-axis, and local normal/tangent actions, each followed by the fixed
25-action zero-motion terminal hold. Every branch is evaluated as one
uninterrupted `proposal -> backup action -> hold` cloned-OSC composition; this
avoids reconstructing an incomplete successor wrapper/controller state. The
selected verified-safe backup action is the corresponding composed branch. If
no verified-safe backup candidate exists,
the complete policy deterministically takes the registered maximum-future-
clearance fallback and retains the resulting unsafe label.  The VLA action
ledger is never an input to the backup candidates or their score.  All
internal MuJoCo substeps are measured.

For seven fixed L5--L7 rows, store positive-is-unsafe risk

\[
R_j(z,A)=1\text{ mm}-\min_{k\in\text{candidate+backup}}h_j(z_k).
\]

Proxy safety requires all seven risks to be nonpositive.  Physical acceptance
additionally requires zero protected MuJoCo contact and paper CAR no larger
than 1 mm.  A case/state without an exact safe candidate is retained as a
failed support case; it is never silently removed.

Gate A passes only when source files and static eligibility reproduce, every
accepted state is initially safe, candidate count/order are exact, all labels
are finite and replayable, every proposal successor agrees across composed
branches within `1e-12`, and
every accepted state has physical/proxy-safe support.  Passing authorizes
training; it is not learned-model evidence.

## Gate B: prediction only

Train one action-conditioned MLP with seven quantitative risk outputs.  Every
state from an episode remains in exactly one of train, validation, test, or
diagnostic.  No QP, calibration, or closed-loop execution is allowed.

The frozen input is 167-dimensional: measured arm joints/velocities, current
EEF/controller goals, obstacle geometry, seven current link-ellipsoid rows,
and the proposed seven-dimensional Cartesian action.  The registered
`256/256/128` SiLU MLP is trained once with deterministic full-batch AdamW and
a boundary-weighted Huber loss.  Only validation episodes select the
checkpoint.  Diagnostic E05/E10 never enter fitting or the final gate.

The learned gate requires on untouched test episodes:

- zero observed false-safe candidates;
- at least 50% safe-action recall;
- at least one accepted exact-safe candidate in every recoverable state.

Only a passed Gate B permits a separately preregistered controller experiment.
Final controller success will require zero L5--L7 contact, paper CAR pass, and
native task completion.
