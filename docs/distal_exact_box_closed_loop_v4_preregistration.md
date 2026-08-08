# Paired distal-prefix closed-loop recovery v4

V4 retains the exact immutable successful AEGIS E05 actions until the first
exact minimum-substep L5--L7 `8 mm` intervention. This preserves the known
failure trajectory and all settled state/action pairing. Immediately after
that first executed correction, the remaining archived plan is discarded and
live pi0.5 plus the unchanged released AEGIS EE QP replans from the corrected
observation every five actions.

The new filter remains exactly seven hard distal constraints against the 15
live MuJoCo boxes. There is no second discrete EE constraint. Candidate set,
finite differences, QP, exact clone verification, raw L5--L7 contact veto,
`0.1 mm` within-step obstacle-motion veto, horizon, and task success criteria
are unchanged from v3.

Passing requires native task completion, no protected contact, no paper CAR,
and exact clone/main agreement. The immutable prefix is evidence pairing, not
a modification of Table 1. Scope remains a single-case privileged feasibility
test rather than deployable, population, whole-arm, or formal safety evidence.
