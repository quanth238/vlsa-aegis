# E05 Cartesian controllability and detour-manifold oracle

This no-learning gate asks whether the candidate family—not its learned
ranker—contains a physically safe, task-compatible E05 action.  It binds the
immutable Table-1 state immediately before archived action 182 and the fixed
20-action continuation 182--201.

First, record the initial state and every one of 25 MuJoCo substeps per
action.  Keep four authorities separate: the seven-row compiled-box Loewner
proxy, exact solid intersection between each robot ellipsoid and compiled
obstacle box, raw protected MuJoCo contact, and obstacle displacement.  Ten
paired axis probes on the generic task-relative basis determine when the new
command first affects L5--L7 and how far their centers can move before the
nominal physical violation.  The candidate search is skipped unless the
initial state is physically safe, influence precedes violation, pre-violation
link-center authority reaches 1 mm, and the nominal continuation has a real
physical violation.

The search then compares a continuous five-dimensional soft/free family, a
continuous four-dimensional endpoint-preserving control arm, and a frozen
54-candidate discretization.  Both continuous arms use deterministic
differential evolution over coefficients in `[-0.75,0.75]`; the ten best
boundary candidates per continuous arm receive internal verification, while
all 54 finite-library candidates receive internal verification. The generic
basis uses two directions transverse to nominal task progress, early and late
temporal deformations, and (only in the soft arm) slowdown.  It is constructed
without the known successful E05 correction.

A candidate passes only if all internal substeps have zero exact compiled-box
overlap, zero protected contact, CAR at most 1 mm, terminal end-effector error
at most 15 mm from the nominal continuation, and at least 50% nominal task
progress.  The conservative Loewner support gap is diagnostic and cannot veto
an otherwise physical pass.  Report separately whether the continuous
manifold has support and whether the finite library captures it.  No model,
ranker, QP, primary corrected execution, live VLA query, or closed-loop claim
is authorized.
