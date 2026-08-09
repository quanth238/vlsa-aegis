# E05 full-action-bound recovery diagnostic

## Question

At immutable Table-1 E05 action 186, does recovery fail only because the
nominal-centered L-infinity 0.5 trust region excludes stop and upward retreat?

This is a single-state, privileged-simulator feasibility diagnostic. It is
not learned generalization and cannot authorize closed-loop E05 by itself.

## Frozen comparison

The nominal input is the released AEGIS `env.step` action at step 186. The
robot bounds are the accepted seven L5--L7 slabs. Clearance uses the exact
15-box MuJoCo moka-pot geometry and is kept distinct from raw contact and
obstacle-motion verification.

Two full-bound proposals are evaluated over normalized XYZ in `[-1,1]^3`:

1. a seven-row QP using central finite-difference next-step clearance rows;
2. a 9-by-9-by-9 exact cloned-simulator grid, plus nominal and reverse-nominal,
   used only as a discrete physical-recovery oracle.

The QP retains the objective

\[
\min_u \lVert u-u_{\mathrm{AEGIS}}\rVert_2^2
\]

subject to all seven predicted next-step L5--L7 margins being nonnegative.
Every QP and oracle proposal is then checked over every internal OSC/MuJoCo
substep. Safety requires all seven exact proxy gaps nonnegative, zero raw
protected-link contact, and at most 0.1 mm obstacle displacement.

## Decision

If no exact-safe full-bound candidate exists, one-step recovery lacks physical
authority and the next method must predict/correct a multi-step action chunk.

If an exact-safe candidate exists and its executed transition exactly matches
the clone, larger-region boundary collection is authorized. Whether the
finite-difference QP itself passes is reported separately. Closed-loop E05
remains blocked until a learned model is trained and passes grouped held-out
gates over that larger recovery region.
