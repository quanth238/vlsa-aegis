# Adaptive native-distance diagnostic preregistration

The validated 1 m `mj_geomDistance` target is rejected because MuJoCo 3.2.3
warns that large positive cutoffs can be inaccurate for general convex pairs,
and the experiment observed negative query values without raw contact.

This follow-up tests the smallest contact-consistent alternative on the 15
critical E05/E10/E15 states. It trains no model, solves no QP, and runs no
closed loop.

For every registered physical geom pair and every cloned-OSC substep:

1. Query `mj_geomDistance` with `distmax=0` to classify overlap.
2. If not overlapping, query the fixed cutoffs 1, 2, 4, 8, 16, 32, and 64 mm.
3. Use the first non-censored nonnegative result as the local clearance.
4. Treat a pair still censored at 64 mm only as a safe lower bound, never as an
   exact regression label.

The gate requires zero negative-zero-cutoff pairs without a registered raw
contact, zero raw pair contacts missed by the zero-cutoff query, zero negatives
introduced only by positive cutoffs, consistent repeated non-censored values,
at least one E05 raw-contact witness, and 15/15 initially safe test states.

A pass authorizes only preregistration of full grouped physical target
collection. A failure rejects `mj_geomDistance` for this population and directs
either a contact-only formulation or an independently validated geometry
engine. Learning and closed-loop E05 remain blocked.

Config SHA-256:
`fe869e9c5e16615999f59b12d00ed33c0e530cc20bc9948ee76ba60a95cf8e82`.
