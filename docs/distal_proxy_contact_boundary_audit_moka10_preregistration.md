# Distal ellipsoid/contact boundary audit preregistration

## Question

On the existing paired two-action cloned-OSC population, how often does the
seven-row L5--L7 ellipsoid proxy disagree with registered protected MuJoCo
contact, especially close to the proxy boundary?

This audit follows the native-distance NO-GO. It does not use positive
`mj_geomDistance`. The optimizer quantity `D_opt` remains the minimum of the
seven ellipsoid rollout margins; simulator authority `D_sim` is only whether a
registered protected L5--L7 contact occurred.

## Frozen population and analysis

Reuse all 10,625 fit-grid and 2,400 validation/test off-grid rollouts. No state
or action is discarded. At the zero threshold:

- false-safe means `D_opt >= 0` and protected MuJoCo contact;
- false-unsafe means `D_opt < 0` and no protected MuJoCo contact.

Report the same confusion table for fixed thresholds from -20 to +20 mm and
inside fixed absolute proxy-margin bands of 1, 2, 5, 10, and 20 mm. These are
proxy-boundary strata, not claims of physical millimetre clearance.

## Gate

The audit is complete only if the immutable 13,025-action population is
present, observed contacts exist, the 5 mm band is nonempty, and the zero
threshold has no observed false-safe. False-unsafes are reported rather than
tuned away. A complete audit authorizes only the separate initial-state audit.
It does not authorize training, a QP, simulation, or closed-loop E05.
