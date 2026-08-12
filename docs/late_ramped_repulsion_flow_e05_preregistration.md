# Late-ramped fixed-repulsion timing gate

This opt-in experiment tests whether moving fixed physical repulsion closer to
the end of pi0.5 denoising reduces cancellation by later Euler updates. It
reuses the paired live step-180 query from the earlier fixed-flow experiment.
That live prefix was already L5--L7 proxy-safe and differs from the archived
dangerous step-185 policy mode, so this is a timing mechanism test only.

All arms share the same simulator state, observation, pi0.5 RNG seed, frozen
policy, analytical seven-row soft-min direction, guided chunk slots 2--4, and
five executed-action rollout. Each guided slot receives total injected physical
XYZ budget 0.25 under one of three schedules:

- uniform final five: 0.05 at Euler updates 5--9;
- late linear last two: 1/12 at update 8 and 1/6 at update 9;
- final step only: 0.25 at update 9.

For each flow arm, the surviving correction is measured after the complete
output transform as the L2 norm of all nine XYZ differences in slots 2--4.
Its post-hoc comparator applies the same physical direction after denoising and
is solved to the same surviving L2 norm, including action clipping. Equal
injected budget is reported but is not used as the fairness comparison.

The timing hypothesis passes only if both late schedules retain more final
correction norm than the uniform-final-five schedule. Exact cloned-OSC
L5--L7 clearance, progress, direction alignment, inference time, and matched
post-hoc clearance are reported. No arm executes on the main simulator. This
experiment cannot establish collision prevention, task completion, population
generalization, or a learned steering method.
