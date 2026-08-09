# Learned-gradient matched-random control

This no-training experiment tests whether job `37416` succeeded because its
learned direction is informative or merely because many directions are safe.
It reconstructs the same held-out states 187 and 190, reloads the immutable
model, and requires the recomputed nominal risks and gradients to match the
validated result.

For each state, 256 directions are sampled uniformly from the unit sphere and
conditioned only on remaining inside normalized Cartesian action bounds at the
largest registered radius 1.0. The same directions are reused at radii
0.1/0.25/0.5/1.0, exactly matching the learned correction norms. Every action
receives a fresh complete cloned OSC rollout with raw L5--L7 contact checked
after every internal MuJoCo step. No mesh distance is calculated.

For each direction, the first registered contact-free radius is recorded. At
each state the randomization value is

`(1 + number of random directions safe no later than the learned direction) /
(1 + 256)`.

The learned gradient has a local directional advantage only if it is freshly
safe and this value is at most 0.05 on both states. Next-EE error relative to
the task-successful oracle action is diagnostic only. Passing does not
establish task preservation, L6/L7 learning, closed-loop success, or
generalization.
