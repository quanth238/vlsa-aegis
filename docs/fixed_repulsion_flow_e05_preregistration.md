# Fixed physical repulsion inside π0.5 flow

This paired E05 mechanism test applies the analytical L5--L7 repulsive
direction earlier than the collision boundary and inside π0.5 generation.
It does not use an MLP or QP.

The state is replayed exactly to action 180. Because π0.5 executes five
actions per query, actions 182--184 correspond to chunk slots 2--4 from the
query at step 180. Paired cloned-OSC rollouts identify a seven-row clearance
Jacobian for this five-action prefix. Fixed soft-min weights produce one
physical XYZ push-away direction.

Three arms share the same state, observation, RNG seed and horizon:

1. ordinary π0.5 denoising;
2. the fixed direction added post hoc to slots 2--4;
3. the fixed direction added after Euler updates 5--9 inside π0.5 denoising.

The late live query is not claimed byte-equivalent to the archived Table-1
query. The `0.005` split-JIT tolerance was calibrated only at the initial
query and is not extrapolated across 36 replans. Its discrepancy is retained
as a diagnostic. Scientific pairing is direct between the ordinary and
guided live calls at the exact replayed step-180 state, using the same
observation, RNG seed, and horizon; the post-hoc arm derives from that same
ordinary live chunk.

The per-Euler physical step is 0.05 action units, giving the same nominal
0.25 budget as the post-hoc arm. Rotation and gripper remain those of the
ordinary policy; deployment still uses the released translational protocol.

The inside-flow arm passes only if its exact five-action L5--L7 minimum is
nonnegative and strictly exceeds both comparators. Only then may its five
actions execute. Execution must exactly reproduce the accepted clone, have
no L5--L7 raw contact or CAR, and retain at least half the ordinary prefix's
end-effector progress. This is a local mechanism test, not task completion or
population generalization.
