# π0.5-DROID joint-velocity L5–L7 pilot preregistration

This is a new opt-in pilot on primary case `vlsa-t1-goal-ii-t0-e05`. It does
not modify or reinterpret Table 1. The purpose is to test whether removing the
end-effector resolved-rate conversion makes the existing three link barriers
effective when the policy, QP, and simulator controller all use seven joint
velocities.

The two arms are π0.5-DROID joint-velocity baseline and the same live policy
with the L5/L6/L7 QP active. Both use the manifest initial state and environment
seed, the same per-query flow-noise seeds, 15 Hz control, 15 model actions per
chunk, eight executed actions per query, and exactly 225 executed actions
unless an explicit method failure prevents execution. Native success is
recorded when it first occurs, but both valid arms continue to the common
15-second horizon.

The policy observation adapter uses processed SafeLIBERO agent and wrist RGB,
the seven Panda joint positions, and a gripper closed fraction computed as
`clip(1 - mean(abs(finger_qpos)) / 0.04, 0, 1)`. The model's first seven
outputs are physical joint velocities clipped to `[-1, 1] rad/s`. Its gripper
output is thresholded at 0.5 and converted to SafeLIBERO `-1` open / `+1`
close. A custom `JOINT_VELOCITY` controller makes its normalized input and
physical output ranges both `[-1, 1]`, avoiding an implicit scale conversion.

The active QP minimizes deviation from the π0.5-DROID joint velocity under the
same end-effector-preserving metric used by the accepted shadow experiment.
It enforces exactly three continuous CBF inequalities, one for each certified
L5/L6/L7 MVEE, joint-velocity limits, and one-step joint-position limits with
a 0.02 rad margin. It uses `alpha=10 s^-1` and `D_opt=0.01 m`. Gripper output
is never changed. An infeasible or invalid QP is retained as method failure;
there is no fail-open action.

The obstacle is the immutable released-AEGIS perception MVEE in the archived
primary result. Raw MuJoCo contact distance and obstacle displacement remain
the outcome authority and distinct from `D_opt`.

Interpretation is gated on baseline competence. If π0.5-DROID does not achieve
the native SafeLIBERO goal, the run establishes only action/controller/QP
compatibility and cannot support a safety-efficacy claim. If baseline is
competent, the active arm passes only if it has no L5–L7 contact, passes paper
CAR, and reaches the native goal. No parameter is tuned after observing the
pair. KKT/VI learning remains out of scope until an oracle joint-space arm
passes.
