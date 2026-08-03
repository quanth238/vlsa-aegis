# Minimal ICRA direction: event-triggered full-body Poisson-CBF rescue

## What the simulation evidence says

The useful result and the failed results differ primarily in the controller
interface, not in the Poisson field.

| Experiment | Safety decision variable | Outcome | Lesson |
|---|---|---|---|
| Full-episode suffix, jobs 34120/34139 | The bounded seven-joint velocity actually sent to `JOINT_VELOCITY` | Link-5 contact removed; 0.5948 m post-correction motion; full task success | The link-aware CBF can work when it directly controls its modeled variable. |
| Joint velocity from action 0, job 34462 | Direct joint velocity, but for the whole task | Baseline lost the original task/contact behavior; treatment stopped before useful correction | Replacing native OSC too early changes the task controller instead of testing a safety layer. |
| Native OSC reference governor, job 34557 | A predicted joint velocity derived from an OSC pose reference | Predicted reversal was not realized; the run stopped and the task failed | A kinematic CBF guarantee does not transfer through an inaccurate OSC pose-to-velocity surrogate. |

The successful video is therefore valid but narrow. It demonstrates a
controller-feasibility result after a manually chosen pre-contact branch. It
does not yet provide a general online integration.

## Minimal method

Keep the released native `OSC_POSE` trajectory unchanged while it is safe.
Before each 20 Hz action, construct the bounded first joint-velocity command
that would be executed by the direct adapter. Evaluate the fixed union of
`robot0_link5` and `robot0_link6` surface samples. Trigger once, at the first
state where:

\[
\min_i\left(\nabla h_i J_i\dot q_{\mathrm{nom}}+\alpha h_i\right)
\le -5\times10^{-7}
\]

and the solved hard-QP command differs materially from nominal. After this
state-derived trigger, switch to the controller already shown to work:

\[
\dot q^*=\arg\min_{\dot q}\frac12\lVert\dot q-\dot q_{\mathrm{nom}}\rVert_W^2,
\qquad
\nabla h_iJ_i\dot q+\alpha h_i\ge0.
\]

The trigger never receives a case ID, historical collision time, or target
link label. Both registered cases protect link 5 and link 6 together.

## Direct experiment

For e05, replay one exact native-OSC prefix until the online trigger. Clone
that complete MuJoCo state into two fresh 100 Hz joint-velocity arms:

1. adapter only;
2. the same adapter plus the link-5/link-6 Poisson-CBF QP.

Both arms receive the identical remaining recorded action suffix. This frozen
suffix is intentional: the first question is whether the controller-level
correction causes collision avoidance and task preservation. It avoids mixing
that question with policy recovery after trajectory divergence.

Only `SAFE_TASK_SUCCESS_USEFUL_CORRECTION` is positive. It requires:

- adapter-only reproduces the selected-obstacle link-5/link-6 contact;
- correction occurs before that contact;
- treatment has no selected-obstacle robot contact and no shifted link-5/link-6 contact;
- treatment completes the full horizon and passes the paper CAR threshold;
- commands, measured joints, and end effector continue moving after correction;
- native BDDL task success occurs after correction and remains true terminally.

Stopping, stalling, QP refusal, shifted contact, CAR failure, or task failure
is a method failure. If the state-derived trigger never appears, the result is
`NO_ACTIONABLE_DIRECT_QDOT_WARNING`, not evidence against Poisson-CBF.

E42 is a held-out replication with the same method and parameters. It runs
only after e05 is independently validated as positive.

## Publication claim if both cases pass

The defensible claim is:

> A fixed-link, event-triggered 3D Poisson-CBF joint-velocity shield can
> complement an end-effector-only AEGIS filter by preventing verified
> arm-link collisions while preserving useful motion and task completion in
> targeted SafeLIBERO simulator cases.

This is not yet a population-safety, learned-perception, moving-obstacle,
real-time, tracking-certified invariance, or closed-loop VLA claim. Those are
follow-up experiments, not prerequisites for this first ICRA feasibility
result.

## Next experiment order

1. Run and independently validate e05.
2. If e05 is positive, run e42 unchanged.
3. If both are positive, replace the frozen suffix with per-arm live policy
   feedback while keeping the same trigger, protected set, and failure rules.
4. Only then expand to a preregistered collision-case population and learned
   perception.

This order tests the central research hypothesis with the least method surface
area and keeps the accepted video's causal mechanism intact.
