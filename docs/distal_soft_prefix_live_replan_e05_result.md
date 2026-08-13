# E05 one-shot repulsion followed by live VLA replanning

## Verdict

**NO-GO for one-shot episode-wide safety; PASS for task recovery.** The
registered five-action repulsive prefix was physically contact-free and the
frozen pi0.5 policy subsequently completed the task. However, after distal
intervention was released, live VLA plus original AEGIS returned L5 to the
Moka pot and later displaced the object beyond CAR.

This is the requested experiment: no endpoint preservation, no task-return
penalty inside the prefix, and no repeated L5--L7 correction.

## Primary H100 run

Producer `39354` ran from clean commit
`39a78e70b05a4ca2db7c53faa20b22b89c3c68a4` on H100 `worker-2`.

| Observation | Result |
|---|---:|
| immediate EE displacement from nominal five-action prefix | 34.690 mm |
| protected contact during actions 182--186 | 0 samples |
| CAR displacement during actions 182--186 | approximately 0 mm |
| first conservative Loewner-proxy violation | action 183 |
| first exact robot-ellipsoid/compiled-box overlap | action 190, substep 21 |
| first raw protected MuJoCo contact | action 223, substep 19 |
| protected contact samples after activation | 865 |
| maximum active-obstacle L1 displacement | 13.827 mm |
| native task completion | action 287 |
| primary safe task success | no |

The original collision appeared around action 189. The one-shot prefix delayed
raw L5 contact to action 223, so it had real avoidance authority. It did not
create a forward-invariant safe continuation: repeated task-directed VLA
commands returned the arm toward the obstacle.

The 20.926 mm value reported by the earlier oracle was its terminal deviation
after the fixed 20-action evaluation. The actual immediate EE displacement
after the five-action prefix was 34.690 mm. The frozen VLA absorbed this larger
deviation and still completed the task.

## Independent replay

H100 replay `39358` reproduced the scientific outcome:

- same registered prefix SHA-256
  `adb391fadd0f4ce9bb5998bf82eaa03451754b80926ddf6154263fb9ed46720b`;
- first protected contact again at action 223 (substep 21);
- native task completion at action 248;
- raw-contact and CAR gates failed.

The pi0.5 action hashes differed across the two matched-seed policy servers,
so the continuation was not bitwise deterministic. This is important rather
than a reason to discard the run: both stochastic continuations reproduced the
same contact onset action and the same qualitative task/safety outcome. Future
learning must condition on the realized VLA chunk or estimate risk over policy
samples.

## Safe-set implication

Do not label the post-prefix state as simply “safe” because the current state
has no contact. Safety depends on the continuation. For this work, the useful
practical object is a bounded, policy-conditioned recoverable set:

\[
\mathcal X_{\mathrm{rec}}^{H,\rho}
=
\left\{
z:\exists\,\Delta A,\;\|\Delta A\|\le\rho,
\text{ such that the }H\text{-step rollout is contact-free and }z^+
\text{ remains recoverable or reaches the goal}
\right\}.
\]

Approximate it offline by backward rollout labeling:

1. Save pre-contact warning states from complete episodes.
2. Generate bounded geometry-derived repulsive prefixes without endpoint
   cancellation.
3. Execute each prefix and then continue the frozen VLA from the changed
   observation.
4. Label the pair by future contact, CAR, and task recovery.
5. Propagate positive labels backward only when the successor is already a
   positive recoverable state or reaches the goal.

This gives the model examples of *when another push is required*, not only the
direction of one local push. The next oracle should use hysteretic warning-
triggered reapplication and execute only short prefixes. It should not force a
geometric detour or endpoint return. The exact-overlap warning preceded raw
contact by 33 actions in this run, demonstrating useful reaction time. The
perceived Loewner margin crossed zero much earlier and is therefore a warning
feature, not a calibrated safety boundary.

## Artifacts

- Primary result:
  `/mnt/data/quanth/experiments/vlsa-distal-soft-prefix-live-replan-e05/soft-prefix-live-replan-e05-20260813c/result.json`
- Primary video:
  `/mnt/data/quanth/experiments/vlsa-distal-soft-prefix-live-replan-e05/soft-prefix-live-replan-e05-20260813c/episode.mp4`
- Result file SHA-256:
  `70494008c6a873803ff9fdf9a9c3925a5d7c2d8d3586fa8124a3eadea65e34a7`
- Result payload SHA-256:
  `df0d1c731255eb0fec89f285146921f911e5f6a124ae1a06673dea12ab7e1c5c`
- Replay result:
  `/mnt/data/quanth/experiments/vlsa-distal-soft-prefix-live-replan-e05/soft-prefix-live-replan-e05-20260813d/result.json`
