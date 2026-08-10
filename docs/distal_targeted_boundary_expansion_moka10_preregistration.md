# Targeted same-task boundary expansion preregistration

## Question

Can the remaining unused complete goal-II/task-0 episodes supply enough
Cartesian-action, joint-configuration, and joint-velocity coverage for every
immutable E05/E10/E15 boundary state, without using held-out inputs or labels
to choose the training states?

This is a data-support gate. It does not train an MLP, run a learned QP, or
execute closed-loop E05.

## Immutable split and source

- Additional training episodes: E25, E35, E40, and E45.
- Immutable test episodes: E05, E10, and E15.
- Existing training/validation/test records and labels are unchanged.
- E05/E10/E15 features, margins, contacts, actions, and labels are not used to
  select a new episode or state.
- E25/E35 are successful complete archived runs. E40/E45 are complete
  300-action archived runs that did not complete the task; they remain valid
  state-coverage evidence and are not represented as successful episodes.

All four remaining same-task episodes are included. This avoids choosing only
episodes that look favorable after comparison with the held-out test set.

## State registration

Each training episode is replayed over its complete archived action sequence.
From step 20 onward the scan records Cartesian actions, seven joint positions,
seven joint velocities, controller goal, seven exact-box current clearances,
and the frozen pair features.

Within each episode independently, select five consecutive states ending at
the smallest nonnegative current seven-row L5--L7 clearance, subject to:

- all five interval starts have nonnegative current clearance;
- the anchor clearance is at most 100 mm;
- ties use the earlier anchor step;
- no future rollout margin, contact label, or held-out episode value is used.

At each fixed state, reuse the unchanged 125-action, two-step cloned-OSC grid
and record every internal simulator substep. Fit the unchanged 27 overlapping
ridge-Huber regions. Single-affine failures are reported but do not discard a
state or its regional labels.

## Frozen decision

The expanded split contains 60 training, 10 validation, and 15 immutable test
states. The current region-aware coefficient-output MLP may be preregistered
for a separate H100 run only if:

- state support passes for 15/15 test states under the existing
  training-derived thresholds; and
- induced multi-region oracle smoothness passes for 15/15 test states.

If either gate fails, do not train and do not run closed-loop E05. If both pass,
the later learned run must still demonstrate zero false-safes, safe-action
support in every test state, valid regional QPs, and fresh exact rollout
verification. If the supported-state MLP then fails, the next architecture is
an action-conditioned conservative safety-value model.

## Identity

- Config SHA-256:
  `9c90aa1047cff37efa2091657a9c0c1d0d7e26719c29ac331602e76045d3813e`.
- Selected-manifest SHA-256:
  `32a88c6e973031bb553c056abb244c39a8f5952f411e3d7b16c2a9c5b3717c3d`.
- Base expanded-dataset payload SHA-256:
  `fc3f38535ec562d091f07c113350a98f15953d7987903fa811baf4bc98623737`.
- Base expanded-oracle payload SHA-256:
  `c7fc7c8ae6658aea387fa162e9925ce8c4b462523bc24f92f58dd88c394ce7cf`.
