# E05 learned-gradient matched-random result

## Outcome

Clean H100 job `37427` compared the immutable job-`37416` learned
negative-contact-risk gradient with 256 uniformly sampled, action-bound-feasible
directions at each held-out E05 state. Every learned and random action received
a fresh complete cloned OSC transition, and safety meant zero raw L5--L7
contact at every internal substep.

The learned gradient did **not** beat the matched random control:

| State | Learned first-safe radius | Random directions safe no later | Add-one empirical value | Gate |
| --- | ---: | ---: | ---: | --- |
| 187 | 0.25 | 56 / 256 | 0.2218 | fail |
| 190 | 0.50 | 122 / 256 | 0.4786 | fail |

The preregistered requirement was at most `0.05` on both states. Random
directions were cumulatively safe at rates `0/21.9/53.1/62.5%` at radii
`0.1/0.25/0.5/1.0` for state 187 and `0/0/47.7/69.9%` for state 190.
Therefore the two safe learned-gradient actions previously observed can be
explained by abundant local safe directions.

Task-reference error does not rescue the safety-gradient claim. At the first
safe learned actions it was `1.070 mm` (state 187) and `2.299 mm` (state 190),
while the mean among random safe actions at the same radii was `0.811 mm` and
`3.234 mm`, respectively. This mixed diagnostic was not an acceptance gate and
does not demonstrate task preservation.

The job performed 2,056 fresh cloned transitions. Their aggregate
`env.step` time was `89.696 s`; complete job wall time was `124.555 s` on an
NVIDIA H100 80 GB HBM3.

## Interpretation and next model test

This rejects the current binary-contact classifier gradient as an informative
steering signal. It does not reject execution-aware learning. The immediate
problem is that pointwise contact classification does not supervise the local
ordering or derivative that the controller uses for correction.

Before adding multi-state or L6/L7 data, the next local mechanism experiment
should change the supervision:

1. Sample paired `+d` and `-d` controller rollouts at matched radii around each
   nominal action.
2. Label which member has lower executed contact burden and the smallest radius
   that becomes contact-free.
3. Train with a pairwise directional-ranking loss, optionally together with
   the existing contact loss, so the network is explicitly penalized when its
   action gradient orders a pair incorrectly.
4. Evaluate on held-out directions and then repeat this same matched-random
   gate on held-out states.

Only a directional model that beats random should receive boundary-focused
multi-state/L6--L7 data. Closed-loop E05 remains unauthorized.

## Evidence

- Result SHA-256: `b25fc41f3fe9db39387b5dff32dbffa2a92e5cc4dc4aefd3e7b61623d249c989`
- Validation SHA-256: `6f644ddc2b48f243b2d7c4e6e2319b63cc7879c2e5207662184405814d48be48`
- Result payload SHA-256: `76d8a4f122a637fd9843c2f4c1d03f22b1856707bca49ddd1e0658e3657186a7`
- Validation payload SHA-256: `8a8bd81a582028601b2d90e9d8189eb0de3cd3e7e81b848a3902312dff8be5c9`
- Local artifacts:
  `/Users/quanth238/personal/Research/probe_vla/output/vlsa_distal_gradient_random_control_e05/gradient-random-e05-20260810b/`

