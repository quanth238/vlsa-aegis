# 0059 — Interpret the sealed AF-00A population and test a reference lift

Status: accepted as a retrospective diagnostic decision on 2026-07-16.  This
decision does not reopen AF-00A, authorize an H100 submission, execute a
generated action in the simulator, or authorize probe/residual-field training.

## Question and evidence boundary

ADR-0058 left two explanations for the one-case `frozen_cem_negative`
outcome: the fixed diagonal CEM may have been inadequate, or the privileged
target may require more control authority than the registered five-step
same-budget residual family provides.  The complete sealed AF-00A population
permits a post-hoc diagnostic of those explanations without another model
call.

This analysis is bound to immutable run
`r05a-actual-forward-cem-canary-20260716c`, case
`crfs-1069f29a8d76463a`, and these source artifacts:

- tensor archive SHA-256
  `ada53973653ba7c21dab33cdbc4b4498c7b2b1840f2fc284bd3d3c76baf1d383`;
- raw payload SHA-256
  `00433437470ca7678a236c4778d5cf68160c8d3299f128b64985b1aeb142d5d1`;
- query-ledger SHA-256
  `619f15ad46390d6e55e74365dc2530e58b383592ec998842d4bbd898c7266b92`;
- published result SHA-256
  `507f25bc381b7bfeaa30abe3a7df970681f3b175f39221034c6e768e72dae914`;
- CPU republication receipt SHA-256
  `5462f875ffb41a84a06c4df715366776af3811f3229385d05be8f5b0dafcb632`.

The checked-in summary is
`evidence/r05a/af00a-sealed-population-diagnostic.json`.  It binds diagnostic
module SHA-256
`a50161a98855f931169accdd99b9753ef4a79562ebeddf30a3b54cf3e881a868`, CLI
SHA-256 `fc8802d044f6721d47bcb14a9f0b64b49d4c6875fc6513024c1e7d50e13fb367`,
focused-test SHA-256
`95a996967c828c94f27509e3be99f6dd314a46a6a3e6c9f52a1f71b4f8dc2efc`,
and deterministic full-report SHA-256
`b1c6265441aa4c9ea0b364e1bf792a11cfdaf57744a07be532174c7414aedc19`.

The analysis is one-case, retrospective, and adaptive-sample evidence.  It is
not an infeasibility certificate, a new teacher evaluation, or simulator
evidence.

## Exact sampler-trace decomposition

Let `D` be the first-five-XYZ physical target displacement from the paired
zero sampler.  For one executed compact schedule, let `U` be the sum of its
five integrated residual increments after applying checkpoint scale only,
let `R` be the paired terminal action displacement, and let `F = R - U`.
The recorded Euler traces independently reconstruct `F` as the integrated
change in the frozen model's base velocity along the controlled trajectory.
For any displacement `Q`, define target-direction completion

```
alpha(Q) = <Q, D> / ||D||^2.
```

The target norm is `3.1457032629272663` in the recorded physical XYZ action
coordinates.  The unchanged XYZ-RMS gate implies the necessary condition

```
alpha(R) >= 1 - sqrt(15) * 0.005 / ||D|| = 0.9938440103492099.
```

For equal-split Arm A, the direct integrated residual supplies essentially
the entire requested displacement (`alpha(U) = 1.0000000087928391`), but the
base-field response is adverse (`alpha(F) = -0.4001357184428276`).  The
terminal retains only `alpha(R) = 0.5998642903500115`; the physical-XYZ cosine
between `F` and `U` is `-0.9035260657943069`.  The largest physical-XYZ L2
closure error between the trace-derived decomposition and terminal response
is `3.18512e-7` over the 520 CEM evaluations.

The same mechanism appears throughout the observed population:

- base-field response completion ranges from `-0.4093303604754565` to
  `-0.24993643490803535` and is adverse in all 520 evaluations;
- terminal completion ranges from `0.2647293020583786` to
  `0.6013971631126284`, far below the necessary `0.9938440103492099`;
- selected query 464 has direct, feedback, and terminal completion
  `0.8795699854255667`, `-0.28116321931770044`, and
  `0.5984067661078663`, respectively.

This directly rejects, on this case, the premise that merely dividing
`Delta A*` across the remaining Euler steps preserves that terminal change.
The direct residual is correct, but moving the latent changes the frozen
velocity field, which cancels about 40 percent of the desired component.  It
does not reject a state- and time-dependent residual that compensates for that
response.

## The fixed CEM was also inefficient

The CEM evidence does not support treating its selected miss as the best
schedule in the registered family.  Query zero was the Arm-A center with
objective `2218.1335502517986`.  The non-elitist mean update moved away from
that strong center: the generation-1 center objective was `4157.288345`, and
centers remained worse than Arm A through generation 7.  No sampled query
beat Arm A until generation 7, only three of 520 queries beat it, and the best
query was first observed in that final generation.  The final search scale
was not fully collapsed: its median coordinate sigma was about `0.1615` of
the initial sigma and only about `2.7%` of coordinates were at the floor.

Therefore AF-00A contains simultaneous evidence of a weak fixed optimizer and
weak observed same-budget authority.  “CEM failed” cannot be promoted to
“no schedule exists.”

## Exploratory empirical response model

A zero-anchored ordinary least-squares map from the 75 executed schedule
coordinates to the 35 gate-scaled first-five action coordinates is used only
as a diagnostic.  The centered sampled input matrix has full numerical rank,
but input coverage is not a controllability certificate.  Generation-held-out
prediction remains useful for coarse response trends but is not accurate
enough to certify any of the four gates.

Every same-budget product-ball optimum of the registered full and held-out
empirical models saturates all five per-step caps and remains far outside the
gates.  The full zero-anchored fit predicts objective `1291.5807101771848`
with XYZ max/RMS approximately `0.643049` / `0.274451`, versus registered
limits `0.010` / `0.005`.  This unevaluated surrogate point is not a teacher,
a lower bound, or proof of infeasibility.  It only shows that a better
optimizer could plausibly improve on CEM while still leaving a much larger
control-authority problem.

## Root-problem interpretation

The exact root of global reachability remains unproved, but the present
failure is localized enough to change the next experiment:

1. The paper's constant residual `-Delta A*/t_s` is not a valid inverse-flow
   conversion on the canary.  It ignores the frozen base field's response to
   the modified trajectory.
2. The hard choice `B = ||Delta A*||` is an additive-displacement budget, not
   a demonstrated bound on the field effort required by the controlled ODE.
3. The diagonal CEM was too sample-inefficient to isolate that authority
   question cleanly.

The research direction remains viable only in its refined form: construct a
privileged state- and time-dependent teacher field first, measure the field
authority it actually requires, and train a student only after that teacher
has passed action fidelity and a separately authorized simulator test.

## Decision and next canary

1. Do not rerun, enlarge, tune, or reinterpret AF-00A.  Do not use the
   post-hoc two-times-budget empirical candidate as the primary method test.
2. Preregister an optimizer-free one-case reference-trajectory lift.  Using
   the exact paired zero trace `xbar`, set the desired masked state after
   active step `k` to `xbar[k+1] + alpha[k+1] * Delta`, with fixed active
   fractions `(1/5, 2/5, 3/5, 4/5, 1)`.  Reevaluate the ordinary pi0.5 base
   field at the actual controlled state and compute the integrated residual
   required to reach the next desired masked state.
3. Run a raw diagnostic lift to measure the required per-step and path
   authority, and a separate lift that projects every newly computed block to
   the unchanged `B/5` cap.  Replay each recorded schedule through the
   ordinary `residual_schedule` path.  Preserve the same case, observation,
   instruction, noise, checkpoint, target, active steps, mask, objective, and
   four fidelity gates.
4. A passing projected lift is a same-budget actual-forward witness and
   identifies AF-00A primarily as an optimizer/parameterization failure.  A
   passing raw lift that exceeds the registered authority while the projected
   lift fails identifies a canonical budget bottleneck, not global
   infeasibility.  An XYZ-only raw pass identifies outside-mask coupling as a
   separate issue.  Any result remains a one-case teacher-mechanism result.
5. Implement the new mode as strictly opt-in.  The baseline sampler and
   ordinary `residual_schedule` behavior must retain regression coverage.
   The config must remain fail closed until a clean implementation, local
   allocation-backed tests, and independent review support a separate exact
   release.
6. Continue to forbid simulator execution of generated actions, IFT-01,
   population scaling, label collection, probe training, residual-field MLP
   training, and safety/progress/generalization/latency/novelty claims.

The next experiment is therefore a direct test of the user's proposed
action-delta-to-vector-field conversion, not another search around the failed
constant-field baseline.
