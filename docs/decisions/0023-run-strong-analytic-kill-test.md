# 0023 — Run the strong analytic kill test before probe training

Status: accepted on 2026-07-15 before implementing the new arm or observing any
of its policy actions, simulator outcomes, or latency measurements.

## Context

R03 established partial privileged steerability on the frozen development
population: the distributed direct-witness direction passed Safe-Progress
Success in 9/17 eligible collision groups, while frozen, equal-L2 random, the
registered one-midpoint analytic direction, and the one-shot bridge passed
0/17.  The registered analytic result is not a strong learned-necessity
baseline.  It selects one minimum-clearance sample once, uses one local OBB
normal, and holds that direction fixed while the remaining nonlinear flow
continues.

Current primary literature makes the missing comparison decisive.  OmniGuide
evaluates a differentiable collision energy on an approximate-clean Cartesian
trajectory and recomputes its gradient during denoising.  Neuro-symbolic
constrained flow matching solves a trajectory-level safety correction inside
the pi0.5 sampling loop on SafeLIBERO.  AEGIS reports a sub-millisecond
post-generation QP.  Therefore neither denoising-time collision guidance nor
avoidance of a supposedly expensive online optimizer is available as an ECG
novelty claim without a faithful analytic comparison and measured latency.

The learned proposal and the analytic field receive the same oracle active-OBB
geometry and use the same controlled EEF sphere.  In this setting the analytic
field has no learned approximation error.  A scalar probe is scientifically
useful only if exact-continuation learning captures flow/simulator effects that
this stronger field misses, or if it later establishes a measured
safety--latency advantage.

## Decision

1. Add one active baseline gate, R03A, before R04.  Do not train a probe while
   R03A is active.  The existing 17 witness-confirmed R03 groups are reused only
   for this development-population necessity diagnostic; they can never become
   learned-probe training, calibration, or final-test groups.
2. Preserve the existing frozen, random, registered analytic, privileged, and
   direct actions and outcomes.  Add two opt-in, paired analytic trajectory
   field arms:
   - `analytic_trajectory_mid`: guidance begins at sampler step 5;
   - `analytic_trajectory_early`: guidance begins at sampler step 1 after the
     first ordinary Euler update.
   Both recompute their direction at every remaining Euler step.  No outcome is
   used to choose between them; both are mandatory reports.
3. At each active step compute the stopped-gradient approximate-clean action
   `A_hat = x_t - t * stopgrad(v_base)`.  Convert its absolute first-five XYZ
   commands through the complete frozen output affine transform.  Under the
   frozen H04 response, sample 26 equally spaced controlled EEF-sphere-center
   points per segment and evaluate every active-obstacle OBB.
4. Freeze the analytic energy to

   ```text
   E = sum softplus((m - d_i) / tau)^2,
   m = 0.005 m, tau = 0.005 m,
   ```

   where each `d_i` is controlled sphere-to-OBB clearance.  This signed-margin
   extension remains differentiable in penetration, unlike `-log(d)`.  The
   gradient is taken with the identity approximate-clean Jacobian and is zeroed
   outside the first-five translation coordinates.  Normalize its controlled
   L2 norm independently at every active step.  If the hard minimum predicted
   clearance is already at least `m`, apply no correction at that step.
5. Match the integrated model-space path budget, not a nominal coefficient.
   For each case let `B` be the immutable R03 privileged direction's first-five
   translation L2 norm.  Add the normalized energy gradient to the reverse-time
   velocity with magnitude `B / T_active`, where `T_active` is the remaining
   registered integration time.  Since `dt < 0`, this moves the final action
   along negative energy gradient.  Do not clip.  An executed first-five XYZ
   command exactly equal to either registered inclusive action bound is a
   `saturation_failure`, using exact float equality with no epsilon.  A command
   strictly outside either bound remains a `bounds_failure`; it is never
   clipped.  Invalid, nonfinite, zero-gradient, budget, or terminal arms are
   also explicit failures.
6. Pair simulator state, observation, instruction, policy noise, saved branch,
   geometry, and five-action horizon exactly.  Reconstruct every final action
   and run the ordinary branch-inclusive `D_sim` rollout.  Keep `D_opt` only as
   the analytic field and margin-stop model.  Because this gate is restricted
   to a pregrasp reach, each rollout retains the additive monotone Boolean
   `task_success_during_prefix` (the OR of `done`/`check_success` after every
   executed action), while the baseline final `task_success` field remains
   unchanged.  Any `task_success_during_prefix=true` is a `terminal_failure`,
   even if its final success, safety, or progress measurements would otherwise
   pass.
   Co-occurring evaluated failures use one deterministic precedence:
   `bounds_failure` > `saturation_failure` > `budget_failure` >
   `terminal_failure` > `zero_gradient_failure` > the ordinary Safe-Progress
   gate.  Thus any `task_success_during_prefix=true` is terminal unless a
   higher-priority pre-rollout failure prevented rollout, and a zero gradient is explicit
   unless terminal or a higher-priority failure is already present.
7. Report per arm: Safe-Progress Success, `D_sim`, contact, reach progress,
   target/obstacle motion, bounds and zero-gradient failures, integrated field
   norm, realized final first-five correction L2/RMS, and warmed batch-one
   policy p50/p95 latency.  Also report analytic-only gradient time.  Training,
   direct-search, and any scalar-search cost remain separate.
8. Interpret the result as a kill test, not a new efficacy claim:
   - if either analytic trajectory arm reaches or exceeds the privileged 9/17
     SPS count, a pure learned-clearance-probe necessity claim is rejected;
     pivot to learned model-mismatch/perception residuals or use the analytic
     field directly;
   - if it remains below 9/17, it becomes the mandatory strong direction
     baseline that a learned probe must beat on untouched groups;
   - regardless of this diagnostic, AEGIS and the closest reproducible
     constrained-flow system remain mandatory for later unconditioned episode
     and safety--latency comparisons.

## Consequences

This gate directly tests the largest current threat to the research direction
before spending allocation time on probe-label collection or training.  It
does not need a new source population because it makes no learned or transport
claim.  A passed R03A implementation/result does not unblock confirmatory R04
by itself: R04 still requires untouched official groups, renewed transfer
evidence, frozen splits, and an independent final test.

The scalar clearance probe remains plausible, but its defensible contribution
is narrow: exact deterministic-continuation labels, conservative false-safe
calibration, and a causal safe-progress advantage over this analytic field.
Gradient-guided VLA sampling, a lightweight predictor, plug-and-play collision
guidance, and low online cost are not novel by themselves.
