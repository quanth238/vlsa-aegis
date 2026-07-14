# 0017 — Freeze the R04A real-continuation label contract

Status: accepted on 2026-07-14, before any R04 label-contract outcome, probe
training, perturbation study, or learned guidance run.

## Context

R03 authorizes a bounded learned-probe study, but it does not validate the
paper draft's probe labels.  The draft attached the clearance of a source
action to an approximate-clean feature representing a different action.  Such
a pair is not a supervised example of continuation clearance and cannot show
that a learned gradient causes a safe, progress-preserving continuation.

The passed R00 manifest contains 120 rows but only 30 simulator state groups;
its four policy-noise rows per group are not 120 independent states.  The
smallest recorded R00 clearance is `0.010326895138387013 m`, so exactly zero
R00 rows lie in the inclusive `[0, 0.010] m` band around the registered
`0.005 m` decision margin.  R00 is therefore useful for checking the label
apparatus, not for training, threshold calibration, held-out testing, or a
boundary-accuracy claim.  The R02 population is already claim-bearing evidence
and must not be recycled into R04 model selection or evaluation.

The current active-obstacle geometry has a variable number of oriented boxes:
observed counts include 1, 4, 11, 15, and 21.  Any future fixed-shape geometry
input must preserve that variability rather than silently truncating or
changing obstacle order.

## Decision

R04 remains a five-action **pregrasp reach** study.  R04A is an allocation-backed
label-contract smoke, not a learning or efficacy gate.  It uses one immutable
R00 manifest row only to exercise the following contract:

1. Reset the exact saved simulator branch and keep the observation, policy
   noise, checkpoint, and ten-action horizon fixed.
2. Issue eager trace-only sampler calls at registered steps 1 through 5,
   corresponding to `t = 0.9, 0.8, 0.7, 0.6, 0.5`.  Repeat every request twice.
   Every call uses `intervention_mode=none`; correction, training, and guidance
   are disabled.
3. Treat the normalized-model `predicted_clean` tensor as the primary trace
   feature.  Retain its fully inverse-transformed physical counterpart only as
   an audit field.  A displacement, if a later registered gate introduces one,
   is converted with normalization scale only and never by subtracting a mean.
4. Every trace request reruns the complete frozen eager trajectory from the
   same noise; R04A does not separately resume from a saved latent.  The
   returned downstream segment is the deterministic continuation along that
   one unedited trajectory.  The current policy boundary exposes normalized
   `10 x 32` intermediate trace tensors but only the inverse-transformed
   physical `10 x 7` final action.  All ten complete calls must return the
   exact same physical final action.  Any mismatch is an apparatus failure,
   not an observation to average; R04A does not add a normalized-final-action
   baseline seam.
5. Execute that exact physical first-five-action prefix twice from the exact
   branch.  Retain both raw rollouts.  The scalar continuation-clearance label
   is the minimum controlled `D_sim` across the two repeats; every trace record
   is content-bound to the final-action hash, both rollout hashes, and the
   resulting label hash.  Clearance, reach progress, forbidden contacts,
   target motion, active-obstacle motion, infeasible outcomes, and all failures
   are retained.

`D_sim` is the inclusive branch-plus-hidden-substep minimum signed distance
between the controlled 6 cm EEF sphere and the active obstacle's OBB union.
The action and geometry frame is world.  This is distinct from optimizer
clearance `D_opt`; R04A runs no optimizer and records `D_opt` as not applicable.
The target is the frozen branch position of
`akita_black_bowl_1`, and reach progress uses the passed R00 threshold
`p_min = 0.029897349105658888 m`.  Neither progress nor clearance is used to
drop a label-contract row.

For a fixed-shape geometry audit, sort active-obstacle geoms by exact geom name,
encode at most 21 world-frame OBBs, pad unused rows deterministically with
zeros, and carry an explicit Boolean validity mask.  Truncation is forbidden;
more than 21 active OBBs is a terminal contract failure.

The apparatus is content-bound to:

- R00 manifest SHA-256
  `3180836991e64b774670218f5fe4edaa10775fbb33fcdb330cc781b8b8937dad`;
- R00 summary SHA-256
  `90fe09e075b30e680e5cfbd81c802afa03c94b58ba7f0e95e66080c61af2096f`;
- R03 summary SHA-256
  `dea3e66c854faa0b659777bfed4b651b19d8d4d0710696ff7bfe52c44d4ee76e`;
- sampler-parity SHA-256
  `26083b0a71ec41cf67e0978b815a77bfe71e4b4a4a08a8ddcdaece608de9818a`;
- converted checkpoint SHA-256
  `988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed`.

The existing R00 rows have the fixed role `apparatus_only`.  They may never be
promoted into an R04 training, calibration, validation, test, or claim
population, even if the smoke passes.

## Claim-bearing R04 requirements

Before training, create genuinely new immutable state groups.  Different
policy seeds at one simulator branch remain one state group.  If an
intermediate pregrasp state is constructed by replay, its immutable source
initial state and complete action history must be recorded and replayed for
every paired branch; a partial simulator-state assignment is not an equivalent
reset.

Freeze the complete group-preserving train/calibration/validation/test split
before generating continuation-clearance labels.  The split and bootstrap unit
is the immutable **source initial episode/state**, not each replay-derived
intermediate branch.  Every derived branch, complete replay history, trace,
policy noise, perturbation, and repeated rollout descending from one source
episode/state remains in the same split and bootstrap group.  The R02 claim
cases are excluded from every R04 split.  A claim-bearing collection needs at
least 200 distinct training state hashes and registered boundary coverage
around 5 mm.  If a frozen split lacks enough unsafe or boundary groups for its
preregistered metrics, the gate is underpowered and stops; examples may not be
moved across splits after labels are inspected.

The later prediction gate must add one-sided, groupwise false-safe control at
the 5 mm margin; MAE and rank correlation alone are insufficient.  The later
gradient-causality gate must retain joint safe-progress success and pair all
arms by state, observation, noise, and executed horizon.  It must include
frozen, learned-gradient, equal-realized-norm random, registered analytic,
margin-stopped/time-structured analytic, privileged oracle-direction, and
direct-planner reference arms.

## Deferred perturbation gate

R04A does not perturb or resume a latent.  Any later local perturbation study
must first add and regression-test an exact resume/edit path, record the
**post-edit** latent or post-edit approximate-clean feature, deterministically
continue that edited state, and label the action that was actually executed.
The existing one-shot bridge diagnostic exposes a pre-edit trace and is
invalid as an edited-feature label source.  It cannot be reused for that gate.

## Consequences

- A passing R04A artifact proves only that the label plumbing is internally
  paired and reproducible on one reused state; it provides no learning,
  prediction, gradient, efficacy, novelty, or transport-phase evidence.
- No R04 probe may be trained from the R00 smoke or R02 claim cases.
- No correction or guidance mode is authorized by this decision.
- A post-grasp transport claim still requires separately frozen states,
  phase-specific progress calibration, and the same oracle ladder.
- Full-arm geometry, dynamic-scene safety, and task success remain outside the
  controlled R04A metric.
