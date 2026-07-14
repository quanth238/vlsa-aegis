# 0016 — Accept R03 and authorize a bounded R04 study

Status: accepted on 2026-07-14 after H100 array `27405` and allocation-backed
verifier `27450`.

## Context

All 20 immutable cases produced valid final artifacts. All frozen collisions
and all 17 applicable direct witnesses were reconfirmed under paired simulator
state, observation, policy noise, and executed horizon. The feasible-conditioned
safe-progress rates were:

```text
direct witness       17/17
distributed oracle    9/17
equal-L2 random        0/17
equal-L2 analytic      0/17
one-shot bridge        0/17
frozen baseline        0/17
```

Oracle minus random was `9/17 = 0.5294118`; its fixed-seed, 10,000-replicate
grouped 95% bootstrap lower bound was `5/17 = 0.2941176`. The preregistered R03
gate passed. The analytic matched gate failed with a zero point difference and
zero lower bound. Oracle minus analytic had the same `5/17` lower bound and an
exact paired one-sided result of `b=9`, `c=0`, `p=1/512=0.001953125`.
Therefore every ADR-0012 authorization condition passed.

The original intent-to-treat rates were oracle 9/20 and direct 17/20. The three
structural negatives began inside the registered 5 mm margin and cannot be
repaired at the inclusive branch sample. Among the eight eligible oracle
failures, four missed only clearance and four missed only progress; every
direct witness passed.

## Decision

Mark R02 and R03 passing and activate R04. Interpret the evidence as follows:

1. removing exact endpoint equality resolves the physical-existence premise
   for all 17 branch-margin-valid cases;
2. under the registered distributed residual, the privileged witness direction
   causally increased safe-progress success to 9/17 versus 0/17 for the matched
   random arm; this does not mean the output exactly realized the direct action;
3. the eight residual failures occur after the physical-existence gate, under
   the registered intervention mapping. R03 does not isolate gain, timing,
   residual parameterization, or nonlinear continuation as the mechanism;
4. the registered equal-L2 one-midpoint analytic normal is insufficient because
   every arm loses progress, while every registered `t=0.5` one-shot bridge
   loses clearance. This does not rule out stronger schedules for either arm.

Authorization permits only a preregistered R04 probe-prediction and
gradient-causality study. It is not evidence that ECG is effective, that a
scalar clearance gradient can recover the planner-witness direction, or that
learned guidance is necessary. A stronger or time-structured analytic method
was not ruled out.

Before claim-bearing guidance, use new episode/state groups and pair all arms
by state, observation, policy noise, and horizon. Evaluate frozen,
learned-gradient, equal-norm random, registered analytic, a preregistered
margin-stopped analytic control, the privileged oracle-direction reference,
and the direct-planner upper reference. Keep safe-progress—not clearance
alone—as the intervention outcome. If intervention time or dose will be
calibrated on calibration-only groups, revise and freeze the draft's current
fixed-dose and Gate-2-only timing rules before collecting outcomes.

The probe target must bind to the exact action outcome represented by its
input. Prefer real sampler latents and label the frozen deterministic
continuation by direct simulator rollout; do not attach the clearance of a
different source planner action to an approximate-clean feature. In addition
to boundary MAE and rank correlation, preregister a one-sided false-safe or
conservative boundary-calibration criterion at the 5 mm trigger.

## Consequences

- No learned training may use the 17 R02 claim cases as test-independent
  training or tuning data.
- The present result is a five-step pregrasp reach pilot. The draft's post-grasp
  transport claim requires new immutable post-grasp states and a phase-specific
  progress calibration; otherwise the manuscript must be reframed.
- The controlled safety metric is a 6 cm EEF sphere against the active
  obstacle's oriented boxes. Full-arm, dynamic-scene, full-task, and real-world
  safety remain open.
- The draft currently defines minimum clearance over exact forbidden
  robot--obstacle geometry. Before R04 is frozen, either narrow that formal
  definition to the controlled EEF-sphere/active-OBB metric or run the stated
  full-geometry experiment.
- The three already-unsafe branches require a separately preregistered earlier
  intervention or recovery study.
