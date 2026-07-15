# 0034 — Recompute R05A status from exact artifact traces

Status: accepted on 2026-07-15 before the ADR-0032 CPU allocation, retry C,
or IFT-01.

## Context

Independent review of the exact retry-B payload found that several stored
summary booleans could remain self-consistent after coupled tampering of their
underlying trace metadata. In particular, a teacher or zero-schedule replay
could change `dt`, `intervention_step`, or `num_steps` together with its stored
check map. The converged canonical-replay branch also used intentionally
different top-level and policy-call shapes: only the top-level record contains
`applicable`. A validator that compared those two objects directly would
reject every valid converged artifact, while a validator that trusted only the
stored checks would not independently establish the vector-field replay.

These are artifact trust-path defects. They do not provide evidence about the
teacher solver or justify changing any scientific setting.

## Decision

1. Recompute every registered teacher invariant from the serialized raw trace.
   A converged teacher requires every invariant to be exactly true. A finite
   nonconverged teacher requires exactly `control_valid`, `schedule_applied`,
   and `fidelity_passed` to be false and every other registered invariant true.
2. Recompute zero-schedule and canonical-replay source, timing, step count,
   initial noise, budget, recurrence, schedule, physical-action, and stored
   check-map bindings from the raw trace. Coupled metadata/check-map tampering
   fails closed.
3. For a converged canonical replay, require the exact registered record shape,
   the teacher schedule, returned teacher actions, teacher final state, teacher
   internal replay final state, independently recomputed diagnostics, and a
   true independently recomputed pass result. Compare its policy-call record
   to the top-level record after removing only the top-level `applicable` key.
4. For finite nonconvergence, accept only the exact sentinel
   `{"applicable": false, "passed": false, "reason":
   "finite_teacher_search_did_not_converge"}` in both locations.
5. Add dependency-backed regressions for singleton artifact scalars, coupled
   teacher/replay metadata tampering, canonical teacher bindings, converged
   top-level versus policy-call shape, and the exact nonconvergence sentinel.
6. Include these trust-path regressions in the still-unused ADR-0032 CPU run
   `r05a-adr0031-apparatus-cpu-20260715a`. Retain that preregistered ID and all
   exact resources and test counts.
7. Do not change the source case, checkpoint, noise, target, mask, budget,
   solver updates, learning rate, tolerance, action horizon, simulator
   protocol, or any future retry-C resource.

## Consequences

The CPU apparatus run can establish that the independent result decision is
derived from exact trace evidence rather than trusted summary labels. It
cannot repair retry B, turn finite nonconvergence into infeasibility, establish
inverse-flow efficacy, authorize retry C by itself, launch IFT-01, or authorize
probe/MLP training.
