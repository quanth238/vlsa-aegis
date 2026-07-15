# 0028 — Retire the pending R03A population and test inverse-flow transport

Status: accepted on 2026-07-15 before implementing an inverse-flow teacher or
observing any inverse-flow policy action, simulator outcome, or latency.

## Context

R01 established a simulator-verified endpoint-free safe-progress action witness
for every one of the 17 branch-margin-valid development groups. R03 then showed
that direct witness execution passed 17/17, while the existing constant
distributed privileged residual passed only 9/17. Frozen, equal-norm random,
the registered static analytic arm, and the one-shot bridge passed 0/17.

R03A asked a different question: whether a stronger geometry field could make
a learned scalar clearance probe unnecessary. Its valid source-node smoke
`27639_0` showed that both analytic arms avoided contact on one case but lost
required reach progress. The 17-case R03A population was not launched and no
population conclusion exists.

The selected research hypothesis is now that a privileged safe-progress action
chunk must be transported through the frozen nonlinear sampler into a
time-dependent residual velocity sequence before it can supervise a deployable
student. Completing the R03A population would not test that hypothesis.

## Decision

1. Intentionally retire the pending R03A population. Preserve its code,
   contracts, ADRs, one-case artifact, and logs as historical evidence and a
   future analytic baseline. Do not report R03A as completed or passing.
2. Keep the scalar ECG feature R04 blocked and superseded for the present
   direction. Add R05A as the only active gate: frozen-sampler inverse-flow
   transport on the existing development cases. No MLP training is authorized
   while R05A is active.
3. Reuse the 17 R01/R03 groups only for teacher feasibility and mechanism
   development. They remain permanently forbidden for student training,
   calibration, validation, final testing, or a generalization claim.
4. For each case, reconstruct the fresh paired frozen output on its immutable
   R02 source host and define the target as

   ```text
   x_target = x_frozen + Delta_star_model,
   ```

   where `Delta_star_model` is the immutable R01 witness first-five XYZ minus
   the fresh paired frozen first-five XYZ, converted with normalization scale
   only. All other target coordinates remain the fresh frozen coordinates.
   Directly execute this fresh paired target twice and require Safe-Progress
   Success before interpreting the teacher.
5. The teacher may use the frozen observation, instruction, explicit policy
   noise, sampler, and paired target. It may not query simulator outcomes,
   collision geometry, a distance field, or a planner while solving controls.
   Freeze all pi0.5 weights.
6. Record a ten-step residual schedule `u_k`, one row per ordinary Euler step,
   but keep `u_0` through `u_4` exactly zero and optimize only the same five
   active steps 5--9 used by the R03 constant-residual baseline:

   ```text
   x_(k+1) = x_k + dt * (v_pi(x_k, t_k, observation) + u_k),
   dt = -0.1.
   ```

   Every `u_k` is exactly zero outside the first-five XYZ coordinates. Let
   `c_k = dt * u_k`, `B = ||Delta_star_model[:5, :3]||_2`, and
   `P = sum_k ||c_k[:5, :3]||_2`. Require `P <= B` and
   `||c_k[:5, :3]||_2 <= B/5` for every active step. These constraints match the R03
   integrated correction budget and prevent a one-step target overwrite.
   The immutable R02 direction and its reported norm are stored in float64,
   while pi0.5 samples in float32. Cast the direction elementwise to float32
   once, form the target with the ordinary float32 addition, and pass
   `float32(source_reported_norm)` to the solver as the explicit runtime
   budget. Record the norm of the cast direction and the norm of the realized
   float32 `target - frozen` difference separately. Floating-point addition can
   make those values differ by ULPs; neither may silently replace the
   source-bound budget.
7. Use a deterministic, fixed-configuration constrained direct-shooting
   solver. Its primary objective is terminal-target fidelity under the hard
   mask and path-budget constraints. Record control energy and retain the
   lowest-energy iterate among fidelity-feasible iterates, but do not describe
   that finite search as a local or global optimum. Its exact algorithm,
   initialization, iteration limits, numerical tolerances, and failure statuses
   must be content-bound in the experiment config and pass synthetic nonlinear-
   flow tests before any real submission. A finite search that finds no valid
   controls is reported as nonconverged, not as an infeasibility certificate.
8. Freeze implementation-fidelity gates before outcomes:
   first-five XYZ maximum absolute target error at most `0.010` and RMS at most
   `0.005` physical action units; first-five seven-channel maximum error at most
   `0.050` and RMS at most `0.015`. These reuse the existing ADR-0010 action
   scale as an apparatus tolerance and are not a semantic safety certificate.
   Simulator Safe-Progress Success remains the scientific outcome.
9. Before any efficacy smoke, run one allocation integration canary on
   `crfs-1069f29a8d76463a`, pinned to its source host `worker-1`. It may restore
   the paired source observation and invoke frozen pi0.5, but it must not
   execute a teacher-generated action in the simulator. Use one H100, eight
   CPUs, 64 GiB host RAM, sequential concurrency, and record actual host and
   GPU memory peaks. Require duplicate schedule determinism, canonical schedule
   replay, exact target pairing/recurrence/mask/time/budget/fidelity checks,
   unchanged compiled/eager zero-control behavior before and after the solve,
   and no model-parameter gradients. Validate its artifact independently in a
   CPU `afterany` allocation. This canary is apparatus evidence only.
10. Freeze the three-case smoke by pre-existing R03 strata, all on `worker-1`:
   - preserve a constant-residual success: `crfs-1069f29a8d76463a`;
   - repair a clearance-only miss: `crfs-7eddaafffb4f9474`;
   - repair a progress-only miss: `crfs-bd7b0adf95145623`.
   The smoke proceeds only if source pairing, target reconfirmation, solver
   constraints, duplicate inference, and duplicate simulator replay all pass.
   The scientific smoke is GO only at teacher Safe-Progress Success 3/3.
   The immutable manifest is
   `manifests/r05a_inverse_flow_teacher_smoke.jsonl`, SHA-256
   `bdb8ccbba01ebf500e0f1bd0fe4a4043054f922a273f9e90eb3860cfe753a633`.
11. Every paired allocation retains frozen, fresh direct target, existing
    constant residual, inverse-flow teacher, and reversed-teacher-time-order
    diagnostic arms. The reverse-order arm has the same controls, mask, path,
    and energy; it tests whether timing rather than only the control multiset
    matters.
12. Only after the three-case smoke passes may the exact 17-case development
    population run. Its GO rule is teacher success at least 14/17, preservation
    of all nine historical constant-residual successes, and rescue of at least
    five of the eight historical misses. Constant residual must first reproduce
    9/17 or the population is a pairing mismatch, not a teacher result.
13. A case counts only when it has exact pairing, direct target reconfirmation,
    valid solver convergence and constraints, target fidelity, no
    bounds/saturation/nonfinite/early-terminal failure, and two exact simulator
    replays satisfying `D_sim >= 5 mm`, no forbidden contact, reach progress at
    least `0.029897349105658888 m`, and target/obstacle motion at most `1 mm`.
    Optimizer failures remain fixed-denominator failures.

## Consequences

R05A tests whether nonlinear action-to-flow transport explains the 9/17 gap;
it does not test student learnability. A passing teacher population authorizes
only a separate new-state data and student protocol. That later protocol must
obtain untouched official groups, repeat baseline/direct/teacher transfer,
freeze group-preserving splits, include naturally safe zero-control examples,
and use only deployment-available student inputs.

A failed teacher gate stops MLP training. A teacher that succeeds only with
invalid budget, clipping, final-step overwrite, simulator feedback, or
privileged geometry does not support this hypothesis.
