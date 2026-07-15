# 0040 — Preregister the same-budget constrained-flow diagnostic

Status: frozen on 2026-07-15 before any real CFS outcome, immutable H100 run-ID
selection, CFS-00A submission, generated simulator action, or probe or
residual-field training.  Implementation and synthetic tests may coexist with
this decision in its implementation commit and are not experimental evidence.

## Context

The sampled-current Run B payload reached frozen pi0.5 and completed two
deterministic 128-update searches.  It was not an accepted scientific result
because its CPU publisher failed, but its preserved raw diagnostic is useful:
the two searches were finite and identical, consumed the exact registered flow
budget, and missed the four physical fidelity gates by a large margin.  The
first-five XYZ maximum error was `0.8631911873817444` and RMS error was
`0.3312218487262726` physical action units.  Failed controls were not applied
and no policy or teacher action was executed in the simulator.

The historical search was not zero-initialized.  For each of the five active
flow steps it initialized the integrated increment as

```text
c_k = Delta_model / 5,
```

projected it onto the registered mask and per-step ball, and then ran the frozen
128-update Adam search.  It also had no separate decoded-action budget.  A
proposal that calls zero initialization the Run B baseline, adds a new
`B_action`, omits the seven-channel gates, or calls the immutable source an A*
planner output does not reproduce Run B and cannot isolate the cause of its
failure.

The corrected hypothesis is narrower:

> Under the exact Run B target and flow budget, a constrained inverse computed
> from the frozen sampler's zero-control physical Jacobian provides a
> coupling-aware integrated-increment candidate or initialization that reaches
> the target more faithfully than Run B's uniform `Delta_model / 5`
> initialization.

This is a transport diagnostic.  It asks whether the privileged final-action
delta can be expressed through the sampler's vector field under the registered
budget.  It does not test collision avoidance, task progress, simulator
efficacy, generalization, or student learnability.

## Frozen source and comparison

1. Register this work as the new mechanism identity **IFT-00B / CFS-00A**.  It
   is not a retry, completion, or retrofit of IFT-00A Run B.  The Run B root,
   logs, payload, receipts, and consumed run ID remain immutable.
2. Use only row zero of
   `manifests/r05a_inverse_flow_teacher_smoke.jsonl`: case
   `crfs-1069f29a8d76463a`, group `safelibero_spatial:II:0:46`, environment
   seed `924805038`, policy seed `1179198633`, and source host `worker-1`.
   Freeze the same observation, instruction, explicit policy noise, checkpoint,
   normalization asset, sampler, Euler recurrence, and source R02 artifact.
3. Preserve the R01/R02 target construction exactly.  Reconstruct the fresh
   paired frozen normalized output and add the immutable R02
   `delta_star_model` only in first-five XYZ coordinates.  Coordinates outside
   that mask remain bitwise equal to the paired frozen output.  Convert
   physical displacement with checkpoint scale only, never by subtracting a
   normalization mean.  Do not query a planner, geometry, collision distance,
   or simulator outcome while constructing or solving the transport.
4. Bind the runtime budget to the immutable source scalar, not to a recomputed
   norm:

   ```text
   B_float32 = 3.6398398876190186.
   ```

   Record separately the source float64 norm, the cast-delta norm, and the
   realized float32 target-minus-frozen norm.  No observed value may silently
   replace `B_float32`.
5. Optimize the same 75 scalar variables as Run B: five active flow steps
   `k = 5,...,9`, each with first-five XYZ integrated increments
   `c_k = dt * u_k`, where `dt = -0.1`.  Steps `0,...,4` and every coordinate
   outside first-five XYZ remain exactly zero.  Require

   ```text
   ||c_k||_2 <= B / 5 for every active k,
   sum_k ||c_k||_2 <= B.
   ```

   The product of five `B/5` balls implies the path bound, but both invariants
   must be validated independently after float32 transport.  Clipping, terminal
   overwrite, post-sampling action addition, and a separate `B_action` are
   forbidden.
6. Measure terminal error after checkpoint physical scaling.  Preserve all four
   Run B gates:

   - first-five XYZ maximum absolute error `<= 0.010`;
   - first-five XYZ RMS error `<= 0.005`;
   - first-five seven-channel maximum absolute error `<= 0.050`;
   - first-five seven-channel RMS error `<= 0.015`.

   The optimization objective also remains the Run B weighted mean-square
   objective: divide first-five XYZ errors by `0.005`, divide the remaining
   first-five seven-channel errors by `0.015`, concatenate all 35 values, and
   average their squares.  The max gates remain acceptance checks rather than
   objective weights.

## Frozen arms

Run all three arms from the same paired zero-control rollout and target.

### Arm A — exact historical Run B

Use the byte-preserved historical direct-shooting implementation: initialize
each active `c_k` to `Delta_model / 5`, project with the historical product-ball
projection, and run exactly 128 Adam updates with learning rate `0.02`,
`beta1 = 0.9`, `beta2 = 0.999`, `epsilon = 1e-8`, eight float32 constraint-slack
ULPs, and no early stop.  This arm must reproduce Run B's finite nonconvergence,
schedule SHA-256
`714ac52e791248e7590196b5dfe47cc5f4a4966dce0a0237092a9cd939b72fc7`,
and returned-action SHA-256
`23c855c3109ffcb78dce3c2671cae52a3fafdd820d2c4ab1d0c248da51fe48c1`.
Any mismatch makes the comparison apparatus-inconclusive.

### Arm B — constrained zero-control linearization

At exactly zero integrated control, differentiate the frozen five active Euler
steps with respect to the 75 active integrated-increment coordinates.  The
output is the first-five seven-channel terminal action after checkpoint
physical scaling, so the registered Jacobian has shape `35 x 75`.  Differentiate
with respect to `c`, not velocity `u`.  Cast the fixed Jacobian and residual to
float64 once and solve the weighted linearized problem

```text
minimize_c  0.5 ||A c + b||_2^2
subject to  ||c_k||_2 <= B / 5,  k = 5,...,9,
```

where `A = W J`, `b = W r_0`, and `W` applies `1/0.005` to XYZ and
`1/0.015` to the remaining four channels in each of the first five action
rows.  The factor differs from the historical mean-square reporting metric by
only a positive constant and therefore has the same minimizers.

Freeze the deterministic CPU `torch.float64` projected FISTA implementation as
follows.  Construct `A` and `b` in `torch.float64`; compute `sigma_max` with
`torch.linalg.svdvals(A)`; use the fixed step `1 / sigma_max^2`; initialize
`x_0 = y_0 = 0`; run exactly 4,096 standard FISTA updates with no adaptive
restart, early stop, or tolerance; and project each of the five 15-coordinate
groups independently onto its exact radius-`B/5` ball after every update.
Select the projected iterate with the lowest historical weighted mean-square
metric among `x_0,...,x_4096`, choosing the earliest iterate on an exact tie.
If and only if the Jacobian is exactly zero, return explicit status
`ZERO_JACOBIAN`, the zero candidate, and zero updates.  Record the projected-
gradient mapping norm as a diagnostic with no acceptance threshold.

Cast the selected candidate to the model dtype once, apply the production
product-ball projection once to protect the exact float32 mask/budget contract,
validate the original mask and both budget checks, and replay it through the
full nonlinear frozen recurrence without optimization.  Record both the
float64 selected candidate and the model-dtype pre/post-projection candidates;
the residual-schedule transport executes the float32 increment
`dt * (c / dt)`, which need not be byte-identical to `c`, so recompute the
reported linear prediction and executed objective from that exact executed
increment.  Arm C still receives the recorded post-projection `c` as its sole
changed initialization.  Only the nonlinear replay can pass a transport
fidelity gate.  No numeric value may be changed after observing the real
result.

The FISTA result is a deterministic candidate for this convex weighted
least-squares model.  Its finite residual, projected-gradient diagnostic, or
`ZERO_JACOBIAN` status is not an SOCP certificate and cannot establish linear
or nonlinear infeasibility.

The allocation client and CPU validator independently reconstruct `W`, `A`,
`b`, both singular spectra, the fixed FISTA recurrence, objectives, linear
prediction, and production projection with NumPy float64/float32 algebra.  To
avoid treating backend reduction order as scientific signal, the independent
candidate must have maximum absolute or relative L2 difference at most
`1e-8`; singular values use relative `1e-10` and absolute `1e-12`; the
spectral step uses relative `1e-10` and absolute `1e-12`; objectives use
relative and absolute `1e-10`; the projected-gradient mapping uses relative
and absolute `1e-9`; the
float32 linear prediction uses relative and absolute `1e-6`; and the NumPy
production projection may differ by at most `2e-7`.  The independently chosen
iteration is diagnostic only because nearly tied iterates can differ across
BLAS backends; the server-selected iteration must still be exactly duplicated
between the two model-server invocations.  These audit thresholds are frozen
before the real run and are not transport tolerances.

The CPU validator must derive `b` from the exact frozen baseline, target, and
checkpoint scale rather than trusting a server summary.  It must also rebuild
the full baseline, linear, nonlinear, and linearization-error tensors from
those source arrays, the exact executed increment, and the canonical rollout.
`ZERO_JACOBIAN` is valid if and only if the recorded Jacobian is exactly zero;
otherwise the status is `SOLVED` with exactly 4,096 updates.  The recorded
float64/CPU solver identity and false certificate flags are acceptance gates,
and the unchanged 128-update refinement must make exactly 129 historical
projection calls including its injected initialization.

Validate the autograd Jacobian before interpreting either new arm.  Use three
fixed unit directions in the 75 controls: all ones; alternating `+1,-1`
starting positive; and `+1` where `(17*j+3) mod 31 < 15`, otherwise `-1`.
Normalize each in float64, cast once to model dtype, and compare `Jd` with the
full frozen recurrence's central difference at epsilon values
`(B/5)*2^-8` and `(B/5)*2^-9`.  For each direction, at least one epsilon must
have relative L2 error at most `0.10` or absolute L2 error at most `0.001`;
the relative denominator is the maximum of the two derivative norms and
`1e-6`.  All three directions must pass.  These values are frozen before the
real case and may not be tuned from its outcome.  Preserve the plus and minus
physical target vectors as raw evidence.  Independently reconstruct each
central derivative from those vectors and epsilon, reconstruct `Jd`, both
error norms, every threshold decision, and the three-direction global pass.
Cross-backend float32 reductions receive only the fixed arithmetic bound
`128 * eps(float32) * max(1, reduction magnitude)`; this is an evidence-
consistency allowance, not a transport tolerance.

### Arm C — unchanged nonlinear refinement from Arm B

Initialize the historical 128-update Adam search with Arm B's validated
float32 integrated increments.  Change no optimizer, objective, constraint,
target, mask, recurrence, tolerance, iteration count, or selection rule from
Arm A.  Arm C therefore changes only the initial integrated increments.  If
Arm B's direct nonlinear replay already passes all four gates, preserve that
mechanism pass even if Arm C later degrades it; such degradation identifies a
refinement/iterate-selection defect rather than failure of the linearized
transport candidate.

## Determinism and acceptance gates

1. Invoke every arm twice with exact paired inputs.  Require exact duplicate
   Jacobians, linear candidates, schedules, statuses, selected iterates, and
   returned-action bytes.  Canonically replay every finite Arm B candidate and
   every selected Arm A/C schedule through the ordinary frozen recurrence.
   Arm B's returned-action bytes are those produced by this no-simulator
   canonical residual-schedule request.  Bind Arm C's original comparison
   reply bytes independently to the audited physical trace, and bind every
   arm's canonical replay reply to its own audited physical trace.  Require
   exact recorded float32 recurrence and returned-action bytes for all three.
2. Recheck compiled and eager zero-control outputs before and after all arms.
   Require unchanged default sampler behavior, no model-parameter gradients,
   finite Jacobian/solver/rollout values, the registered finite-difference
   Jacobian validation, exact mask/time support, and both
   budget constraints.  Record and label both the raw-Jacobian and weighted-
   matrix singular values.  Define the raw-Jacobian diagnostic effective rank
   by the fixed threshold
   `sigma_max * max(35,75) * eps(torch.float64)`;
   weighted residuals, projected-gradient or gradient-mapping diagnostics, and
   linear-versus-nonlinear prediction error as diagnostics only.
3. A valid **mechanism pass** requires exact Arm A historical reproduction and
   either Arm B's nonlinear replay or Arm C's selected nonlinear replay to meet
   all four physical fidelity gates under the exact budget.  It supports only
   the one-case claim that coupling-aware flow-space inversion can repair this
   historical transport failure.
4. A valid **frozen-method negative** requires exact Arm A reproduction, a
   numerically valid registered linear solve, two deterministic finite Arm B/C
   runs, valid constraints and replays, and neither Arm B nor Arm C meeting all
   four gates.  It rejects this frozen diagnostic configuration on this case,
   not all vector-field steering.
5. Nonfinite values, source/pairing/hash mismatch, Arm A reproduction mismatch,
   duplicate mismatch, invalid Jacobian, failed numeric solver checks,
   recurrence mismatch, budget violation, default-sampler drift, missing
   telemetry, test failure, or publication failure are
   **apparatus-inconclusive**, not a method result.

## Causal interpretation frozen before outcomes

- Arm B meets its linear prediction and its nonlinear replay passes: the local
  coupling model is sufficient at this budget; nonlinear refinement is not
  needed for this case.
- Arm B's linear prediction passes, Arm B's nonlinear replay fails, and Arm C
  passes: the Jacobian is useful as an initializer but finite-displacement
  nonlinear refinement is necessary.
- Arm B's nonlinear replay passes but Arm C fails: retain the Arm B pass and
  diagnose the fixed Adam refinement or iterate selection; do not discard a
  valid transport candidate.
- Arm B improves on Arm A but neither B nor C passes: the initialization carries
  useful coupling information, but the frozen local model/search is not yet
  sufficient.
- Arm B's linear prediction passes while both nonlinear replays fail: the main
  observed issue is local-linearization error at the registered displacement,
  not an infeasibility certificate.
- The registered linear solve itself misses the gates: the selected weighted
  linear objective and budget did not produce a passing candidate.  This does
  not prove that another candidate, nonlinear method, or budget-valid transport
  is impossible.
- All valid arms fail with little separation: this one-case result does not
  support the proposed initializer and stops automatic expansion.

No outcome may be described as collision avoidance, safety improvement, task
success, or simulator efficacy because CFS-00A executes zero generated actions
in the simulator.  No outcome authorizes IFT-01, a three-case efficacy smoke,
population testing, labels, probe training, or MLP training automatically.

## Release boundary

This decision authorizes implementation and local/allocation-backed apparatus
tests only.  The scientific config remains `ready_to_run: false`.  Before one
H100 submission, require a new result schema and fail-closed semantic validator,
focused synthetic linear/nonlinear tests, default-sampler regression tests,
duplicate and canonical-replay tests, exact resource/telemetry/publication
contracts, complete `./init.sh`, independent scientific/HPC/adversarial review,
a clean pushed and synchronized release commit, immutable hashes for every live
source file, an unused exact run ID, and a live VinUni preflight.  The login node
remains control plane only.

There is no tuning loop.  A terminal valid negative stops this exact direction
for review; a mechanism pass permits only a separately preregistered simulator
efficacy experiment.
