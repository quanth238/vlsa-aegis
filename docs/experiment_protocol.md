# Oracle experiment protocol

## Registered pilot

- Baseline: VLSA-Aegis plus π0.5 LIBERO checkpoint.
- Task: one frozen SafeLIBERO suite/task split at a time.
- Executed prefix: first five of ten predicted actions.
- Controlled dimensions: world-frame OSC translation only; orientation and gripper stay nominal.
- Primary intervention: midpoint, sampler step 5 of 10 (`t_s = 0.5`).
- Repair: minimum squared translation change subject to valid action bounds, positive safety margin, and zero-sum temporal correction.
- Pairing: identical initial state, settle history, rendered observation, and 10x32 policy-noise tensor.

## Required arms

1. Nominal frozen policy.
2. Equal-norm endpoint-preserving random residual.
3. Exact oracle distributed residual.
4. Direct execution of repaired `A+` as teacher upper bound.
5. One-shot bridge edit as a distinct mechanism diagnostic.

Do not select among arms or intervention times on the test set.

## Measurement

For each of five control actions, observe every robosuite/MuJoCo physics substep immediately after `sim.step()`. Compute the minimum `mj_geomDistance` over Panda gripper contact geoms and the active obstacle’s contact geoms. Cross-check all simulator contacts for those geom ID sets. Store the minimum pair, sample count, contact flag, start/end EEF positions, and task-success flag.

The released obstacle-displacement threshold is not the CRFS supervision or primary mechanism metric.

## Ordered execution

1. Preflight live Slurm, QOS, storage, jobs, processes, code, checkpoint, and output paths.
2. Validate the immutable manifest and select exactly one case by array index.
3. Start the modified policy server inside the allocation.
4. Construct and settle the baseline environment; serialize one policy observation.
5. Query the policy twice with exactly the same observation and noise. Abort unless actions and trace are byte-exact.
6. Reset, replay, and execute the nominal prefix twice. Abort unless endpoint and clearance agree to `1e-9`.
7. Measure nominal collision. If safe, record zero repair; it is not part of the colliding oracle population.
8. For a collision, solve the offline repair and directly replay `A+` from the identical branch point.
9. Run random, oracle residual, and bridge arms with the same observation/noise.
10. Write `results.json` atomically. Validate before counting completion.

## Population discipline

The primary oracle population is `D(A-) < 0` and a feasible verified repair. Separately report all colliding cases including infeasible and failed repair attempts. Bootstrap complete `group_id` clusters. A one-case smoke validates apparatus only and cannot satisfy a population gate.

## Stop rules

- Stop on non-deterministic state/noise replay.
- Stop if contact and signed distance disagree beyond a diagnosed numerical tolerance.
- Stop if direct repaired safety is below 0.95 over feasible cases.
- If five-step feasibility is below 0.40, try H=10 once; if it remains below 0.60, reject the local endpoint-preserving premise.
- If oracle steering does not beat equal-norm random after registered intervention-time fallbacks, do not train a learned predictor.
