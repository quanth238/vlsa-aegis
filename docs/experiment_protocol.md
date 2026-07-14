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

For each of five control actions, observe every robosuite/MuJoCo physics substep immediately after `sim.step()`. The primary controlled metric is Eq. (3): exact signed point-to-oriented-box distance from the simulator EEF-sphere center to every collision box in the active obstacle union, minus the preregistered 6 cm radius. Cross-check physical EEF--obstacle contacts as a one-way conservatism condition and store the raw mesh--box `mj_geomDistance` only as advisory diagnostics. Store the minimum pair and transform, sample count, contact flag, start/end EEF positions, and task-success flag.

H03 evidence is valid only after the separate sphere--box boundary calibration, five repeated executions on each of 50 unique saved states, clearance variation below $10^{-4}$ m, correct 126-sample count, no physical contact at positive proxy clearance, and a rendered minimum-pair artifact.

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

## Endpoint-free pivot after H05

H05 is preserved as a terminal negative result for exact endpoint equivalence.
The new sequence is opt-in and does not reinterpret H05 as a solver failure.

### R00: reach-progress calibration

The frozen 20 H05 branches are initial pre-grasp reach states for
`safelibero_spatial:II:0`, not post-grasp transport states. For the same-case
diagnostic, freeze `akita_black_bowl_1` at its branch-start position and define

```text
DeltaPhi_reach = ||eef_0 - target_0|| - ||eef_5 - target_0||.
```

Use the first five executed actions. Calibrate `p_min` as the `inverted_cdf`
lower quartile of positive nominal progress among at least 50 simulator-safe,
phase-valid chunks from episode groups disjoint from R01. Reject calibration
chunks that move the target bowl or active obstacle beyond 1 mm. Retain and
report unsafe, nonpositive, moved-object, and invalid-phase cases.

Measure bowl and obstacle displacement from direct MuJoCo body positions after
every physics substep. Endpoint-only displacement is insufficient because a
body can move beyond tolerance and return before the fifth action ends.

### R01: endpoint-free physical feasibility

Search all 5x3 translation commands under action bounds with no zero-sum or
fixed-endpoint constraint. Keep nominal orientation and gripper commands. The
frozen H04 response and static branch boxes provide `D_opt` for candidate
search only. Direct repeated simulator replay decides whether a witness has:

- `D_sim >= 5 mm` at the branch point and all 125 physics substeps;
- no forbidden EEF--obstacle contact;
- `DeltaPhi_reach >= p_min`;
- target-bowl and active-obstacle displacement at most 1 mm.

Also search and report witnesses at `p = 0`. A case safe at `p = 0` but not at
`p_min` diagnoses a short-horizon progress conflict rather than absence of a
local safe detour. A finite search miss is `not_found_within_budget`, never an
infeasibility certificate. The development gate passes at 12/20 verified
safe-progress witnesses.

Before entering the rescue numerator, the paired nominal replay must reproduce
the registered collision (`D_sim < 0` or contact). Record a non-reproduced
nominal as a population mismatch rather than a trivial rescue. A rescue must
change at least one translational command while preserving all nominal
orientation and gripper commands. Direct candidates nominated by both the
`p_min` and `p = 0` searches are pooled and classified against both progress
thresholds; the two searches use the same registered random stream.

### R02--R04: causal ordering

After R01 passes, compare a direct planner reference, endpoint-free oracle flow
direction, analytic geometry direction, and equal-norm random direction under
identical state, observation, policy noise, and executed horizon. Only a
passing paired oracle intervention and grouped analysis can authorize probe
training. Probe labels must describe the actual deterministic continuation (or
the rolled-out approximate-clean action), not pair `A_hat_t` with the clearance
of a different action.

Before any R02 policy outcome is interpreted, an allocation-backed parity
artifact must compare the public JAX checkpoint and its PyTorch conversion on
identical transformed observations, explicit Gaussian noise, ten Euler steps,
and unnormalized actions. Record per-step and final absolute errors and fail
closed if the registered tolerances are not met.

For each R01 changed-action witness, define `Delta_star` as the first-five
translation commands of the selected immutable direct witness minus the fresh
paired eager commands from the current allocation. Convert this displacement
to normalized model coordinates with scale only. Keep the historical R01
nominal and historical delta as diagnostics, but do not use a stale historical
delta as the intervention. Keep rotation, gripper, and the discarded action
tail at zero. At flow time `t_s = 0.5`, run these paired arms:

1. frozen sampler;
2. direct witness execution (physical reference, not a flow arm);
3. endpoint-free equal-norm random distributed residual;
4. analytic sphere/box geometry direction with the same translation mask and
   intervention budget;
5. endpoint-free oracle distributed residual targeting `Delta_star` over the
   remaining Euler interval;
6. one-shot bridge edit as a separate diagnostic.

Every model arm must reuse the exact reconstructed R01 state and the same fresh
observation, instruction, policy noise, pre-intervention eager trace, and
five-action execution horizon. Duplicate and final eager replays must be exact.
Fresh-eager versus historical-R01 action drift is reported and must pass the
already-frozen ADR-0010 numerical limits; no outcome-fitted tolerance is added.
ADR-0013 defines the current paired baseline as the causal reference. Reconfirm
the direct witness and nominal collision. Do not clip an out-of-bounds sampled
action into a pass; record it as a bounds failure. Cases without an R01 witness
remain in the all-collision report and are never silently dropped.

R02 is an apparatus gate requiring complete schema-valid paired artifacts. R03
is the scientific gate. On the R01-feasible population, the preregistered
primary distributed-oracle conditions are:

```text
SPSR_oracle >= 0.50
SPSR_oracle - SPSR_random >= 0.20
paired complete-group 95% bootstrap LCB(SPSR_oracle - SPSR_random) > 0
```

Report feasible-conditioned and original-20-case populations separately. If
the oracle gate fails, do not train a probe. R03 passing establishes oracle
steerability only. Apply ADR-0012's separate learned-probe authorization: the
analytic arm must fail its matched oracle-style gate, the grouped 95% bootstrap
LCB for oracle minus analytic must be positive, and the exact one-sided paired
label-swap test must have `p < 0.05`. Bounds/direction failures remain fixed-
denominator zeroes. If analytic geometry already explains the gain at the same
intervention budget, this pilot does not justify a learned probe.

### R04A: real-continuation label-contract smoke

R04A is the no-learning prerequisite in ADR-0017. It reuses exactly one R00
row as `apparatus_only`; that row may never enter training, calibration,
validation, testing, or a claim. From one fixed branch, observation, checkpoint,
and 10x32 policy-noise tensor, request eager no-intervention traces twice at
steps 1--5. Every request must return the exact same physical 10x7 final action,
and each normalized `predicted_clean` record must reconstruct exactly as
`x_t - t*v_base`.

Execute the resulting physical five-action prefix twice from the exact branch.
For each replay, independently reconstruct the inclusive branch-plus-125-
substep sphere/OBB `D_sim` witness and the tracked-body reach annotation. Bind
every trace to the final-action hash, both raw rollout hashes, and the minimum
`D_sim` label. `D_opt` and optimizer status are explicitly not applicable.
Invalid phase, collision, negative progress, moved objects, and launch failures
remain recorded rather than filtered.

R04A does not resume or edit a latent and cannot validate gradient causality.
A later perturbation gate must expose and test a deterministic post-edit
resume path; the existing bridge trace is pre-edit and invalid for that label.
For claim-bearing data, split and bootstrap immutable source initial
episode/state groups: every replay-derived branch, history, policy noise,
perturbation, and rollout from one source remains together. New grouped states
and registered 5 mm boundary coverage are required before training.
