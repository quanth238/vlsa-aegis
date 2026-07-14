# 0018 — Accept R04A plumbing and stop before probe training

Status: accepted on 2026-07-14 after the allocation-backed R04A smoke and an
independent allocation-backed validator, before any latent perturbation label,
probe training, learned guidance, or R04 claim outcome.

## Evidence accepted

Slurm array task `27514_0` ran the frozen ADR-0017 apparatus from clean reviewed
commit `1138be311e255d891e8a3dfaace1a4e631da7479` on
`worker-mig-3g40gb-0`. It completed in 88 seconds with exit code `0:0` and wrote

```text
/mnt/data/quanth/experiments/crfs-oracle/
  r04a-label-contract-smoke-20260714a/
  crfs-93365b8b851365f2/r04-label-contract.json
```

The artifact SHA-256 is
`b820793ec42a5228c858e297d13806d8ae7f02e5cc7a765769316473d795f285`.
All five trace steps had exact duplicate traces, all ten complete sampler calls
returned the same physical action, both five-action simulator replays were
exact, and the geometry contained all 21 valid OBB slots without truncation.

Independent CPU Slurm job `27516` loaded that final artifact on `worker-2`,
reconstructed its hashes, traces, actions, raw simulator witness, reach label,
and outcome, and returned zero validation errors. The retained descriptive
label was `D_sim = 0.011531402902936597 m`, no contact, and reach progress
`0.02673899897703208 m`. Progress was below the frozen R00 threshold; the row
was correctly retained. It remains
`apparatus_only_never_train_calibrate_validate_test_or_claim`.

This accepts only the real trace-to-action-to-label plumbing. It does not make
R04 passing and is not evidence that a probe predicts, supplies a causal
gradient, improves Safe-Progress Success, or transports beyond the current
pregrasp controlled metric.

## Blocking design findings

### Exact post-edit continuation

The current bridge mode is not a perturbation-label source. It starts from the
original noise, applies `(1-t) * correction`, and returns the pre-edit trace.
The next apparatus substage is named **R04B exact latent resume/edit parity**.
It must add a distinct opt-in mode that consumes the captured absolute
normalized `x_t`, the captured float32 `time`, and a direct normalized latent
edit. The captured time is authoritative: repeated float32 Euler subtraction
does not in general equal `1 - step / num_steps` bit-for-bit. The post-edit
trace must contain the actual edited latent, recomputed velocity,
post-edit `predicted_clean`, and final normalized action.

Before any perturbation label, a zero-edit resume at every registered step must
exactly reproduce the full eager normalized final action and physical action in
an allocation. Duplicate edited resumes must also be exact. Residual, bridge,
and direct latent-edit semantics remain distinct, and the no-envelope baseline
path remains unchanged.

### Source-state estimand

The released `safelibero_spatial:II:0` finite state file has exactly 50
checked-in states and all 50 have already been used in R00--R03. Different policy noise,
replay-derived branches, or jitters of those states are not new independent
source groups.

Plain BDDL `env.reset()` is not a valid replacement for a released Level-II
state. The BDDL places six obstacles in workspace regions, while the released
states expose exactly one active workspace obstacle. `safety_level` only
selects a `.pruned_init` file; the repository contains no generator for the
pruning or parking transformation.

The preferred source is additional task-0 Level-II draws from the benchmark
authors' original generator. If that generator cannot be obtained, the fallback
must be an explicitly new estimand such as
`task0_single_obstacle_generated_v1`, never called official SafeLIBERO Level II.
It must store the finalized model XML, flattened simulator data state, complete
setup/settle history, generator and rejection records, BDDL/assets/controller
hashes, and observation/geometry hashes. A flattened MuJoCo data state alone is
insufficient because reset may change model body poses.

A new controlled generator changes the population. In that case R00 progress
calibration, direct feasibility, and privileged oracle steerability must be
reconfirmed on preregistered groups from the new population before training.
R03 authorization from the old 50-state population does not silently transfer.

### Perturbation and guidance support

The manuscript's 16 one-sided draws from `[-0.15, 0.15]^15` have
root-mean-square population scale `sqrt(E[RMS^2]) = 0.0866` and maximum RMS
`0.15`. Doses `0.25` through
`4.0`, and the prior witness range `0.77`--`1.15`, are outside that declared
input support. Reaching a large final-action norm does not repair this mismatch.

The one-sided design is therefore rejected before labels. The replacement must
use preregistered antithetic directions and multiple radii, retain invalid
continuations, and measure support in the actual post-edit probe feature as
well as latent and final-action coordinates. Candidate causal doses must lie
inside demonstrated valid feature support. Otherwise the result is a support
limitation, not evidence that the learned direction has no effect.

### Statistical unit and power

Every final statistic uses the immutable source episode/state as its unit, with
equal source-group weight. `N` never means latents, times, perturbations, or
noise seeds. All descendants of one source remain in one split and bootstrap
group.

The current minimum of 30 unsafe test groups makes the false-safe rule
effectively zero-error: at `0/30` the one-sided exact 95% upper bound is
`0.09503`, while `1/30` gives `0.14860` and fails. A powered final manifest must
be frozen from an explicit alternative and unsafe-group prevalence. For
reference, 103 unsafe groups allow at most five false-safe groups and give over
90% probability of satisfying the registered upper-bound rule when the true
conditional false-safe rate is 3%.

Nineteen calibration groups is only the mathematical minimum for a finite 95%
groupwise conformal quantile; at that size the quantile is the single maximum
score. Boundary MAE/rank also cannot be accepted from an empty set, two labels,
or one correlated group. The boundary band, minimum independent boundary groups
at every active time, conformal-group count, trigger-positive count, and paired
Gate-3/Gate-4 power calculation remain pre-training blockers. The power model
must include trigger prevalence and paired discordances, then simulate the
joint probability of all gate conditions rather than taking separate nominal
minimums.

## Decision

1. Accept R04A as allocation-backed apparatus evidence and retire its submission
   command. Repository artifact name `R04A` is canonical; the paper will refer
   to later scientific stages by Gate number rather than reuse `R04A/R04B` for
   different experiments.
2. Keep feature R04 `active`. Do not train a probe or collect perturbation labels.
3. Implement and allocation-test R04B zero-edit resume parity next.
4. In parallel, seek the original Level-II generator. If unavailable, freeze a
   new controlled generator and repeat the prerequisite ladder for that estimand.
5. Freeze the support-matched perturbation design, exact boundary definition,
   group counts, and final statistical code only after the state estimand exists
   and before labels or outcomes are opened.

## Consequences

- The proposed continuation-clearance probe remains scientifically plausible,
  but it is not yet an executable or powered efficacy experiment.
- Exact continuation labels solve the earlier label-mismatch problem; they do
  not solve large-action support, intervention timing, nonlinear flow response,
  or the clearance/progress tradeoff exposed by R02/R03.
- A scalar clearance gradient may repeat the registered analytic arm's failure:
  improve clearance while destroying reach progress. Safe-Progress Success and
  the stronger margin-aware analytic control therefore remain mandatory.
- No endpoint-preserving experiment is reopened.
