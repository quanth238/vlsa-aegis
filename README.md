# CRFS oracle harness on VLSA-Aegis

This repository is a baseline-first research harness for testing Counterfactual Residual Flow Steering (CRFS) in π0.5 on SafeLIBERO. The Git history and implementation start from [THU-RCSCT/VLSA-Aegis](https://github.com/THU-RCSCT/vlsa-aegis) commit `57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b`. CRFS is an additive experiment layer, not a replacement benchmark or a clean-room reimplementation.

The original oracle question was deliberately narrower than learning a safety probe:

> Given the exact nearest safe, endpoint-preserving correction for a colliding five-action prefix, can an intervention inside the frozen π0.5 action flow rescue the same observation–noise sample more often than an equal-norm random direction?

H05 stopped before that intervention: exact endpoint preservation made all 20
five-action and all 20 ten-action repairs analytically infeasible. R00/R01 then
showed endpoint-free safe-progress actions for 17/20 pregrasp reach states.
R02/R03 passed the preregistered oracle-steerability and analysis gates: the
distributed oracle residual passed 9/17 eligible cases versus 0/17 for matched
random and registered analytic controls. A valid R03A smoke showed that two
stronger analytic fields avoided contact on one case but lost task progress;
ADR-0028 intentionally retires its unlaunched population because it does not
test the selected action-to-flow transport hypothesis. R05A is now the only
active gate: it tests whether the safe-progress action target can be converted
through the exact frozen sampler into a budgeted time-dependent residual
velocity sequence. No probe or residual-field MLP has been trained.
IFT-00 now passes as synthetic implementation evidence; the next gate is a
one-case, no-efficacy allocation canary that measures real sampler integration
and memory before any robot rollout. Two immutable attempts remain incomplete:
retry B reached real pi0.5 and produced a deterministic finite-search miss, but
host-memory finalization failed, so no accepted canary result exists.
ADR-0017 froze a one-case R04A label-contract smoke; Slurm job `27514_0` and
independent validator `27516` passed that plumbing contract. ADR-0018 still
stops before perturbation labels or training. ADR-0019 froze the final
apparatus subgate, and Slurm task `27558_0` plus independent validator `27559`
now pass its exact saved-latent resume/parity and current-commit
ordinary-baseline contract. Independent state provenance,
perturbation support, and powered group counts remain unresolved. No probe has
yet been trained or shown effective.

ADR-0021's proposed non-Level-II generated source route stopped at its
outcome-blind runtime-identity canary. ADR-0022 retires that route and its nine
dependent groups. A claim-bearing learned study still needs an untouched
official source population; no generated group may train or evaluate ECG.

## Baseline and extensions

The released baseline remains in its original locations:

- `main/main_aegis.py` and `main/main_aegis_translational.py`: AEGIS evaluation runners;
- `safelibero/`: SafeLIBERO task definitions, initial states, objects, and environment wrappers;
- `openpi/`: the baseline OpenPI fork and π0.5 policy stack.

The harness adds opt-in seams:

1. the PyTorch sampler can expose the midpoint trace and apply either distributed residual velocity or a one-shot bridge edit;
2. policy requests may carry a reserved `__crfs__` control envelope with fixed noise and a physical or normalized correction;
3. SafeLIBERO can call back after each hidden MuJoCo physics substep without changing ordinary `env.step`;
4. `main/crfs_oracle/` runs same-state/same-noise nominal, direct-repair,
   equal-norm random, registered analytic, oracle-residual, and bridge-edit branches.
5. the R00/R01 path separately measures five-action reach progress and searches
   endpoint-free candidates while keeping `D_opt` and direct `D_sim` verification distinct.

No-control sampler calls retain the baseline return type and integration path.

## Evidence ladder

The completed CRFS sequence is H00–H10; the endpoint-free sequence and current
transport pivot are R00–R05A in `feature_list.json`. R00–R03 are passing,
R03A/R04 are historical blocked gates, and R05A is active.

```text
baseline + provenance + replay
             |
             v
substep measurement -> kinematics / D_opt-vs-D_sim calibration
             |
             v
projection teacher -> direct A+ replay -> sampler trace/sign
             |
             v
paired oracle intervention -> population analysis -> go / stop

H05 endpoint contradiction -> R00 progress calibration
                              -> R01 endpoint-free feasibility
                              -> R02/R03 oracle steerability and analysis
                              -> R03A one-case analytic diagnostic (retired)
                              -> R05A inverse-flow teacher transport
                              -> student study only after new-state prerequisites
```

The dependency-free synthetic fixture checks implementation only. Real artifacts
remain preliminary: released SafeLIBERO obstacles are movable, the calibrated
6 cm EEF sphere is a controlled proxy rather than full-arm geometry, and H04's
response model is independently validated only over its registered action range.

## Local verification

The local machine does not need MuJoCo or PyTorch for the harness gate:

```bash
./init.sh
make synthetic
```

`./init.sh` compiles all new Python modules, runs unit tests, audits required harness artifacts and baseline compatibility, and checks the patch for whitespace errors.

## Allocation-backed experiments

Real execution uses the existing two-environment baseline boundary: Python 3.11/PyTorch for the OpenPI server and the LIBERO client environment for simulation. Both processes run inside one Slurm allocation; the transient server is terminated by a shell trap.

The R00–R04B allocation evidence and exact Slurm job IDs are recorded in the
archived detailed progress record and `evidence/`. There is intentionally no
R04 or R05A training command.
R04A verified real sampler-trace to physical-action to raw-simulator-label
plumbing on one reused R00 state; it cannot support a learning or efficacy
claim. R04B is limited to one exact post-edit continuation/parity smoke and
does not execute a guidance outcome. ADR-0028 keeps that apparatus available
but blocks the scalar-probe study while R05A tests action-to-flow teacher
transport. Any later student study still requires genuinely new source-episode
groups from a defined estimand, a frozen group-preserving split, support-matched
coverage, and matched controls. Run live preflight immediately before every
submission.

The detailed procedure is in [the experiment protocol](docs/experiment_protocol.md) and [VinUni runbook](docs/infrastructure/vinuni_h100_runbook.md).

## Research materials

- `main.tex`: historical proposal containing the stopped exact-endpoint premise;
- `Chat - Probe in Diffusion Model.md`: design discussion and the endpoint-free pivot record;
- `docs/decisions/0017-freeze-r04a-continuation-label-contract.md`: frozen R04A contract;
- `docs/decisions/0018-accept-r04a-stop-before-training.md`: passed plumbing evidence and current stop;
- `docs/decisions/0019-freeze-r04b-exact-resume-parity.md`: final apparatus contract before direct probe experiments;
- `docs/decisions/0020-accept-r04b-focus-direct-probe-test.md`: passed R04B evidence and frozen direct research question;
- `docs/decisions/0021-freeze-generated-source-canary-and-direct-transfer-pilot.md`: new-state identity, canary, and direct privileged-transfer stop rule;
- `docs/decisions/0023-run-strong-analytic-kill-test.md`: frozen no-learning R03A necessity test;
- `docs/decisions/0027-register-source-node-grouped-r03a-population.md`: historical allocation-backed smoke evidence and the now-retired grouped-population launch contract;
- `docs/decisions/0028-pivot-to-inverse-flow-transport.md`: current controlled pivot and R05A stop rules;
- `docs/decisions/0031-preserve-ift00a-retry-b-and-repair-apparatus.md`: retry-B terminal interpretation and apparatus-only repair boundary;
- `EXPERIMENTS.md`: short active experiment ledger and exact go/no-go sequence;
- `evidence/r04a/r04a-validation.json`: compact allocation/validator record;
- `evidence/r04b/r04b-validation.json`: compact exact-resume allocation/validator record;
- `references.bib`: bibliography;
- `CRFS_Proposal_Preview.pdf`: compiled preview.

The proposal’s reported safety measure is minimum controlled simulator clearance
over every physics substep, cross-checked against contact pairs. R00/R01 also
measure maximum target and active-obstacle displacement over those substeps;
endpoint-only motion does not satisfy the no-pushing gate.
