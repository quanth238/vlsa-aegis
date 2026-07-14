# CRFS oracle harness on VLSA-Aegis

This repository is a baseline-first research harness for testing Counterfactual Residual Flow Steering (CRFS) in π0.5 on SafeLIBERO. The Git history and implementation start from [THU-RCSCT/VLSA-Aegis](https://github.com/THU-RCSCT/vlsa-aegis) commit `57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b`. CRFS is an additive experiment layer, not a replacement benchmark or a clean-room reimplementation.

The original oracle question was deliberately narrower than learning a safety probe:

> Given the exact nearest safe, endpoint-preserving correction for a colliding five-action prefix, can an intervention inside the frozen π0.5 action flow rescue the same observation–noise sample more often than an equal-norm random direction?

H05 stopped before that intervention: exact endpoint preservation made all 20
five-action and all 20 ten-action repairs analytically infeasible. R00/R01 then
showed endpoint-free safe-progress actions for 17/20 pregrasp reach states.
R02/R03 passed the preregistered oracle-steerability and analysis gates: the
distributed oracle residual passed 9/17 eligible cases versus 0/17 for matched
random and registered analytic controls. R04 is now the only active gate. This
authorizes a bounded learned-probe investigation, not training by default.
ADR-0017 froze a one-case R04A label-contract smoke; Slurm job `27514_0` and
independent validator `27516` passed that plumbing contract. ADR-0018 still
stops before perturbation labels or training because exact latent resume,
independent state provenance, perturbation support, and powered group counts
remain unresolved. No probe has yet been trained or shown effective.

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

The completed CRFS sequence is H00–H10; the endpoint-free pivot is R00–R04 in
`feature_list.json`. R00–R03 are passing and R04 is the only active gate.

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
                              -> R04 learned probe only after oracle success
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

The R00–R04A allocation evidence and exact Slurm job IDs are recorded in
`PROGRESS.md` and `evidence/`. There is intentionally no R04 training command.
R04A verified real sampler-trace to physical-action to raw-simulator-label
plumbing on one reused R00 state; it cannot support a learning or efficacy
claim. Claim-bearing R04 still requires exact post-edit continuation parity,
genuinely new source-episode groups from a defined estimand, a frozen
group-preserving split, support-matched perturbations, powered boundary and
false-safe coverage, gradient causality, and matched controls. Run live
preflight immediately before every submission.

The detailed procedure is in [the experiment protocol](docs/experiment_protocol.md) and [VinUni runbook](docs/infrastructure/vinuni_h100_runbook.md).

## Research materials

- `main.tex`: historical proposal containing the stopped exact-endpoint premise;
- `Chat - Probe in Diffusion Model.md`: design discussion and the endpoint-free pivot record;
- `docs/decisions/0017-freeze-r04a-continuation-label-contract.md`: frozen R04A contract;
- `docs/decisions/0018-accept-r04a-stop-before-training.md`: passed plumbing evidence and current stop;
- `evidence/r04a/r04a-validation.json`: compact allocation/validator record;
- `references.bib`: bibliography;
- `CRFS_Proposal_Preview.pdf`: compiled preview.

The proposal’s reported safety measure is minimum controlled simulator clearance
over every physics substep, cross-checked against contact pairs. R00/R01 also
measure maximum target and active-obstacle displacement over those substeps;
endpoint-only motion does not satisfy the no-pushing gate.
