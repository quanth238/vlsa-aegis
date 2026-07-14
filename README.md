# CRFS oracle harness on VLSA-Aegis

This repository is a baseline-first research harness for testing Counterfactual Residual Flow Steering (CRFS) in π0.5 on SafeLIBERO. The Git history and implementation start from [THU-RCSCT/VLSA-Aegis](https://github.com/THU-RCSCT/vlsa-aegis) commit `57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b`. CRFS is an additive experiment layer, not a replacement benchmark or a clean-room reimplementation.

The original oracle question was deliberately narrower than learning a safety probe:

> Given the exact nearest safe, endpoint-preserving correction for a colliding five-action prefix, can an intervention inside the frozen π0.5 action flow rescue the same observation–noise sample more often than an equal-norm random direction?

H05 stopped before that intervention: exact endpoint preservation made all 20
five-action and all 20 ten-action repairs analytically infeasible. The active
R00/R01 pivot now asks whether the same pre-grasp reach states admit any
endpoint-free action that preserves calibrated task progress. A learned probe
remains forbidden until endpoint-free oracle intervention and analysis pass.

## Baseline and extensions

The released baseline remains in its original locations:

- `main/main_aegis.py` and `main/main_aegis_translational.py`: AEGIS evaluation runners;
- `safelibero/`: SafeLIBERO task definitions, initial states, objects, and environment wrappers;
- `openpi/`: the baseline OpenPI fork and π0.5 policy stack.

The harness adds opt-in seams:

1. the PyTorch sampler can expose the midpoint trace and apply either distributed residual velocity or a one-shot bridge edit;
2. policy requests may carry a reserved `__crfs__` control envelope with fixed noise and a physical or normalized correction;
3. SafeLIBERO can call back after each hidden MuJoCo physics substep without changing ordinary `env.step`;
4. `main/crfs_oracle/` runs same-state/same-noise nominal, direct-repair, equal-norm random, oracle-residual, and bridge-edit branches.
5. the R00/R01 path separately measures five-action reach progress and searches
   endpoint-free candidates while keeping `D_opt` and direct `D_sim` verification distinct.

No-control sampler calls retain the baseline return type and integration path.

## Evidence ladder

The completed CRFS sequence is H00–H10; the endpoint-free pivot is R00–R04 in
`feature_list.json`. R00 is the only active gate.

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

## One-case real experiment

Real execution uses the existing two-environment baseline boundary: Python 3.11/PyTorch for the OpenPI server and the LIBERO client environment for simulation. Both processes run inside one Slurm allocation; the transient server is terminated by a shell trap.

```bash
scripts/hpc/preflight.sh
scripts/hpc/submit_reach_progress_smoke.sh manifests/reach_progress_calibration.jsonl
```

Do not submit a validation array until:

- the converted PyTorch checkpoint has a recorded hash and parity evidence;
- deterministic policy and simulator replay pass;
- the substep geometry/contact audit passes;
- one `reach-calibration.json` validates;
- the preliminary scientific deviations are reviewed.

The detailed procedure is in [the experiment protocol](docs/experiment_protocol.md) and [VinUni runbook](docs/infrastructure/vinuni_h100_runbook.md).

## Research materials

- `main.tex`: proposal and preregistered staged tests;
- `Chat - Probe in Diffusion Model.md`: design discussion and corrections;
- `references.bib`: bibliography;
- `CRFS_Proposal_Preview.pdf`: compiled preview.

The proposal’s reported safety measure is minimum controlled simulator clearance
over every physics substep, cross-checked against contact pairs. R00/R01 also
measure maximum target and active-obstacle displacement over those substeps;
endpoint-only motion does not satisfy the no-pushing gate.
