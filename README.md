# CRFS oracle harness on VLSA-Aegis

This repository is a baseline-first research harness for testing Counterfactual Residual Flow Steering (CRFS) in π0.5 on SafeLIBERO. The Git history and implementation start from [THU-RCSCT/VLSA-Aegis](https://github.com/THU-RCSCT/vlsa-aegis) commit `57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b`. CRFS is an additive experiment layer, not a replacement benchmark or a clean-room reimplementation.

The immediate question is deliberately narrower than learning a safety probe:

> Given the exact nearest safe, endpoint-preserving correction for a colliding five-action prefix, can an intervention inside the frozen π0.5 action flow rescue the same observation–noise sample more often than an equal-norm random direction?

If the oracle direction is not causal, the project stops before training a predictor.

## Baseline and extensions

The released baseline remains in its original locations:

- `main/main_aegis.py` and `main/main_aegis_translational.py`: AEGIS evaluation runners;
- `safelibero/`: SafeLIBERO task definitions, initial states, objects, and environment wrappers;
- `openpi/`: the baseline OpenPI fork and π0.5 policy stack.

CRFS adds four opt-in seams:

1. the PyTorch sampler can expose the midpoint trace and apply either distributed residual velocity or a one-shot bridge edit;
2. policy requests may carry a reserved `__crfs__` control envelope with fixed noise and a physical or normalized correction;
3. SafeLIBERO can call back after each hidden MuJoCo physics substep without changing ordinary `env.step`;
4. `main/crfs_oracle/` runs same-state/same-noise nominal, direct-repair, equal-norm random, oracle-residual, and bridge-edit branches.

No-control sampler calls retain the baseline return type and integration path.

## Evidence ladder

The harness is organized into gates H00–H10 in `feature_list.json`. The current gate is H03, allocation-backed measurement audit.

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
```

The dependency-free synthetic fixture checks artifact and intervention plumbing only. Real artifacts currently use `real_safelibero_preliminary` because two preregistered conditions remain unresolved: the optimizer is not yet independent of `D_sim`, and released SafeLIBERO obstacles are movable rather than a static asymmetric-convex controlled pilot.

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
scripts/hpc/submit_oracle_smoke.sh manifests/oracle_smoke.jsonl
```

Do not submit a validation array until:

- the converted PyTorch checkpoint has a recorded hash and parity evidence;
- deterministic policy and simulator replay pass;
- the substep geometry/contact audit passes;
- one `results.json` validates;
- the preliminary scientific deviations are reviewed.

The detailed procedure is in [the experiment protocol](docs/experiment_protocol.md) and [VinUni runbook](docs/infrastructure/vinuni_h100_runbook.md).

## Research materials

- `main.tex`: proposal and preregistered staged tests;
- `Chat - Probe in Diffusion Model.md`: design discussion and corrections;
- `references.bib`: bibliography;
- `CRFS_Proposal_Preview.pdf`: compiled preview.

The proposal’s reported safety measure is minimum simulator geom clearance over every physics substep, cross-checked against contact pairs. The released obstacle-displacement flag remains useful only as a later benchmark-comparability metric.
