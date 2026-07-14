# R00 reach-progress calibration

`r00-summary.json` is the passed allocation-backed R00 verifier artifact.

- Strict smoke: Slurm array task `27291_0`
- Full calibration: Slurm array `27292`, 120/120 cases, zero batch failures
- Verifier: Slurm job `27298`, exit code 0
- Frozen threshold: `p_min = 0.029897349105658888 m`
- Artifact SHA-256: `90fe09e075b30e680e5cfbd81c802afa03c94b58ba7f0e95e66080c61af2096f`
- Remote raw artifacts: `/mnt/data/quanth/experiments/crfs-oracle/r00-calibration-20260714b`

The threshold is the registered inverted-CDF lower quartile of positive five-action
pre-grasp reach progress. Calibration uses 30 complete groups disjoint from the 20
frozen R01 evaluation groups. This is reach evidence, not transport evidence.
