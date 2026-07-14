# R01 endpoint-free reach-feasibility evidence

`r01-summary.json` is the passed allocation-backed R01 verifier artifact.

- Full frozen population: Slurm array `27306`, 20/20 cases, zero task failures
- Verifier: Slurm job `27364`, exit code 0
- Registered gate: at least 12/20 changed-action safe-progress witnesses
- Result: 17/20 witnesses; all 20 nominal collisions reproduced
- Artifact SHA-256: `715d9326f02df531e0c6d7aa7b94629569b320800b4283c36fff41335d9b8bc5`
- Ordered raw-result digest: `6cc9bcf435dbe06396b90e34a0f4930994039538a6cc6ecad82876a538513140`
- Remote raw artifacts: `/mnt/data/quanth/experiments/crfs-oracle/r01-endpoint-free-population-20260714a`

The three negative cases began below the immutable 5 mm branch margin; one was
already in penetration. All 17 positive witness corrections were outside the
H04 local calibration domain and saturated at least one translation component.
R01 therefore establishes controlled physical existence only. It does not show
that the frozen flow can reach the witnesses, and it is not transport or
full-arm safety evidence.
