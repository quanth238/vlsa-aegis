# R02/R03 oracle-flow evidence

`r03-summary.json` is the passed allocation-backed R03 verifier artifact.

- Clean-commit smoke: Slurm job `27404`, exit code 0
- Full H100 population: Slurm array `27405`, 20/20 tasks, all exit code 0
- CPU verifier: Slurm job `27450`, exit code 0
- Feasible-conditioned result: oracle 9/17, random 0/17, analytic 0/17
- Oracle-minus-random grouped 95% bootstrap LCB: 5/17 = 0.2941176471
- Oracle-minus-analytic exact one-sided paired test: p = 1/512 = 0.001953125
- Original-population result: oracle 9/20, direct witness 17/20
- Artifact SHA-256: `dea3e66c854faa0b659777bfed4b651b19d8d4d0710696ff7bfe52c44d4ee76e`
- Ordered result-set digest: `fa79a132f2fecbedab5917111ef71f022e52c933d8e535aaca118add5f5b7895`
- Remote raw artifacts: `/mnt/data/quanth/experiments/crfs-oracle/r02-oracle-flow-population-20260714a`

All 20 frozen collisions and all 17 applicable direct witnesses were
reconfirmed. The oracle result passes R03 and the stricter ADR-0012 conditions
authorize starting R04. This is not evidence that a learned ECG probe works.
It is a five-step pregrasp, single-task pilot under the controlled EEF-sphere
versus active-obstacle metric, not post-grasp transport or full-arm safety.

The eight oracle failures divide into four clearance-only and four
progress-only misses. The registered equal-L2 one-midpoint analytic arm passed
clearance but failed progress in all 17 cases; the registered `t=0.5` one-shot
bridge failed clearance in all 17. These patterns motivate a preregistered
timing/dose calibration and gradient-causality test, but do not isolate a
mechanism or establish that learning is necessary.
