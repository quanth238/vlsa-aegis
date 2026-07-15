# R03A one-case analytic diagnostic

R03A did not produce a population result. This directory records only the
accepted source-node smoke so the intentional ADR-0028 retirement cannot be
misread as missing or erased evidence.

- Slurm task: `27639_0`
- Remote run directory:
  `/mnt/data/quanth/experiments/crfs-oracle/r03a-analytic-kill-smoke-20260715e`
- Final artifact:
  `/mnt/data/quanth/experiments/crfs-oracle/r03a-analytic-kill-smoke-20260715e/crfs-1069f29a8d76463a/r03a-analytic-kill-test.json`
- Client log:
  `/mnt/data/quanth/experiments/crfs-oracle/r03a-analytic-kill-smoke-20260715e/case-0/r03a-client.log`
- Policy-server log:
  `/mnt/data/quanth/experiments/crfs-oracle/r03a-analytic-kill-smoke-20260715e/case-0/policy-server.log`
- Slurm log:
  `/mnt/data/quanth/slurm_logs/crfs-oracle/crfs-r03a-h100-smoke-27639_0.out`
- Source host: `worker-1`
- Clean code commit: `fd8871962e1424aa8d43a1d69a52e7663bdb4acf`
- Final artifact SHA-256:
  `c9fd2f362b487429cc6e896fce827a72d4ab67231f0e2fe018d686ca4943919d`
- Client log SHA-256:
  `0eff1190867d683af9826b4961921a9194993c9cc7a447f148d652f5beeb89cc`
- Policy-server log SHA-256:
  `5fab4cfa270e5e8514bae92270e2f15d1bcf1df06877d661a1d212ccb7122acd`
- Slurm log SHA-256:
  `cf99292abedd7b46dbdfb6d3462663d6409ffeae995053af15bbdafbdd0c225b`

The artifact passed independent semantic and Draft-2020-12 validation with
exact source/current trace pairing and exact duplicate policy/simulator
replay.

| Arm | Minimum clearance | Reach progress | Result |
|---|---:|---:|---|
| Frozen | -4.272 mm | 36.169 mm | Collision |
| Analytic trajectory, midpoint | 16.596 mm | 25.489 mm | Safe, but progress failure |
| Analytic trajectory, early | 14.749 mm | 26.713 mm | Safe, but progress failure |

The frozen required progress was 29.897 mm. Both analytic arms therefore count
as Safe-Progress failures. This is a valid paired 0/1 diagnostic only; it cannot
decide the registered 9/17 R03A population rule. ADR-0028 retires that population
unrun and preserves this analytic implementation as a future baseline.
