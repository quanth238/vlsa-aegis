# H05 projection-teacher evidence

Decision: **stop CRFS at H05 and pivot the research formulation**.

The frozen population contains all 20 nominally colliding states from the H03 manifest. Every admissible correction preserves the frozen H04 translational endpoint by construction. For each case, the endpoint itself violates the requested 5 mm safety margin, which is an analytic infeasibility certificate because swept clearance cannot exceed endpoint clearance.

- H=5: 20/20 certified infeasible, `F_proj = 0.00`; endpoint-clearance upper bounds ranged from -48.221 mm to -7.723 mm. Final restartable array submission: Slurm `27149`; raw root: `/mnt/data/quanth/experiments/crfs-oracle/h05-population-20260714a`.
- H=10 registered refinement: 20/20 certified infeasible, `F_proj = 0.00`; endpoint-clearance upper bounds ranged from -75.234 mm to -44.895 mm. Slurm array `27191`; raw root: `/mnt/data/quanth/experiments/crfs-oracle/h05-h10-population-20260714b`.
- H=5 ordered result-set digest: `1495d6910228a840ac9142474ae7ddc0c8d643b75bbf9cf8a3ef1ab3f32c643b`.
- H=10 ordered result-set digest: `af52eb6ccf09ff67f931d2582dd0d8590b8b96c05c8a96afae37e4ac85e70357`.
- Checkpoint SHA-256: `988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed`.

The earlier one-case SLSQP runs are retained as diagnostics, but they are not the basis of this decision. The endpoint certificate removes optimizer convergence as an explanation. Since the preregistered pass threshold is 0.60 and the only permitted H=10 refinement also yields zero feasibility, H06--H09 were not run. No learned probe should be trained from this label construction.
