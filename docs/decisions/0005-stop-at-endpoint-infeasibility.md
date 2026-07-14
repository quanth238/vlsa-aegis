# 0005 — Stop at endpoint infeasibility

Status: accepted, 2026-07-14.

## Decision

Do not advance the current CRFS formulation to direct-replay, sampler, oracle-flow, or learned-probe experiments. Pivot to an earlier or longer-horizon planner, or replace exact local endpoint equivalence with a task-level preservation constraint.

## Evidence

On the frozen 20-case colliding population, both H=5 and the single permitted H=10 refinement produced 20/20 analytic infeasibility certificates. Exact zero-sum translation fixes the H04 endpoint, and that endpoint violates the 5 mm safety margin in every case. Thus `F_proj=0.00` at both horizons, below the registered 0.60 gate.

## Consequence

Flow steering cannot be meaningfully evaluated because no direct safe teacher action exists under the current label definition. Adding network capacity or tuning the intervention would hide the failed premise rather than test it.
