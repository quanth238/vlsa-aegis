# 0058 — Accept the AF-00A frozen-CEM negative result

Status: accepted terminal interpretation on 2026-07-16. This decision closes
the consumed AF-00A and publication-recovery releases. It does not authorize a
retry, a changed solver, IFT-01, simulator execution, or training.

## Terminal publication evidence

ADR-0057 released exactly one CPU-only republication for immutable source run
`r05a-actual-forward-cem-canary-20260716c`. Recovery publisher `28291`
completed `0:0` on worker-0 in `00:00:08`; Slurm requested and allocated two
CPUs and 8 GiB host RAM with no GPU. Its exact Slurm log SHA-256 is
`47b768bda62f801a16ca3fc5a224c9ae18396411238dd67a1d3e1d9f15204131`.

The CPU job published schema-valid `results.json` with SHA-256
`507f25bc381b7bfeaa30abe3a7df970681f3b175f39221034c6e768e72dae914`
and a separate recovery receipt with SHA-256
`5462f875ffb41a84a06c4df715366776af3811f3229385d05be8f5b0dafcb632`.
The receipt reports `passed: true`, `published: true`, and outcome
`frozen_cem_negative`. The original failed receipt remains byte-exact at
`f8e9ef108630cad9a9691dba0827a8be68d3b15e41ee3de2e4e6cfaf03736d27`.

The recovery source-contract, submission, and final-fingerprint SHA-256 values
are respectively
`249087a42023d73e65589158bc8be5407605b57773b81a2c1d6138c13a820eab`,
`68a6cd3a08c4d502fb97f0a0db18afcd88fe23935e6c2bf5c2d4d807204e4c7c`,
and `1fa71d24d37b1ee6d7d19e130f29c9d3de55f82967b8e648fd3981a932eaa921`.
They separately bind source commit
`bd14f97eeffafd20525454db4d7a52614e1146c4`, recovery commit
`eb3be3a86c1336a9090413cf7d6c83f28a0f365c`, source task `28281_0`,
failed publisher `28282`, and recovery publisher `28291`.

The terminal interpretation passed all 49 dependency-backed AF-focused tests
and the complete local gate with 687 tests passed and 225 expected dependency
skips.

## Scientific result

All 534 registered ordinary sampler requests, 520 fixed CEM evaluations,
duplicate schedules, canonical replays, target construction, float32 budget,
constraints, source pairing, telemetry, and 36 allocation tests with zero
skips validate. No policy- or teacher-generated action was executed in the
simulator.

- Equal-split Arm A failed: objective `2218.1335502517986`, XYZ max/RMS
  `0.7806643492412315` / `0.35969860072305654`.
- Selected changed Arm B failed: objective `2121.5404274315442`, XYZ max/RMS
  `0.7899218065691934` / `0.35177249996585547`, selected pool index `465`
  and CEM query `464`. Its full-action max/RMS were
  `0.7899218065691934` / `0.23039493162555724`.
- Reversed Arm C failed and was worse: objective `2968.7447114541746`, XYZ
  max/RMS `0.9606940421772645` / `0.416131221925226`.

Arm B reduced the scalar objective by only about 4.35% from Arm A and still
missed every physical fidelity gate by a large margin. The registered XYZ
max/RMS limits are `0.010` / `0.005`; the full-action max/RMS limits are
`0.050` / `0.015`.

## Interpretation and root problem

The publication bug is resolved and is not the scientific failure. The
scientific failure is that the best schedule found inside the frozen
five-flow-step, 75-coordinate, same-budget control family did not reproduce
the privileged action target closely enough. Reversing timing did not rescue
it.

This result cannot distinguish two remaining causes: the target may lie
outside the reachable image of this budgeted residual-schedule family, or the
fixed 520-query diagonal CEM may be an inadequate optimizer. Therefore it
rejects only this exact one-case CEM teacher. It is not an infeasibility
certificate and does not reject every possible action-to-vector-field
transport method.

## Decision and next research boundary

1. Close both consumed configs and forbid automatic AF-00A resubmission,
   solver/tolerance/budget tuning, or a GPU rerun.
2. Do not launch IFT-01, execute the generated action in the simulator, or
   train a probe or residual-field MLP. A student cannot be justified before a
   teacher can reproduce the privileged target.
3. Keep the broader transport hypothesis open only as a focused feasibility
   question. Before another H100 experiment, use the already sealed 520-query
   input/output population only to diagnose evidence bearing on weak
   optimization versus weak controllability; it cannot prove that distinction
   by itself. Then preregister a materially different teacher and an unchanged
   one-case target-fidelity gate.
4. Only a later teacher that passes the same action-target fidelity boundary
   may justify a separately reviewed simulator efficacy test. Safety,
   progress, population, novelty, latency, and learning claims remain blocked.

R05A remains active as an unresolved transport-feasibility study; AF-00A
itself is terminal and negative.
