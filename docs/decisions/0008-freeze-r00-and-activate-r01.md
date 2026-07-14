# 0008 — Freeze R00 reach progress and activate R01

Status: accepted on 2026-07-14.

## Context

The first R00 apparatus run exposed transport-dependent configuration hashes and
endpoint-only scene-motion measurement. After those defects were fixed, strict
smoke array task `27291_0` passed and full Slurm array `27292` completed all 120
calibration cases without a batch failure.

Allocation-backed verifier `27298` accepted all 120 positive chunks from 30
calibration groups. These groups are disjoint from the 20 frozen H05 evaluation
groups. Target and active-obstacle motion were measured directly after all 125
MuJoCo substeps, and runtime routing fields did not affect scientific identity.

## Decision

Freeze the registered inverted-CDF lower-quartile reach threshold at

\[
p_{\min}=0.029897349105658888\ \mathrm{m}.
\]

Bind R01 to `evidence/r00/r00-summary.json` by SHA-256
`90fe09e075b30e680e5cfbd81c802afa03c94b58ba7f0e95e66080c61af2096f`, mark R00
passing, and make R01 the sole active gate.

## Consequences

- R01 may now test physical endpoint-free feasibility on the same frozen 20 H05
  pre-grasp cases.
- A counted repair must reproduce the nominal collision and provide a changed-action,
  repeated-`D_sim` witness with clearance at least 5 mm and progress at least
  `p_min`.
- The threshold does not establish transport progress or flow steerability.
- No learned clearance probe is authorized unless R01 and the later oracle-flow
  gates pass.
