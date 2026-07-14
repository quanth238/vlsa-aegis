# 0007 — Require substep scene motion and transport-neutral config hashes

Status: accepted for R00/R01 on 2026-07-14.

## Context

The first full R00 apparatus run completed 120/120 cases, but its summary
correctly failed because the scientific config hash included each array task's
WebSocket port. The two ports created two hashes for otherwise identical code,
checkpoint, manifest, and experiment settings.

The same audit found that target and obstacle stationarity used only their
positions after action five. Endpoint displacement cannot exclude transient
motion beyond the 1 mm tolerance followed by rebound.

## Decision

Host, port, output root, and run ID are runtime routing fields. Exclude them
from the scientific config identity, while continuing to record allocation,
host, run, code, checkpoint, manifest, case, and seed provenance separately.

For R00 and R01, track the target and active obstacle directly from MuJoCo after
all 125 physics substeps. Eligibility and witness verification use maximum
branch-relative displacement, while endpoint displacement remains recorded as
a secondary diagnostic. Each rollout must also reproduce the branch EEF,
target, and obstacle positions within 1e-9 m.

## Consequences

- Slurm array transport choices cannot split a scientific configuration.
- The completed `r00-calibration-20260714a` run is apparatus evidence only.
- R00 must be rerun before freezing `p_min` or activating R01.
- Default `SafeLiberoCase.rollout(actions)` output remains unchanged; body
  tracking is opt-in for reach diagnostics.
