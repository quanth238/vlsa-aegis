# R06 evidence

- `groundingdino-setup-28391.json`: allocation-backed dependency setup only.
  Slurm job 28391 downloaded and content-bound the official GroundingDINO
  Swin-T checkpoint. It contains no simulator execution and no AEGIS efficacy
  result.
- `aegis-label-capture-canary-20260717a.json`: exact terminal evidence for
  failed capture-only task `28409_0`. It records the controller-state
  generator/validator mismatch and forbids using the partial assets as a
  label or AEGIS result.

No paired canary or population result exists yet.
