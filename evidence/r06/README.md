# R06 evidence

- `groundingdino-setup-28391.json`: allocation-backed dependency setup only.
  Slurm job 28391 downloaded and content-bound the official GroundingDINO
  Swin-T checkpoint. It contains no simulator execution and no AEGIS efficacy
  result.
- `aegis-label-capture-canary-20260717a.json`: exact terminal evidence for
  failed capture-only task `28409_0`. It records the controller-state
  generator/validator mismatch and forbids using the partial assets as a
  label or AEGIS result.
- `aegis-label-capture-canary-20260717b.json`: validated terminal evidence for
  repaired capture-only task `28428_0`. It records exact boundary-20 pairing,
  two reproduced registered clearance violations, the live controller state,
  lossless image binding, and zero AEGIS/perception/QP/training execution.
- `../../manifests/r06_codex_obstacle_labels_canary.jsonl`: immutable one-row
  outcome-blind canary label ledger. Codex froze `red milk carton` only after
  retry-B capture validation and before any AEGIS outcome.
- `aegis-paired-launch-a.json`: exact terminal evidence for paired GPU task
  `28447_0` and CPU validator `28448`. The GPU stopped at dependency preflight
  because its client environment lacked CVXPY. It started no policy server,
  ran no simulator replay, and executed zero GroundingDINO, MVEE, QP, or AEGIS
  steps.

No paired AEGIS canary or population result exists yet. The failed first
launch is apparatus-inconclusive; neither it nor the valid capture/label shows
whether AEGIS prevents the collision.
