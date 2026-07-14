# Progress

Last updated: 2026-07-14 (Asia/Ho_Chi_Minh)

Active branch: `agent/crfs-oracle-harness`

Baseline: THU-RCSCT/VLSA-Aegis commit `57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b`

## Verified

- The baseline repository, including SafeLIBERO and its OpenPI fork, is the Git worktree foundation.
- The ordinary policy and environment entry points remain available; CRFS controls are opt-in.
- Local dependency-free unit suite passes, including geometry, paired artifacts, restart semantics, and synthetic end-to-end flow.
- Live VinUni access, Slurm/QOS, storage, and existing OpenPI environments were inspected without login-node compute.
- Commit `2bbd53a` is pushed to `quanth238/vlsa-aegis:agent/crfs-oracle-harness`.
- Slurm job `27077` reached the MIG worker and failed cleanly before conversion because the existing OpenPI environment lacked its patched Transformers modules. The final checkpoint path was not created; the diagnostic `.incomplete-27077` path is retained.
- Slurm job `27079` completed conversion in 1m57s. `model.safetensors` is 6.8 GB with SHA-256 `988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed`; normalization assets are present.
- Slurm job `27082` loaded the converted policy and opened the WebSocket server, then failed before simulation because SafeLIBERO attempted an interactive first-import config prompt. No research result was written; server cleanup succeeded.
- Slurm job `27083` passed noninteractive SafeLIBERO import and began the paired run. It exposed `hard_reset=True` recompiling the MuJoCo scene for every branch; after exact-job inspection and confirming no result existed, job `27083` alone was canceled. The runner now uses soft reset plus fixed-state restoration and emits structured progress events.
- Slurm job `27091` isolated a false raw `mj_geomDistance` signal to the `gripper0_hand_collision` mesh versus an obstacle box. The raw query was negative while the collision-enabled pair had no contact, so projection was stopped before optimization.
- Slurm job `27096` calibrated the controlled sphere--box primitive on 261 states: zero boundary error over 23 samples within $\pm1$ cm, zero sign disagreements, and first contact at -0.603 mm for a 1 mm sweep step.
- Slurm jobs `27116` and `27117` completed the strict H03 manifest: 50 unique saved states, five identical replays per state, 126 measurements per replay, zero clearance variation, zero endpoint-coordinate variation, and no conservative-proxy contact miss.
- H03 population composition was 20 proxy-colliding, 30 proxy-safe, and 4 physical-contact cases. Sixteen cases reproduced the raw mesh-query inconsistency, which is now advisory-only evidence rather than the primary safety signal.
- The H03 summary and minimum-pair visualization are committed under `evidence/h03/`; raw allocation artifacts remain under `/mnt/data/quanth/experiments/crfs-oracle/h03-population-20260714a`.
- Slurm job `27123` passed H04 on 10 calibration states and 20 held-out prefixes: median endpoint error 0.971 mm, 95th-percentile absolute `D_opt`/`D_sim` error 1.082 mm, and zero false-safe predictions at a 10 mm margin.
- H05 added an optimizer-independent endpoint certificate: zero-sum translation fixes the H04 endpoint, so an unsafe endpoint proves the swept-path constraint infeasible before optimization.
- The H=5 population run completed all 20 H03-colliding states: 20/20 certified infeasible, `F_proj=0.00`, with endpoint-clearance upper bounds from -48.221 mm to -7.723 mm.
- The one registered H=10 refinement also completed all 20 states: 20/20 certified infeasible, `F_proj=0.00`, with endpoint-clearance upper bounds from -75.234 mm to -44.895 mm.
- The preregistered H05 stop rule fired. H06--H09 were not run, and no learned probe was trained.
- R00 apparatus run `r00-calibration-20260714a` completed all 120 cases on Slurm array `27276` with no case failures. Its allocation-backed summary job `27285` correctly refused promotion because the scientific config hash included per-array WebSocket ports. This run also measured scene motion only at the endpoint, so it remains apparatus evidence rather than calibration evidence.
- R00 now excludes host, port, output root, and run ID from the scientific config identity; it records immutable manifest/case hashes, verifies exact reset branches, and measures target/obstacle maximum displacement over all 125 physics substeps. A fresh allocation-backed R00 run is required.

## Active gate

R00 — freeze a five-action pre-grasp reach-progress threshold on baseline-safe calibration groups. The frozen H05 cases are initial reach states, not transport states; `akita_black_bowl_1` is the fixed branch-start target.

## Next

1. Push and sync the strict R00 apparatus, then run `RUN_ID=r00-calibration-20260714b scripts/hpc/submit_reach_progress_calibration.sh manifests/reach_progress_calibration.jsonl 60` and summarize it in Slurm.
2. Run endpoint-free five-action planning on the same 20 H05 cases and count only direct simulator-verified safe-progress witnesses.
3. If at least 12/20 witnesses exist, test endpoint-free oracle flow steerability against equal-norm random guidance before training any probe.
4. Collect new post-grasp states if a separate transport-phase claim is pursued.

## Open scientific risks

- Raw MuJoCo mesh--box `mj_geomDistance` is unreliable in this stack. The primary controlled metric is the proposal's EEF-sphere/known-box signed distance evaluated from simulator transforms; physical contacts remain a one-way conservatism check.
- Released SafeLIBERO obstacles are movable; the proposal preregisters a static asymmetric-convex controlled pilot.
- The endpoint certificate is relative to the frozen H04 linear response and static branch geometry; it rejects this registered local-equivalence formulation, not all possible collision-avoidance planners.
- Exact PyTorch conversion parity with the public JAX checkpoint must be measured before interpreting outcomes.
- The 30 calibration groups may require multiple fixed policy-noise samples to yield 50 valid safe-progress chunks; all inference must continue to preserve complete episode groups.
- A bounded endpoint-free search miss is not an infeasibility certificate. Only a simulator-verified witness proves existence.
