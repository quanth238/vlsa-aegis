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

## Active gate

`H03-measurement-audit`: verify the new 25-substep callback and geom-distance/contact monitor inside a real SafeLIBERO allocation.

## Next

1. Re-run the one-case smoke with an explicit noninteractive SafeLIBERO path config.
2. Run one paired oracle case with fixed state and noise.
3. Audit the result schema and limitations before deciding whether validation-scale experiments are justified.

## Open scientific risks

- The first runner uses simulator geometry during optimization, so it does not yet establish an independent `D_opt` versus `D_sim` audit.
- Released SafeLIBERO obstacles are movable; the proposal preregisters a static asymmetric-convex controlled pilot.
- A one-case smoke cannot pass the population-level oracle gate; it only validates the causal experiment apparatus.
- Exact PyTorch conversion parity with the public JAX checkpoint must be measured before interpreting outcomes.
