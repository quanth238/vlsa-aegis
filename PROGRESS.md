# Progress

Last updated: 2026-07-14 (Asia/Ho_Chi_Minh)

Active branch: `agent/crfs-oracle-harness`

Baseline: THU-RCSCT/VLSA-Aegis commit `57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b`

## Verified

- The baseline repository, including SafeLIBERO and its OpenPI fork, is the Git worktree foundation.
- The ordinary policy and environment entry points remain available; CRFS controls are opt-in.
- Local dependency-free unit suite passes, including geometry, paired artifacts, restart semantics, and synthetic end-to-end flow.
- Live VinUni access, Slurm/QOS, storage, and existing OpenPI environments were inspected without login-node compute.
- A fork exists at `quanth238/vlsa-aegis`; the working branch has not yet been pushed.

## Active gate

`H03-measurement-audit`: verify the new 25-substep callback and geom-distance/contact monitor inside a real SafeLIBERO allocation.

## Next

1. Convert the cached JAX `pi05_libero` checkpoint to PyTorch in a Slurm allocation and hash `model.safetensors`.
2. Run one MIG allocation-backed environment/import smoke.
3. Run one paired oracle case with fixed state and noise.
4. Audit the result schema and limitations before deciding whether validation-scale experiments are justified.

## Open scientific risks

- The first runner uses simulator geometry during optimization, so it does not yet establish an independent `D_opt` versus `D_sim` audit.
- Released SafeLIBERO obstacles are movable; the proposal preregisters a static asymmetric-convex controlled pilot.
- A one-case smoke cannot pass the population-level oracle gate; it only validates the causal experiment apparatus.
- Exact PyTorch conversion parity with the public JAX checkpoint must be measured before interpreting outcomes.
