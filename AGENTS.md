# CRFS oracle harness rules

This repository starts from THU-RCSCT/VLSA-Aegis. Preserve the baseline and make CRFS behavior opt-in.

## Startup

1. Confirm the worktree and branch with `pwd`, `git status --short`, and `git branch --show-current`.
2. Read `README.md`, `PROGRESS.md`, `DECISIONS.md`, and the active item in `feature_list.json`.
3. Run `./init.sh` before changing code.
4. Work on at most one `active` gate. A later gate cannot pass before its dependencies.

## Scientific invariants

- Pair every comparison by simulator initial state, observation, policy noise, and executed action horizon.
- Preserve episode/state groups across train, validation, and test splits.
- Keep optimizer clearance (`D_opt`) distinct from simulator clearance/contact verification (`D_sim`).
- Convert physical displacement to normalized model coordinates with scale only; never subtract the normalization mean from a displacement.
- Endpoint preservation for the five-step translation pilot means the correction sums to zero across time.
- Keep distributed residual velocity and one-shot bridge editing as distinct experiment arms.
- Report infeasible repairs and failed cases; never silently drop them.
- Synthetic results are implementation evidence only. They cannot support a research claim.
- Do not train a learned steering model before the oracle intervention and oracle analysis gates pass.

## Baseline compatibility

- `main/main_aegis*.py`, `safelibero/`, and `openpi/` are baseline components.
- The default OpenPI sampler call must return the same tensor and execute the same Euler updates as the baseline.
- CRFS WebSocket controls live in the reserved `__crfs__` envelope and are removed before normal transforms.
- New simulator measurement APIs are additive; the ordinary `env.step` path remains unchanged.
- Every baseline modification needs a regression or structural test.

## Artifact contract

- Case identities and seeds come from an immutable JSONL manifest.
- `results.json` is written atomically only after validation-ready output is complete.
- Resume skips only a schema-valid final artifact.
- Record code and dirty state, baseline revision, model/checkpoint hash, data/task/state identity, all seeds, sampler/intervention settings, action frame, normalization space, simulator measurement, optimizer status, Slurm IDs, host, and device.
- Aggregate feasible-conditioned and all-collision populations separately.
- Bootstrap complete state/episode groups, not individual frames.

## VinUni H100 constraints

- The login node is control plane only. Never run Python experiments, CUDA, rendering, metrics, training, or model serving there.
- Submit compute through Slurm. A policy server is allowed only inside its allocation and must be terminated by a trap.
- Run live preflight before every significant job. Live Slurm/QOS/storage state overrides checked-in examples.
- User ceiling: two GPU-equivalents, 16 CPUs, and 256 GB RAM. Smoke concurrency is `%1`; full-H100 arrays are at most `%2`.
- Do not target `DOWN`, `DRAIN`, or `NOT_RESPONDING` nodes.
- Use `/home/quanth/working_space` for source and `/mnt/data/quanth/{experiments,slurm_logs,cache}` for artifacts.
- Do not bulk-transfer datasets/results over the login connection. Source sync must exclude outputs, checkpoints, caches, and secrets.
- Never use broad `scancel`, `pkill`, `rm -rf`, `du`, or `rsync --delete` on data/output paths.
- Inspect an exact job before canceling it. Cleanup is print-first and requires explicit authorization.

## Verification and state transitions

- `./init.sh` is the local gate.
- Real-experiment gates require allocation-backed evidence and validated artifacts in addition to local tests.
- Only a verifier may change a feature to `passing`; record the exact evidence in `feature_list.json` and `PROGRESS.md`.
- Before handoff, update progress, decisions, active status, unresolved risks, and exact next command.
