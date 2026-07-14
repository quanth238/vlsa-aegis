# VinUni H100 runbook for CRFS

This project-specific runbook follows the local VinUni H100 guide. Live cluster output always overrides checked-in examples.

## Boundaries

The login node is for SSH, bounded source synchronization, Slurm submission, and read-only inspection. Never run Python inference, CUDA checks, MuJoCo rendering, aggregation, training, or a policy service there. The WebSocket policy server is transient and may run only inside the same Slurm allocation as its client; `run_oracle_case.sh` terminates it with a trap.

The user ceiling observed on 2026-07-14 was two GPU-equivalents, 16 CPUs, and 256 GB RAM. MIG and full-H100 availability changes. Preflight every significant run. Do not target drained or non-responding nodes.

## Paths

```text
source       /home/quanth/working_space/vlsa-aegis-crfs
experiments  /mnt/data/quanth/experiments/crfs-oracle
Slurm logs   /mnt/data/quanth/slurm_logs/crfs-oracle
caches       /mnt/data/quanth/cache
OpenPI env   /mnt/data/quanth/venvs/openpi
LIBERO env   /mnt/data/quanth/venvs/openpi-libero-client
```

Large data and results do not move through the shared login/VPN connection. Use the approved SFTP path for bulk transfer. Never `rsync --delete` a dataset or output tree.

## Preflight

From the local repository:

```bash
scripts/hpc/preflight.sh
```

It records host, jobs, partitions/nodes, QOS, storage, and possible stray processes in `.harness/live`. It is read-only and never cancels or submits.

Fail closed if the host is unexpected, the resource budget is occupied, intended nodes are unavailable, storage is too full, paths/checkpoint/source are missing, or a login-node compute/service process is present.

## Checkpoint conversion

The public cached π0.5 LIBERO checkpoint is JAX. The CRFS trace path is PyTorch. Convert once in a Slurm allocation, save outside home, and hash `model.safetensors`. Keep the source checkpoint immutable. Record conversion source commit, command, output hash, and a JAX/PyTorch fixed-input parity check before interpreting an oracle result.

## Staged run

1. Local: `./init.sh` and `make synthetic`.
2. Allocation import smoke: policy/client environments, CUDA, MuJoCo, robosuite, SafeLIBERO paths.
3. One manifest case on MIG with concurrency one if memory fits.
4. Validate atomic result, deterministic replay, peak memory, server cleanup, and bounded logs.
5. Only then use a small validation array. Full-H100 array concurrency is at most `%2`; begin at `%1`.
6. Test configuration remains frozen and unavailable for tuning.

Long-run templates omit a wall-time unless current policy requires one. The one-case MIG diagnostic uses a bounded time because it is a smoke, not a sweep.

## Monitoring and cancellation

Use `squeue`, exact `scontrol show job JOBID`, `sacct`, `tail -n 200`, and project-scoped completion counts. Pending `Priority`, `Resources`, or `JobArrayTaskLimit` is not an implementation failure.

Inspect an exact job before `scancel JOBID`. Never cancel all user jobs. Never broad-`pkill`. Cleanup scripts print candidates only; deletion requires separate explicit authorization.

## Completion

A case is complete only when `results.json` exists, parses, and passes the harness validator. Partial server/client logs do not count. Reruns skip only valid completion artifacts. Failures retain bounded logs and a structured failure JSON.
