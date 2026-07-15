# 0029 — Allow the source-pinned R05A canary to wait in Slurm

Status: accepted on 2026-07-15

## Context

IFT-00A was ready at a clean synchronized commit, the exact run ID was unused,
the user queue was empty, worker-1 was healthy, and more than 64 GiB host RAM
was free. All eight worker-1 H100s were allocated to other users. The first
launcher additionally required one H100 to be idle before it would submit.

That idle-at-submission requirement is operational, not scientific. Slurm
enforces the frozen worker-1, one-H100, eight-CPU, and 64-GiB request when the
allocation starts. A control-plane-only `sbatch --test-only` accepted the exact
request while worker-1 was fully allocated and predicted a later start. It did
not create a job.

## Decision

1. Permit the exact canary to enter Slurm in `PENDING` state when worker-1 has
   zero immediately free H100s.
2. Retain every existing fail-closed gate: clean identical local/remote commit,
   immutable unused run ID, empty user queue, healthy worker-1, at least 64 GiB
   reported FreeMem, exact H100 GRES, frozen input hashes, source trace and
   checkpoint hashes, one H100, eight CPUs, 64 GiB, `%1`, and no requeue.
3. Validate that allocated GPU count is between zero and configured GPU count.
   Record the observed count and whether immediate capacity existed. Do not call
   a fully allocated node "capacity verified."
4. Preserve the atomic launch sequence: submit the exact GPU job held, persist
   its ID, register and verify the CPU `afterany` validator, persist both IDs,
   then release only that GPU job.
5. Keep the job pinned to worker-1. Do not reroute, lower resources, change a
   scientific config, tune the solver, execute actions on the login node, or
   authorize IFT-01 or learned training.

## Consequences

The scheduler, rather than a timing loop on the login node, waits for capacity.
Queue delay has no research meaning. IFT-00A remains apparatus-only and produces
no scientific result until the exact allocation and its independent validator
finish and their artifacts pass semantic inspection.
