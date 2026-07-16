# 0052 — Preserve AF-00A launch A and normalize VinUni array task identity

Status: accepted apparatus diagnosis and narrowly scoped repair on 2026-07-16.
This decision does not authorize another H100 submission.

## Terminal launch-A evidence

Release `6ff5d5cb851bc983b7873be0b120df173525266e` reserved immutable run
`r05a-actual-forward-cem-canary-20260716a` and submitted exact held task
`28275_0`. VinUni accepted the requested worker-1 resources: one H100, eight
CPUs, 64 GiB host RAM, two hours, array task zero with throttle one, and no
requeue.

The submitter then queried `scontrol show job 28275_0 -o`. On this Slurm
installation the exact task record reported `JobId=28275`, together with the
unambiguous tuple `ArrayJobId=28275 ArrayTaskId=0`. The preregistered check
required the alternate display form `JobId=28275_0`, so the transaction
stopped while the GPU task was still user-held. It wrote only the launch
reservation, live preflight, and provisional GPU-ID receipt. It did not create
the CPU publisher, release the task, allocate worker-1, start Python or pi0.5,
or execute a model or simulator action. After exact inspection, task
`28275_0` was cancelled under the user's explicit cancellation authorization;
Slurm records `CANCELLED`, exit `0:0`, elapsed `00:00:00`, and no node.

This consumed run is apparatus-inconclusive. It contains no Arm A, CEM, Arm B,
Arm C, transport, safety, progress, efficacy, or infeasibility result.
The compact terminal record is
`evidence/r05a/af00a-actual-forward-launch-a.json`.

## Narrow repair

For records returned by the exact query `scontrol show job ${array}_0 -o`,
accept the displayed `JobId` only when it is either `${array}` or
`${array}_0`. Continue to require all of the following independently:

- `ArrayJobId=${array}` and `ArrayTaskId=0` on the exact task record;
- the parent task range `0` or `0%1` and `ArrayTaskThrottle=1`;
- user-held state before release and running state inside the allocation;
- exact node, partition, account, QOS, time, CPU, memory, GPU, requeue, and
  ReqTRES fields;
- the complete receipt/fingerprint chain and one release only.

This normalizes only Slurm's display spelling. It does not weaken exact task
identity and does not change any AF-00A scientific value, model path, query,
budget, objective, tolerance, selection rule, action, or resource.

## Release boundary

Return the config to fail closed. After focused and repository gates plus an
independent review pass, a direct-child release may change only the config and
`docs/decisions/0053-release-corrected-actual-forward-cem-canary.md`, select a
fresh unused immutable run ID, and authorize one submission. No automatic
retry, IFT-01, probe, or MLP is authorized here.
