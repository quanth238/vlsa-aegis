# 0045 — Accept the runtime-identity regression and require a fresh CFS release

Status: accepted on 2026-07-16 from terminal Slurm job `28043`.

## Evidence

Exact zero-GPU release `8415b659a46699757de1e99558713e56b95255b5`
used immutable run ID `r05a-runtime-identity-regression-20260716a`.  Job
`28043` completed `0:0` on `worker-1` in one second.  Its held receipt,
source contract, atomic submission, result, and Slurm log have SHA-256 values
`30be8432...3b990`, `45656005...0fbc`, `bef280b1...d1d9`,
`3bda039c...b0de`, and `a8383717...8d1`, respectively.  All eight bound
repository files match the clean pushed release, and no failure artifact
exists.

Both frozen interpreter identities passed from inside the allocation without
invoking either interpreter.  OpenPI resolved to
`/mnt/data/quanth/anaconda3/bin/python3.11` with SHA-256
`c7171890...16bc9`; LIBERO resolved to the registered CPython 3.8.20 binary
with SHA-256 `c70efda0...4f62`.  The result records zero model, simulator,
rendering, metrics, training, or H100 execution.

The job requested one CPU and 256 MiB.  Slurm truthfully reported
`ReqTRES=cpu=1,mem=256M,node=1` and
`AllocTRES=cpu=2,mem=256M,node=1` while retaining `CPUs/Task=1`.  The two
allocated logical CPUs are consistent with worker-1's two threads per core and
Slurm's live `CR_CORE_MEMORY` allocation granularity.  No GPU was requested or
allocated, and the truthful count remains below the live 16-CPU user ceiling.

Compact evidence is
`evidence/r05a/runtime-identity-regression-20260716a.json`.

## VinUni operating-guide audit

The complete 1,298-line VinUni H100 Server Guide was read at SHA-256
`acee44c...b108`.  The saved preflight at SHA-256 `c351ec19...5221` recorded
an empty queue, healthy/mixed worker-1, live normal-QOS ceilings of 16 CPUs,
256 GiB and two GPU equivalents, `/mnt/data` at 73% usage, all required paths,
and no suspicious login-node process.  Only shell control-plane inspection and
source Git synchronization ran on login.  The check ran through Slurm, wrote
small artifacts below the registered experiment/log roots, and performed no
bulk transfer, broad scan, cleanup, cancellation, or service hosting.

## Decision

1. Accept job `28043` as a passed **apparatus-only** runtime-identity
   regression.  The repaired standalone symlink-aware validator accepts both
   frozen production link chains on worker-1.  Its integration into the full
   CFS workload remains untested until it is bound into a new release.
2. Permanently consume the runtime-identity run ID and preserve all receipts,
   result, log, and hashes.  Return its checked-in config to fail closed.
3. Draw no CFS, action-to-flow, collision, progress, safety, generalization,
   learnability, or infeasibility conclusion.  Neither interpreter was
   invoked, and no H100 scientific arm ran.
4. Permit preparation of one new CFS-00A release only if the scientific method
   stays byte/fingerprint frozen: same case, source, observation, policy noise,
   target, checkpoint, mask, active times, budget, solvers, iterations,
   tolerances, replay gates, worker-1 pin, and zero-simulator-action boundary.
5. This decision itself does not authorize H100 submission.  A new clean
   pushed direct-child release, unused immutable run ID, exact held transaction,
   independent review, and fresh live VinUni preflight are still required.
6. The future canary remains one H100, eight requested CPUs, 64 GiB host RAM,
   singleton `%1`, plus the registered CPU `afterany` publisher.  This stays
   within the guide's live two-GPU, 16-CPU, 256-GiB ceiling.  The registered
   two-hour limit is an explicit short-canary project policy, not a training
   template or login-node timeout.
7. No automatic cancellation or resubmission is allowed.  Do not launch
   IFT-01, execute a generated action in the simulator, tune a solver or
   tolerance, collect labels, or train a probe/MLP.

## Exact next action

Update only the CFS release apparatus to bind this terminal identity evidence,
re-run focused and full gates, obtain independent review, and then create a
separate exact two-stage release decision.  Do not submit an H100 job under
ADR-0045 alone.
