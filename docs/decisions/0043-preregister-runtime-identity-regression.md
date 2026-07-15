# 0043 — Preregister the R05A runtime-identity regression

Status: accepted for implementation on 2026-07-16.  Execution remains blocked
until an independently reviewed implementation commit receives a separate exact
release.

## Basis

CFS-00A launch A stopped before scientific execution because its workload
treated the canonical virtual-environment Python launchers as forbidden
symlinks.  ADR-0042 permitted only a runtime-identity repair.  Commit
`5a5f3306b96f1491a4baa52657193afc1bfad2b5` implements that repair and passed:

- 72/72 dependency-backed focused CFS tests;
- 593/593 dependency-light repository tests with 198 declared skips;
- exact shell syntax, JSON, source-binding, and whitespace checks; and
- three independent reviews with no P0/P1 finding.

The repair freezes each public launcher, its direct `readlink` target, its
`readlink -f` resolved executable, regular/non-symlink/executable status, and
the resolved binary SHA-256.  It never invokes an interpreter during identity
validation.  The public virtual-environment paths remain the eventual execution
entrypoints.

## Registered question

From inside one worker-1 Slurm allocation, do both production interpreter
launchers satisfy the exact ADR-0042 identity contract using shell-only
inspection?

This is an apparatus test.  It cannot support or refute action-to-flow
transport, collision avoidance, progress, safety, learnability, generalization,
or infeasibility.

## Frozen execution shape

- one non-array Slurm job on `worker-1`;
- partition/account/QOS `main`/`normal`/`normal`;
- one CPU, exactly 256 MiB host RAM, zero GPUs;
- time limit `00:02:00`, no requeue;
- source submitted held, exact fields checked, immutable source/submission
  receipts written, then released exactly once;
- no automatic cancellation, resubmission, H100 launch, or next gate.

The allocation may use only shell builtins and ordinary control tools needed to
validate and receipt the contract: `readlink`, `sha256sum`, `git`, `jq`,
`hostname`, `awk`, `sed`, `tr`, `date`, `mktemp`, `mv`, `scontrol`, and
Slurm-provided environment values.  It must not invoke either Python launcher
or any other Python command.  It must not load a checkpoint, model, CUDA
runtime, simulator, renderer, metric, search, or training path.

## Pass condition

A pass requires one immutable `results.json` that binds the exact clean pushed
commit, Slurm job, worker-1, zero-GPU resources, held receipt, source contract,
atomic submission receipt, and all eight reviewed runtime identity strings.
The result must record `shell_only=true`, both interpreter-invocation flags
false, and every scientific/H100 authorization false.

Any path, link, resolved target, file type, executable bit, binary hash, source
binding, resource, job, node, receipt, or output mismatch is a failed apparatus
check.  A failure has no scientific interpretation.

## Release boundary

The checked-in config remains fail closed with `execution_release: null`.  A
later direct-child decision may select one unused immutable regression run ID
and authorize exactly one zero-GPU submission after local tests, independent
review, push, VinUni synchronization, live preflight, an empty user queue, and a
healthy worker-1.  That release still cannot authorize an H100 job.
