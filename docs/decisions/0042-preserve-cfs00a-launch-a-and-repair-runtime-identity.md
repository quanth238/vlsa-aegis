# 0042 — Preserve CFS-00A launch A and repair runtime identity

Status: accepted on 2026-07-16 from terminal jobs `28021_0` and `28022`.

## Evidence

The exact ADR-0041 release used immutable run ID
`r05a-constrained-flow-same-budget-canary-20260715a`, accepted implementation
`65bc57772c2648baa0e75a05d42161f01d9e3634`, and release commit
`77bf9f639abe1da224762b0d8e63c64d85698811`.  The atomic submission receipt
SHA-256 is
`0252d40ae5d6aa3fa409e867edfadfd77843bf3225d0b9d994719b9d7ddcbd01`.
The source-contract SHA-256 is
`8bf3501c0a9af17c1f65ed16812f976ec978a642b32d1bb3666866b43dcf96de`;
all 51 repository bindings matched the clean release commit.

GPU task `28021_0` received the registered worker-1 allocation but failed with
exit `2:0` at `allocation_contract` before scientific work.  Its complete log
is one message:

> missing or symlinked CFS-00A input: /mnt/data/quanth/venvs/openpi/bin/python

That canonical path exists, is executable, and is intentionally a symlink to
`/mnt/data/quanth/anaconda3/bin/python`; its resolved executable is
`/mnt/data/quanth/anaconda3/bin/python3.11`, SHA-256
`c71718900fe84a9124d39abdd9d68d029930e0dcff1764686d8d6aad97216bc9`.
The registered LIBERO interpreter is also an intentional symlink: it resolves
to
`/home/quanth/.local/share/uv/python/cpython-3.8.20-linux-x86_64-gnu/bin/python3.8`,
SHA-256
`c70efda0ee43d9a0014ee570cad3abb4f46b0c11f6ea88f7c467a91faafd4f62`.
The runner correctly froze the two public paths against environment
redirection, but then applied a blanket `test ! -L` input rule that is invalid
for these canonical virtual-environment launchers.

The cleanup-only failure writer produced
`failures/case-index-0.json`, SHA-256
`e641fad17864e53ed2f45f51b580a078c266b5659dac0f27a4f3d4355d00349c`.
No allocation test, policy server, checkpoint load, Arm A, Arm B, Arm C,
finite-difference check, FISTA solve, Adam refinement, canonical replay,
telemetry monitor, or generated simulator action ran.  No legacy or CFS raw
payload, hidden candidate, or `results.json` exists.

CPU `afterany` validator `28022` then correctly observed exact source task
`28021_0` as `FAILED|2:0` and failed with exit `3:0` before publication.  Its
receipt SHA-256 is
`7b0652b8b1b9194835d6e72a73fc65892e44496f0b1aa9e177d768d5a33512dc`;
it records `published=false`, `passed=false`, and
`failure_stage=exact_source_task_accounting`.

Compact evidence is
`evidence/r05a/cfs00a-same-budget-launch-a.json`.

## Decision

1. Classify launch A as **apparatus-inconclusive**.  It contains no observation
   of action-to-flow transport and therefore neither supports nor refutes CFS.
2. Permanently consume the run ID.  Preserve its root, receipts, logs, source
   contract, and failure artifact.  Never resume either job, reuse the ID,
   rerun only its CPU publisher, or retrofit a result.
3. Return both checked-in configs to a fail-closed terminal state: no execution
   release, `ready_to_run: false`, a nonempty blocker list, no H100 submission
   authorization, and no replacement run ID.
4. Freeze the scientific method: case, source observation, noise, checkpoint,
   target, normalization, active steps, mask, budget, Arm-A solver, Arm-B
   Jacobian/FISTA method, Arm-C initialization, tolerances, and zero-simulator
   boundary cannot change in an apparatus repair.
5. Repair only runtime identity.  Keep the two canonical public interpreter
   paths fixed, require their exact link targets and resolved executable
   targets, bind the resolved executable hashes, and require existing executable
   regular-file targets.  Retain the no-symlink rule for repository sources,
   configs, manifests, raw evidence, checkpoint, and other immutable inputs.
6. Add adversarial tests for a changed link target, changed resolved target,
   changed executable hash, missing target, non-executable target, and inherited
   runtime-path redirection.
7. Before another H100 release, run one separately registered zero-GPU,
   shell-only worker-1 allocation that validates the exact link chains and
   executable hashes from inside Slurm without invoking either interpreter.
8. A clean local repair and zero-GPU identity pass do not automatically
   authorize another H100 submission.  A new reviewed direct-child release,
   unused run ID, live preflight, and explicit no-resubmission decision remain
   required.

## Scientific boundary

Launch A did not reach the vector field.  It says nothing about whether the
privileged action delta can be represented by same-budget integrated flow
increments.  It also says nothing about collision avoidance, progress, safety,
generalization, learnability, or infeasibility.  IFT-01, label collection,
probe training, and residual-field MLP training remain forbidden.

Allowed wording:

> CFS-00A launch A stopped at a canonical virtual-environment symlink check
> before every scientific arm; the run is apparatus-inconclusive and the
> transport hypothesis remains untested.

## Exact next action

Implement only the runtime-identity contract above, run the focused CFS tests
and `./init.sh`, independently review the repair, and then preregister the
zero-GPU shell-only identity check.  Do not select another H100 run ID or submit
another CFS canary under this decision.
