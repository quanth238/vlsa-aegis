# 0037 — Require an exact single-canary execution identity

Status: generic enforcement accepted on 2026-07-15. The implementation commit
containing this prefix is unreleased. Any later execution authority exists only
in the exact canonical appendix added by its reviewed direct child.

## Context

ADR-0036 produced a reviewed full-lifetime sampled-current apparatus at commit
`d3d51d3d5fe4d318760c89a87113ee4c3c0500d7`. It deliberately left
`ready_to_run: false` and selected no run ID. Review of the transition to
execution found that flipping those two fields alone would be unsafe: the
submitter accepted any syntactically safe unused `RUN_ID` and treated whatever
clean `HEAD` happened to be present as authorized.

That behavior is adequate for a blocked implementation but not for an
intentional, single scientific experiment. The exact run identity, reviewed
implementation ancestor, release commit, host, and resources must all be
explicitly controlled before `preflight.sh`, SSH, run-root creation, or
`sbatch`.

## Decision

1. Use a two-commit transition.

   - The first commit adds generic release enforcement and tests while keeping
     the apparatus unreleased. This becomes the reviewed implementation commit.
   - A direct child release-only commit may change only the apparatus config
     and this ADR. It selects exactly one run ID and changes no tracker,
     feature status, executable, schema, scientific config, model path, or
     test. Root trackers remain pre-execution until terminal evidence exists.

2. A launchable apparatus config must have `ready_to_run: true`, an empty
   `blocked_on` list, and one exact-key `execution_release` object containing:

   - schema and artifact-role identifiers;
   - this decision path;
   - the full reviewed implementation commit;
   - one exact immutable run ID;
   - `single_submission: true`;
   - source host `worker-1`;
   - the complete registered H100 and CPU-validator resource object;
   - `release_only_parent_required: true`;
   - the exact allowlist of release-only changed paths;
   - `automatic_resubmission_allowed: false`;
   - `automatic_next_experiment_allowed: false`.

3. The submitter must receive `EXPECTED_RELEASE_COMMIT` externally. It may not
   derive authorization from `HEAD`. Before local preflight it must require:

   - caller `RUN_ID` equals the config's one registered ID;
   - local clean `HEAD` equals `EXPECTED_RELEASE_COMMIT`;
   - local `origin/agent/crfs-oracle-harness` equals that commit;
   - the registered implementation commit is the sole direct parent of that
     commit, so a merge release is rejected;
   - every changed path belongs to the release-only allowlist;
   - after removing only `ready_to_run`, `blocked_on`, and
     `execution_release`, the release config is canonically identical to its
     implementation-parent version;
   - this ADR is byte-identical to its implementation-parent version followed
     by one canonical appendix derived from the validated implementation
     commit, run ID, worker/resource contract, single-submission rule, and
     no-efficacy/no-training boundary; no rewrite or trailing claim is allowed;
   - the config host and resources equal the frozen Slurm contract.

4. The remote control-plane transaction independently requires the same clean
   release commit and origin ref, exact config hash, exact run ID, sole direct
   implementation parent, release-only diff, identical non-release config
   projection, exact canonical ADR appendix, worker-1 placement, resource
   contract, empty user queue, unused run root, healthy node, and at least
   65,536 MiB `FreeMem` before creating anything.

5. The H100 wrapper, CPU publisher wrapper, and Python finalizer independently
   require the source-contract run ID to equal the registered config run ID.
   The release decision is added to every repository source-binding set and is
   hashed into the source contract.

6. The exact run root is single-use. Atomic `mkdir` is the reservation. Its
   existence, including a partial failed launch, consumes the ID. Never resume
   or reuse it under another command. A failure after creating a held job must
   be inspected by exact job ID; no broad cancel or cleanup is allowed.

7. The only authorized execution shape remains one singleton `0-0%1` task on
   worker-1: main/normal/normal, one H100, eight CPUs, exactly 65,536 MiB, two
   hours, no requeue, followed by one main/normal/normal zero-GPU CPU
   `afterany` validator with two CPUs, 8,192 MiB, and 15 minutes.

8. Execution-release tests must prove that unreleased config, missing or wrong
   external release commit, arbitrary or alternate run IDs, dirty state,
   origin/remote drift, non-parent releases, non-allowlisted changes, host or
   resource drift, pre-existing run roots, and repeated submissions fail
   closed. The one exact fake transaction must still create a held GPU job,
   register its CPU `afterany`, write all receipts, and release only that GPU
   job once.

## Scientific boundary

This decision changes execution authorization only. It does not change the
case, observation, instruction, policy noise, target, checkpoint,
normalization, solver, 128-update limit, optimizer, tolerances, control mask,
active steps, budget, per-step cap, telemetry semantics, or CPU publication
contract.

It authorizes no run by itself. It authorizes no IFT-01 rollout, efficacy
claim, reroute, solver tuning, population, probe, label collection, or MLP
training. A later release-only commit must name the exact single canary and be
reviewed before any launch.
