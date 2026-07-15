# 0041 — Require an exact CFS-00A execution release

Status: preregistered release protocol.  At the implementation commit this
decision is fail-closed; H100 submission becomes authorized only when a direct
child release commit appends the canonical `Exact execution release` section.

## Decision

CFS-00A may execute only from a clean, pushed, origin-synchronized release
commit that is a direct child of one independently reviewed implementation
commit.  The release commit may change only:

- `configs/experiments/r05a_constrained_flow_canary.json`;
- `configs/experiments/r05a_constrained_flow_canary_apparatus.json`;
- this decision.

The two configs must contain the same `execution_release` object.  It binds the
accepted implementation commit, one unused immutable run ID, source host
`worker-1`, the exact H100 and CPU-afterany resource contracts, the three
release-only paths above, one submission, no automatic resubmission, and no
automatic next experiment.  The scientific config additionally changes only
release bookkeeping: `config_status` becomes
`released_exact_single_canary`, `ready_to_run` becomes true, `blocked_on`
becomes empty, and `h100_submission_authorized` becomes true.  These fields do
not alter the frozen scientific hash in ADR-0040.

Before releasing the held H100 job, the transactional submitter must prove:

1. local `HEAD`, the configured release commit, the checked-out VinUni source,
   and the tracked origin branch are the same commit with clean trees;
2. the release parent is the accepted implementation commit and its diff is
   limited to the three paths above;
3. every source/config/schema/script/test byte in the source contract matches
   the release tree;
4. the run root and immutable run ID are unused;
5. the exact source node is healthy, the queue/QOS permit the registered
   resources, and no routing or tolerance is changed;
6. one held `0-0%1` H100 array and one CPU `afterany` publisher are created,
   recorded atomically, and only then released.

The H100 task writes raw `canary-payload.json` and
`constrained-flow-payload.json` evidence only.  The CPU job is the sole owner
of `results.json`; it independently rehashes and validates the raw artifacts,
telemetry, allocation tests, source task, release identity, and scientific
semantics.  A terminal apparatus failure may omit the legacy raw payload only
when its explicit absence receipt and filesystem agree.  No result authorizes
simulator efficacy claims, probe/MLP training, IFT-01, expansion, or retry.

## Pre-release placeholder

At the implementation commit the experiment is unreleased.  The reviewed
implementation commit, immutable run ID, and exact resource authorization are
inserted only by the canonical `Exact execution release` section appended in
the direct-child release commit.  That appended section supersedes this
placeholder.  Live preflight evidence, source-contract digest, submission
receipt digest, and final Slurm job IDs remain terminal execution evidence,
not preregistered values.  Until the canonical section exists, submission is
forbidden.
