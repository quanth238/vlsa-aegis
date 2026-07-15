# Archived R05A sampled-current launch A

This record preserves the detailed control-plane history of the first
ADR-0037-released sampled-current IFT-00A canary. It is provenance, not an
instruction to resume or resubmit anything.

## Registered identity

- Run: `r05a-inverse-flow-sampled-current-canary-20260715a`
- Release commit: `223667c91b05be9ab403e4d92d0cd1a96b45246f`
- Accepted implementation: `6221dedde75b9fd0f038a439d030d052ce223fb7`
- Case: `crfs-1069f29a8d76463a`
- Source node: `worker-1`
- Resources: one H100, eight CPUs, 64 GiB host RAM, two hours, singleton
  `0-0%1`, no requeue
- GPU job/task: `27928` / `27928_0`
- CPU `afterany`: never submitted

## What happened

The exact release, local/remote source identities, unused run ID, empty user
queue, node health, and 64-GiB memory requirement all passed. The launcher made
the run-root reservation and Slurm returned the GPU task user-held with the
correct resources. A compound shell pattern then falsely rejected the valid
record because real Slurm printed adjacent `JobState=PENDING` and
`Reason=JobHeldUser` fields with one shared separator.

No held-job receipt, source contract, CPU validator, atomic submission receipt,
or GPU release followed. The exact task was repeatedly inspected, then
cancelled under explicit authorization. Its terminal record is
`CANCELLED by 1073`, `00:00:00`, start `None`, node `None assigned`, and no
allocated TRES. Only `launch-reservation.json` exists in the immutable run
root; its SHA-256 is
`bec4583c5c046c7ca9f1155debc1f389d058daede42d3c485a39398f19c36ed4`.

## Meaning

This is a fail-closed launcher apparatus failure. It is not a GPU-capacity,
RAM, pi0.5, target, solver, or vector-field failure. No scientific computation
ran, so it provides no evidence for or against the research hypothesis.

ADR-0038 permits only independent exact-token parsing and its regressions. The
old job and run ID must never be resumed or reused. Any replacement requires a
new clean implementation commit and a separately reviewed ADR-0037 release
child with a new immutable identity.
