# Historical progress snapshot

`2026-07-15-pre-inverse-flow-pivot.md` is a byte-exact archive. It contains an
obsolete R03A population command because that command was current when the
snapshot was made. **Do not run it.** ADR-0028 retired the population before it
was launched. Current authorization lives only in the root `PROGRESS.md`,
`EXPERIMENTS.md`, and `feature_list.json`.

`2026-07-15-r05a-pre-execution-apparatus.md` is a curated archive of completed
R05A apparatus attempts through ADR-0036. It preserves exact run/job identities
and diagnostic meaning without acting as authorization for another retry,
IFT-01, or training.

`2026-07-15-r05a-sampled-current-launch-a.md` preserves the first exact
ADR-0037 release and its fail-closed control-plane parser failure. Job
`27928_0` had zero runtime and was cancelled; the record is not authorization
to resume or reuse its consumed run ID.
