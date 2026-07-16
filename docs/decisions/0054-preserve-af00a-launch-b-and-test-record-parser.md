# 0054 — Preserve AF-00A launch B and behavior-test Slurm record parsing

Status: accepted apparatus diagnosis and parser repair on 2026-07-16. This
decision does not authorize another H100 submission.

## Terminal launch-B evidence

Release `778bfb1a9b3d0c6b8351946885b5e3fa9b8cc5c6` reserved immutable run
`r05a-actual-forward-cem-canary-20260716b` and submitted exact held task
`28279_0`. The exact child query returned the valid VinUni record beginning
`JobId=28279 ArrayJobId=28279 ArrayTaskId=0 ArrayTaskThrottle=1`.

ADR-0052's accepted semantic rule was correct, but its inline shell extraction
used a pattern requiring a space before `JobId=`. Because `JobId=` is the first
token, extraction returned an empty value and the transaction stopped while
the task was still user-held. No held receipt, source contract, CPU publisher,
fingerprint, or GPU release followed. After exact inspection, task `28279_0`
was cancelled under the user's explicit authorization. Slurm records
`CANCELLED`, exit `0:0`, start `None`, elapsed `00:00:00`, no node, and no
allocated TRES.

The immutable root contains only its launch reservation, preflight, and
provisional GPU-ID receipt. No allocation, Python, pi0.5 request, Arm A, CEM,
Arm B/C, or simulator action ran. This consumed identity is apparatus-
inconclusive and supplies no transport, safety, progress, efficacy, or
infeasibility evidence. Its compact record is
`evidence/r05a/af00a-actual-forward-launch-b.json`.

## Narrow parser repair

Replace both inline `JobId` extractors with one shared shell-only field parser.
It must require exactly one nonempty `JobId`, `ArrayJobId`, and `ArrayTaskId`
token. It accepts displayed `JobId` only as the exact parent or exact child
spelling, then independently requires the exact parent and task zero.

The regression test must execute the parser, not merely inspect source text.
It uses the real VinUni parent-form record and the permitted child form as
positive cases, and rejects wrong displayed job, wrong array parent, wrong
task, missing `JobId`, and duplicate `JobId`. Every existing held/running
state, resource, ReqTRES, source, receipt, fingerprint, and one-release check
remains unchanged.

This is a parsing repair only. It changes no AF-00A case, model, target, noise,
budget, CEM query, objective, tolerance, selection rule, action, or resource.

## Release boundary

Return the config to fail closed. After focused and complete gates plus
independent review, one direct-child release may change only the config and
`docs/decisions/0055-release-behavior-tested-actual-forward-canary.md`, choose
a fresh unused immutable run ID, and authorize one submission. No automatic
retry, IFT-01, probe, or MLP is authorized here.
