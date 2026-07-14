# 0009 — Accept endpoint-free physical witnesses and require oracle flow

Status: accepted on 2026-07-14.

## Context

H05 rejected exact endpoint preservation because every frozen nominal endpoint
was unsafe. R01 removed that equality and searched all 15 first-five
translation commands while preserving nominal orientation and gripper commands.
`D_opt` only nominated candidates; repeated direct `D_sim` replay decided the
registered safety-progress predicate.

Slurm array `27306` completed all 20 immutable cases with no case failure and
reproduced all 20 nominal collisions. Allocation-backed verifier `27364`
recomputed the gate from raw actions and repeated rollouts. It accepted 17/20
changed-action safe-progress witnesses, above the registered 12/20 threshold.
The evidence is `evidence/r01/r01-summary.json`, SHA-256
`715d9326f02df531e0c6d7aa7b94629569b320800b4283c36fff41335d9b8bc5`.

The three negative cases are exactly the three whose immutable branch clearance
was already below 5 mm; one branch was already in penetration. No action chosen
at that instant can change the initial sample in the inclusive branch-plus-125-
substep predicate. Conversely, all 17 branch-margin-valid cases had a verified
witness.

The witnesses are not local. All 17 are outside the H04 `+/-0.15` calibration
domain and saturate at least one translation component. Their correction L2
range is 2.608--3.869 (median 3.243), and correction RMS range is 0.673--0.999
(median 0.837) over 15 translation coordinates. `D_opt` produced no false-safe
attempt, but it produced 25 clearance false-negative attempts across five cases.

## Decision

Mark R01 passing and activate R02. The accepted conclusion is only that an
endpoint-free, five-action, pre-grasp reach prefix physically exists for at
least 60% of this frozen development population under the controlled `D_sim`
metric. It is not evidence of transport safety, full-arm safety, flow
reachability, or learned ECG efficacy.

Before R02 outcomes are interpreted, establish exact public-JAX/PyTorch
conversion parity on identical transformed input and explicit noise. Then keep
the paired frozen, direct-witness, equal-norm random, analytic geometry,
distributed oracle residual, and one-shot bridge arms distinct. Content-bind
R02 to the R01 summary and preserve the original 20-case population alongside
the feasible-conditioned analysis.

No learned probe is authorized unless the distributed oracle intervention and
grouped R03 analysis pass. In the controlled oracle-geometry pilot, a learned
probe also requires a demonstrated gap relative to the analytic geometry arm;
approximate-clean gradient guidance alone is not a sufficient novelty claim.

## Consequences

- Removing endpoint equality solves the registered physical-existence failure
  on every case that begins outside the safety buffer.
- The next root question is whether a large, saturating safe direction lies in
  reachable and task-preserving frozen-flow support.
- The three already-too-close branches motivate a separately preregistered
  earlier-intervention or recovery-barrier study, not a post-hoc R01 retry.
- A post-grasp transport study requires new immutable state identities and a
  new phase-specific progress calibration.
- Probe labels, if eventually authorized, must be the clearance of the actual
  deterministic continuation or the directly rolled-out approximate-clean
  action, never the clearance of a different source action.
