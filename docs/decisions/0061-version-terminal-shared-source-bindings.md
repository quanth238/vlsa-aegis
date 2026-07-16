# 0061 — Version terminal shared-source bindings by their release commit

Status: accepted for implementation on 2026-07-16. This is a historical
validation repair only. It changes no terminal artifact, frozen hash,
scientific method, target, budget, outcome, or execution authorization.

## Problem

The terminal sampled-current protocol correctly recorded the exact SHA-256 of
`pi0_pytorch.py` and `policy.py` used by release
`06b365b5899c2cb31db12187350cce48a3a0ea20`. Its later unit validator,
however, compares those historical hashes with the bytes in the current
worktree. That silently assumes two shared integration files can never receive
another strictly opt-in experiment mode. A legitimate later edit therefore
causes validation of the old source contract to fail before the validator can
inspect the immutable release bytes.

This is the wrong time reference. A terminal source contract describes the
recorded release, not every later analysis checkout. Replacing the old hashes
with new hashes would be worse because it would rewrite the identity of an
already completed run.

The permanently closed sampled-current submitter has the same ordering issue:
it hashes live shared source before reading `ready_to_run: false`, so a later
opt-in edit changes the intended fail-closed exit into an unlabelled shell
failure. No submission occurs, but the diagnostic contract becomes unstable.

## Decision

1. Preserve every recorded sampled-current hash unchanged.
2. Bind the two shared mutable integration paths to exact terminal release
   commit `06b365b5899c2cb31db12187350cce48a3a0ea20`:
   - `openpi/src/openpi/models_pytorch/pi0_pytorch.py` SHA-256
     `80366dcc7b2ddc598717d4c71c0e68e46312a1e5ffd3fc04479d5599433f4c55`;
   - `openpi/src/openpi/policies/policy.py` SHA-256
     `d16767ff2073d5c177cdfcc06dc05dcbf7cdb9ef2a2a0150023f7953b03508b9`.
3. Historical validation must read those exact Git blobs from the registered
   release commit and hash their bytes. It must still require the artifact's
   recorded repository hashes to equal the frozen values. It may not replace
   them with current-worktree hashes or accept a missing Git object.
4. All repository paths that were not designated shared/versioned remain
   validated against the current checkout exactly as before.
5. The sampled-current submitter must check its already-closed apparatus state
   before hashing live source. With `ready_to_run: false`, it exits `2` with
   the registered not-released message and performs no SSH, Slurm, or run-ID
   mutation. Its released-path checks remain unchanged and unreachable while
   the config is closed.
6. Tests must prove the release-commit blobs retain the old hashes, a wrong or
   missing blob is rejected, artifact-recorded hash drift is rejected, other
   bound-file drift is still rejected, and the closed submitter stops before
   remote work.
7. Any later protocol test that asserts the old sampled-current shell bytes as
   a historical dependency must likewise read them from that protocol's
   registered implementation/release Git tree, never from the live analysis
   checkout.

## Boundary

This decision supplies no evidence for TRL-00A, teacher transport, simulator
safety/progress, generalization, or learnability. It does not authorize a job
or relax baseline parity. The new reference-lift implementation still needs
its own clean source contract, regressions, review, direct-child release, and
allocation-backed evidence.
