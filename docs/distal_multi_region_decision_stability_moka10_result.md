# Multi-region decision-stability audit result

## Verdict

**GO for decision-level resampling stability at zero margin.** All frozen
accepted-set, false-safe, support, selected-action, and fresh exact-rollout
gates passed. The 13 raw coefficient-norm failures observed in job 37690 are
therefore decision-harmless on this sampled 50-state population.

This does not change job 37690: its fixed multi-region all-gates result remains
a strict NO-GO because the +1 mm arm missed its hard buffer and the immutable
off-grid support gate at state 19 did not pass. This audit trained no MLP and
ran no closed-loop E05 experiment.

## Frozen gates

| Gate | Requirement | Observed | Result |
|---|---:|---:|---:|
| Replicated off-grid false-safes | 0 / 76,800 | 0 / 76,800 | pass |
| Minimum global accepted-set Jaccard | at least 0.98 | 0.999495 | pass |
| Minimum state/replicate Jaccard | at least 0.80 | 0.952381 | pass |
| Complete-fit supported states retained | 49 / 49 | 49 / 49 in every replicate | pass |
| Fresh exact-safe selected QPs | 800 / 800 | 800 / 800 | pass |
| Selected-action L2 shift p95 | at most 0.05 | 0.002991 | pass |
| Selected-action L2 shift maximum | at most 0.15 | 0.031053 | pass |

The mean selected-action shift was `0.000667`. Region identity matched the
complete fit in 693/800 selections, but the regions overlap: changing the
region index is not a changed controller action. All 107 region-index changes
still produced small selected-action shifts and exact-safe cloned-OSC
rollouts.

The minimum state-level Jaccard occurred at state index 44,
`vlsa-t1-goal-ii-t0-e10`, step 156, and remained `0.952381`. The largest
selected-action shift, `0.031053`, occurred at the sparse state index 19;
that state retained Jaccard 1.0 and every selected action remained exact-safe.

## Execution evidence

- Slurm job: `37701`, H100 `worker-2`.
- Scientific wall time: `549.361 s`.
- Source commit: `f4fee4b8f78d145b45a35c46b0cfd44eca83f90c` (clean).
- Remote root: `/mnt/data/quanth/experiments/vlsa-distal-multi-region-decision-stability/region-decision-stability-20260810a`.
- Local copy: `output/vlsa_distal_multi_region_decision_stability/region-decision-stability-20260810a`.
- Result file SHA-256: `1ac02e8f45730b8961628e2b1f3ea985146f07b1f7574ad44fb8570f7c08aeeb`.
- Result payload SHA-256: `d142df3ba81d9129884dfb5bf281dc77b279ab59814b58e52a85322c40fed2ee`.
- Validation SHA-256: `3fe8943f0cf20daa0b621da61df8b42aea4ee3b9530987a0a78b40b10565a7cc`.
- Independent validation status: `valid`.

## Interpretation and next gate

The experiment supports the fixed overlapping multi-region oracle as a stable
zero-margin control representation on the sampled population even when its
raw affine coefficients are non-unique. It clears a separately preregistered
grouped state-conditioned learned-model experiment; it does not establish
learned generalization, formal safety, positive-buffer safety, deployment
reliability, or task completion.

No learned job should run until its grouped episode split, region output,
one-sided uncertainty tightening, zero-false-safe test, and fresh exact-rollout
gates are frozen. Closed-loop E05 remains blocked until that learned gate
passes.
