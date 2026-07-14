# 0010 — Freeze R02 parity and direction semantics before outcomes

Status: accepted on 2026-07-14, before any R02 policy outcome was run.

## Context

R01 establishes direct physical witnesses, but R02 uses the converted PyTorch
sampler because the public JAX implementation has no intervention seam.  The
upstream project documents matched JAX/PyTorch precision and LIBERO validation,
but this repository had neither a conversion-parity artifact nor a numerical
acceptance tolerance.  Those values must be fixed before observing the result.

All 17 R01 witnesses are large and outside H04's local calibration domain.
Consequently the random and analytic comparators must share the intervention's
normalized-model-space norm, not its physical-space norm.  A displacement is
converted with normalization scale only.  An absolute midpoint predicted-clean
action is inverse-normalized with the complete output transform, including its
offset.

## Decision

The parity input is one deterministically reconstructed frozen SafeLIBERO
branch observation when available.  A synthetic same-shaped input is only an
apparatus fallback and must be labeled as such.  Both backends receive the same
already-transformed tensors, explicit float32 noise of shape `1 x 10 x 32`, and
ten Euler updates.  For steps 0 through 9, record the pre-update latent `x_t`
and velocity `v_t`; also record the final normalized `x_0` and the exact normal
output transform to physical LIBERO actions.

Each instrumented backend loop must reproduce its own untouched default final
sampler output with `array_equal`.  The converted PyTorch compiled default and
its eager trace-only path must also be `array_equal`.  Cross-framework BF16
parity passes only when every registered bound passes:

| quantity | maximum absolute error | RMS absolute error |
| --- | ---: | ---: |
| each step's normalized latent | 0.10 | 0.025 |
| each step's normalized velocity | 0.20 | 0.050 |
| final normalized `10 x 32` action | 0.10 | 0.025 |
| final physical first-five translation | 0.010 | 0.005 |
| final physical first-seven LIBERO action | 0.050 | 0.015 |

The 0.010 translation bound is fixed from the H04 response scale: even a
worst-direction error of that size across five commands is approximately 1 mm,
the same order as the held-out H04 clearance error.  It is not selected from a
parity run.  Exact input/noise hashes, checkpoint tree/model hashes, norm asset
hash, code/dirty state, device, dtypes, all errors, and the pass decision are
written atomically in the allocation.

For each R01 witness,

```text
Delta_star_phys = A_witness[0:5,0:3] - A_nominal[0:5,0:3]
Delta_star_model = Delta_star_phys / action_scale
```

with zeros in rotation, gripper, and the discarded tail.  The pi0.5 LIBERO
checkpoint uses quantile normalization.  Its registered translation scale is
`(q99-q01+1e-6)/2 = [0.8422505, 0.827813, 0.937313]`, bound to norm-stat asset
SHA-256 `b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84`.

The random comparator is deterministic from the immutable case seed, has no
temporal-mean subtraction, is nonzero only in the first-five translation mask,
and has exactly `||Delta_star_model||_2`.

The analytic comparator is the exact subgradient of the registered sampled
swept EEF-sphere/OBB `D_opt` at the midpoint predicted-clean physical action.
It evaluates 26 samples on each of five segments and selects the global minimum
by `(distance, segment index, sample index, obstacle name/index)`.  The OBB SDF
uses its ordinary outside gradient; inside/on the box it chooses the lowest
axis attaining the maximum signed face coordinate and defines `sign(0)=+1`.
The selected point gradient is propagated through the frozen H04 response
matrix, multiplied by physical action scale to obtain the model-coordinate
gradient, masked, and normalized to `||Delta_star_model||_2`.  A zero or
nonfinite gradient is an explicit analytic-arm failure; no alternate pair,
clipping, smoothing, or post-outcome tuning is allowed.

Final sampled physical actions, rather than intermediate corrections, are
checked against `[-1,1]` on the first-five translation channels.  An invalid
sample is a bounds failure and is never clipped into a simulator pass.

## Consequences

- A failed parity artifact blocks interpretation of all PyTorch R02 outcomes.
- The old endpoint-preserving random helper cannot be reused.
- The analytic arm is an extrapolative diagnostic because H04 was calibrated
  only to `+/-0.15`; repeated `D_sim` remains the outcome authority.
- The direct witness, distributed residual, and one-shot bridge remain distinct
  arms.  The bridge is diagnostic and does not substitute for the primary
  distributed oracle arm.
