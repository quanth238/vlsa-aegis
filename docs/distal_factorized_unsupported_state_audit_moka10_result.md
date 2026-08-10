# One-sided factorized unsupported-state audit result

## Verdict

H100 job `38075` completed the frozen audit and an independent metric replay.
The one-sided model remains a strict NO-GO. The missing 3/15 state support is
not one failure mode:

- two states contain no exact-safe action in the registered 64-action region;
- one state contains 44 exact-safe actions, but the model and every ensemble
  member reject all of them.

No training, new controller rollout, correction fit, calibration, QP, closed
loop, classifier, or Poisson/SDF ran.

## State audit

| Case and state | Exact-safe actions | Best exact margin | One-sided margin | Explanation |
|---|---:|---:|---:|---|
| E05 step 185, state 39 | 0/64 | -0.294 mm | -0.721 mm | sampled candidate region has no safe support |
| E10 step 156, state 44 | 0/64 | -0.788 mm | -3.031 mm | sampled candidate region has no safe support |
| E15 step 150, state 49 | 44/64 | +2.448 mm | -1.968 mm | small-threshold but 4.416 mm conservative model error |

All three complete OSC states pass the frozen training-support test. Their
one-sided joint-trajectory RMSEs are 4.516, 9.266, and 10.126 mrad, all below
the registered 14.356 mrad large-error reference. Therefore missing grouped
state support and unusually large trajectory error do not explain these
outcomes.

For E15, all five members accept zero exact-safe candidates; their best
predicted worst margins range from -5.397 to -2.706 mm. The failure is not
caused by conservative ensemble aggregation. The exact worst constraint is L5
row 2 at substep 50, while the ensemble-mean prediction activates L5 row 1 at
substep 47. This is consistent with systematic conservative boundary/mode
bias in the current model, not uncertainty isolated to one ensemble member.

For E05 and E10, the original factorized baseline predicts the least-unsafe
candidate as safe even though its exact margin remains negative. The one-sided
model correctly rejects those observed candidates. A bias correction cannot
create a safe action where the sampled region contains none.

## Consequence

The next feasibility test must separate two mechanisms:

1. Expand or continuously refine the registered recovery region around the
   best E05/E10 candidates and use exact rollout margins to establish whether
   safe support exists nearby.
2. On validation episodes only, test a link/time/state-conditioned residual
   correction for supported states like E15. It must restore accepted safe
   support without introducing a false-safe.

Only their combination can potentially restore 15/15 support. E05/E10/E15
remain diagnostic cases and cannot fit the correction or support the final
generalization claim; newly reserved complete episodes are required.

## Provenance

- Slurm allocation: `38075`, `worker-2`, one NVIDIA H100 80 GB HBM3
- elapsed time: `00:04:32`
- source commit: `b6e61417e66d371b4131867119a279edca8a6fc4`
- records SHA-256: `fa55559bed38efe0bc581d8e3d6887e0f990a5f21ae5c7555443ef5d8246b632`
- result SHA-256: `1b9a475c3090b4cff32c0df161e3ee99b4695a7cd5f481cbfe520ba26e0eca8b`
- result payload SHA-256: `7987336948790f35fc9816bfd5015e6e223dbf94e129dfdd209bce3db8115ceb`
- validation SHA-256: `b74a1f47e1e4ddb3848af6225cebf858dfe0aeaf317263c7c142e62b7f37a511`
- validation payload SHA-256: `96031a2fe96c0dcb7ab990c805765d1e36b9d4ebd2652e2e5b3422d312459ba5`
