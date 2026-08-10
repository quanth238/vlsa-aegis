# Adaptive native-distance diagnostic result

## Verdict

**Strict NO-GO for quantitative `mj_geomDistance`; GO only for zero-cutoff
binary overlap.**

## Validated H100 evidence

- Job `37875`, `worker-2`, `00:01:29`.
- Clean commit `e7d2c41b15e9464171d206649fae3be029c20456`.
- Three immutable test episodes and 15 critical states.
- Three stable physical protected groups and 15 obstacle geoms.
- All queries finite; all raw contacts registered.

The zero-cutoff result is strong:

- 0 negative queries without a raw pair contact;
- 0 raw pair contacts missed by the zero-cutoff query;
- 53 registered primary contact observations;
- 15/15 starts correctly classified contact-free.

The positive-clearance result fails:

- 88 values became negative only after increasing the positive cutoff;
- 2,255 pair/substep records returned inconsistent values across repeated
  non-censored cutoffs;
- the strict adaptive gate failed, so no full target collection is authorized.

## Meaning for the research idea

MuJoCo 3.2.3 can provide the authoritative rollout event

\[
\text{protected L5--L7 contact occurred or did not occur},
\]

but this API does not provide a stable smooth clearance field for these
mesh/box pairs. Therefore it cannot directly supervise the proposed
execution-margin MLP or produce reliable action gradients and affine QP rows.

This does not disprove execution-aware learning. It identifies a missing
measurement layer. The next method must choose one of two explicit scopes:

1. retain the certified ellipsoid/box margin as the differentiable safety
   surrogate and use zero-cutoff/raw contact only as an independent false-safe
   gate and exact rollout verifier; or
2. introduce an independently validated geometry engine (for example FCL/GJK
   with a registered convex representation) to produce quantitative physical
   clearance, then repeat the target audit before learning.

Pure binary-contact regression is not the preferred immediate continuation:
the earlier classifier experiment already showed that classification alone
does not yield a meaningful steering gradient.

## Decision

- Reject all positive-cutoff `mj_geomDistance` targets for this experiment.
- Preserve zero-cutoff overlap/raw contact as binary `D_sim` authority.
- Do not train, solve physical-row QPs, or run closed-loop E05.
- Ask the advisor whether the paper should claim a certified proxy-margin
  method with exact contact verification, or require a new external physical
  distance layer. This choice changes the core method and claim.

## Artifacts

- Result SHA-256:
  `d096e6c8c18fd86961fbd59b47f08e10d824c37c8313bf7d21dbb147c804b785`.
- Result payload SHA-256:
  `97438521d8b3513c5402fa09c91085ba15877135490eaf7684b878ed15c0a8ef`.
- Validation SHA-256:
  `6a606382d118c5320fa121d16047d910fa3326d5bdbb73a5e18297a610806fa2`.
- Preflight SHA-256:
  `5fc2d1be2423b4ded2f1f04929bfe81502e8bf137a9741b76f3f51f665893208`.
- Run root:
  `/mnt/data/quanth/experiments/vlsa-distal-adaptive-native-distance/adaptive-native-distance-20260810a`.
