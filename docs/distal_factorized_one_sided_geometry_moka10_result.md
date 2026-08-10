# One-sided geometry supervision v1: validated conditional NO-GO

## Result

H100 job `38064` completed the preregistered validation audit and an
independent replay. It correctly skipped training because validation did not
reproduce the test set's L5-specific pattern.

- Validation random actions: `640`.
- Baseline validation false-safes: `93`.
- False-safes at terminal substep 50: `93/93`.
- False-safes whose exact worst row was L5: `0/93`.
- Exact worst row: row 3 for all 93 actions, the first L6 ellipsoid row.
- Validation local `dh/dq` random-action RMSE: `0.201 mm`.
- Validation clearance-delta cosine: `0.974`.

Thus the registered L5-specific validation condition is a strict NO-GO and
cannot be retroactively passed. No model was trained.

The outcome refines the mechanism: validation strongly reproduces terminal
optimism, but on L6 rather than L5. Since the proposed auxiliary loss is
defined over all seven L5--L7 rows and all substeps, this supports a separate,
explicitly validation-adapted link-generic experiment. It does not change the
v1 result.

No controller rollout label, surface-position loss, binary classifier,
uncertainty calibration, Poisson/SDF, QP, or closed-loop execution ran.

## Reproducibility

- Job: `38064`, `COMPLETED` in `00:10:49` on `worker-2`.
- Source commit: `4729acdba0109d32c94611761f861644a11f95f6`.
- Result SHA-256:
  `bad80f43ce205246515ba5aaffcb582b167d211bf7cb6862aea0781edd392afd`.
- Result payload SHA-256:
  `5bd63eaf284c0d99e4f76fde7df2780e470bec39e295e2a610be9147846f6189`.
- Validation SHA-256:
  `5cb612cdf35b7b891b820d5430c445e780f7af307832fc7eb6a2af2a2ac28bf4`.
- Validation payload SHA-256:
  `883bab58cbec9b87f57288a90253e771594100981cb636d4f4af38516f22d914`.
