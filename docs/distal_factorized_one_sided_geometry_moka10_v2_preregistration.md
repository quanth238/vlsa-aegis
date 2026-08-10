# One-sided geometry supervision v2: validation-adapted preregistration

## Why v2 is separate

The v1 gate required the immutable baseline's validation false-safes to repeat
the test set's exact L5/substep-50 identity. Validated job `38064` found a
strict v1 NO-GO: all 93 validation false-safes were terminal substep 50 but on
the first L6 ellipsoid row.

This result is not retroactively counted as a v1 pass. It motivates a separate
validation-adapted question: is terminal optimism shared across protected
distal links? That is the relevant condition for the already frozen loss,
which sums over all seven L5--L7 rows.

## Frozen v2 adaptation

The only protocol change is the conditional authorization test:

- at least one baseline validation false-safe; and
- at least 50% of validation false-safes have their exact worst L5--L7 row at
  terminal substep 50.

The v1 result and validation hashes are immutable inputs. The model,
population, split, features, action rows, local `dh/dq` construction,
architecture, five seeds, optimizer, schedule, joint loss, sensitivity loss,
one-sided geometry loss, weights, evaluation, and untouched test gates are
identical to v1.

In particular, the experimental network still outputs only the 51-by-7 joint
trajectory. The local train/validation-only `dh/dq` supplies signed geometric
supervision; known FK and the unchanged L5--L7 ellipsoid evaluator calculate
test safety.

## Untouched test gate

The 960 E05/E10/E15 random actions remain the final gate. All are required:

- zero false-safe actions;
- safe support in 15/15 states;
- at least 50% safe recall;
- near-boundary RMSE no greater than 3.058 mm;
- joint-sensitivity cosine at least 0.8;
- safety-margin sensitivity cosine at least 0.8; and
- strictly fewer false-safes than the immutable baseline's 44.

No new controller labels, generic surface loss, classifier, calibration,
Poisson/SDF, QP, or closed-loop action is allowed. A failure redirects the
execution model to a shared time-conditioned dynamics decoder.

Frozen config:
`configs/vlsa_distal_factorized_one_sided_geometry_moka10.v2.json`, SHA-256
`c2d4d6b69294531100a0b361b703c156b276238a9581e6b414132c2564995ec2`.
