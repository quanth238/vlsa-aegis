# Direct-horizon split-wise sensitivity audit result

## Verdict

H100 job `38560` validates a strict NO-GO for the frozen job-`38376`
execution model and identifies the weak action sensitivity as a fitted-model
problem already present on training episodes:

\[
\boxed{\text{loss scaling / direct-horizon decoder underfitting, not primarily unseen-state coverage}}
\]

No model was trained and no rollout labels were collected. Calibration, QP,
closed-loop execution, Poisson/SDF, classifiers, and the frozen future task-3
population were not opened or run.

## Split-wise paired sensitivity

For every registered paired action perturbation, the audit compares the model
with the cloned-OSC secant

\[
\frac{q(x,A+\epsilon d,k)-q(x,A-\epsilon d,k)}{2\epsilon}.
\]

The gate requires mean cosine at least `0.8`, median predicted/exact norm ratio
in `[0.5,1.5]`, and mean relative norm error at most `0.5` for both joint and
safety sensitivity.

| Split | Joint cosine | Joint norm ratio | Joint relative error | Safety cosine | Safety norm ratio | Safety relative error | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| Train | 0.809 | 0.416 | 0.598 | 0.741 | 0.422 | 0.553 | Fail |
| Validation | 0.751 | 0.443 | 0.586 | 0.678 | 0.513 | 0.444 | Fail |
| Old diagnostic | 0.814 | 0.451 | 0.572 | 0.707 | 0.469 | 0.489 | Fail |
| Reserved unseen | 0.736 | 0.383 | 0.647 | 0.646 | 0.411 | 0.590 | Fail |

Training passes only joint direction. It fails joint magnitude, safety
direction, and safety magnitude. Validation also fails before the reserved
episodes are considered. Therefore, adding more grouped episodes alone is not
the first justified intervention.

## Horizon localization

The defect is an action-to-horizon gain mismatch. Direction is often good at
individual horizons, while magnitude is systematically distorted.

| Split / substep | Joint cosine | Joint norm ratio | Safety cosine | Safety norm ratio |
|---|---:|---:|---:|---:|
| Train, 15 | 0.960 | 0.909 | 0.961 | 0.957 |
| Train, 25 | 0.964 | 0.621 | 0.951 | 0.600 |
| Train, 26 | 0.945 | 4.622 | 0.953 | 0.559 |
| Train, 50 | 0.954 | 0.217 | 0.936 | 0.225 |
| Validation, 15 | 0.919 | 0.928 | 0.854 | 1.280 |
| Validation, 50 | 0.867 | 0.237 | 0.833 | 0.281 |
| Reserved, 15 | 0.895 | 0.785 | 0.812 | 0.925 |
| Reserved, 50 | 0.859 | 0.174 | 0.807 | 0.206 |

Substep `26` is where the second action begins to affect execution. The model
initially overestimates this new response, then severely underestimates both
actions by the terminal horizon. This same pattern appears on training,
validation, and reserved episodes; it is not created only by distribution
shift.

All `14 x 51` action-coordinate/horizon cells are stored in `result.json`.
At training substep `50`, the first-action joint norm ratios span
`0.291--0.448`, while the second-action ratios span `0.071--0.172`. Terminal
safety ratios span `0.231--0.472` for the first action and `0.107--0.207` for
the second. Translation and rotation both show attenuation, so rotation is
not a unique failure source. Gripper sensitivity direction has no valid
nonzero OSC arm-motion secant in this population and is recorded as undefined,
not silently counted as passing.

## Supporting fit and geometry evidence

The recomputed fixed-k0 geometry audit finds `122` training false-safes and a
`2.865 mm` training near-boundary RMSE. Validation has `140` false-safes and a
`7.751 mm` boundary RMSE. The exact joint trajectories passed through the same
static FK/ellipsoid evaluator have zero geometry false-safes in every split.
Thus the audit does not reopen FK or ellipsoid geometry.

The reserved population still has six recoverable states with exact-safe
candidates but no predicted-safe candidate. This is important for deployment,
but it is downstream of the training sensitivity failure and cannot by itself
justify a coverage-only conclusion.

## Decision

Keep the frozen job-`38376` model as strict NO-GO. Do not calibrate it or send
its gradients to a QP. The next permitted experiment is a matched retraining
of the same direct-horizon representation with:

1. normalized paired-action secant supervision that matches both sensitivity
   direction and magnitude at each action coordinate and horizon; and
2. the previously successful one-sided near-boundary geometry loss.

The matched retraining must first pass training and validation sensitivity
gates. It does not authorize calibration, QP, or closed-loop control.

## Reproducibility

- Slurm job: `38560`, H100 on `worker-1`, elapsed `01:02:44`.
- Clean source: `10b2e52ddd4d618288b81bc13cead94ad08187a9`.
- Config v2 SHA-256:
  `69c40e545aeff59a603b4afa6a95c01b5d6c0bea0c8f3370b52a4a7e0d915f49`.
- Artifact directory:
  `/mnt/data/quanth/experiments/vlsa-distal-direct-horizon-root-cause-audit/direct-horizon-root-cause-audit-20260811c`.
- `records.npz` SHA-256:
  `626cd8b16beb5f080f8c00bc5fd3e08c5697c73bcef20a2ff30df8e9d242eb19`.
- `result.json` SHA-256:
  `6bf2e196f88d79e7bb3c29cccb9bc6995db00356b00e330f5fd5751b5530e793`.
- Result payload SHA-256:
  `0520bed5ce3699a93e0bcca8c6ea769189647ff1757554bc9dd8829b28c14642`.
- `validation.json` SHA-256:
  `51ae572097e5d5174726baa6562023318af981c43038c59bb269cf5474cdf9b2`.
- Validation payload SHA-256:
  `16a6ad1ba7aab0f37ff5916c91157c54527f6b784c91bbc49e24fceae2bcfadf`.
- H100 allocation preflight: all ten audit-specific tests passed. Local
  `./init.sh` could not complete because the host system Python has no NumPy
  (24 import-only errors; no failed assertion in the changed audit).
