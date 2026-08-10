# Factorized terminal-bias audit result

## Verdict

H100 job `38072` independently validates a strict **NO-GO for the proposed
local-Jacobian one-sided-loss ablation**. The terminal bias is not confined to
the repeatedly inspected E05/E10/E15 episodes: it is present in train and
validation as well. The five-member ensemble does not identify false-safes by
high disagreement, and the local geometry Jacobian is not valid over the
model's own larger joint-prediction errors.

No training, new controller rollout, calibration, QP, closed loop, surface or
classification loss, or Poisson/SDF ran.

## Residual localization

The immutable job-37980 ensemble mean, test exact-q static margins, and test
predicted-q static margins reproduced exactly. The audit then reconstructed
all 51-by-7 exact and predicted static clearance traces for the 64 out-of-fit
random actions at all 85 states.

| Split | False-safes | Exact worst location | Terminal L5 median optimism | Terminal L6 median optimism |
|---|---:|---|---:|---:|
| Train | 27 | 27/27 at row 1 (L5), substep 50 | 0.660 mm (66.1% optimistic, n=1,343) | 1.798 mm (100%, n=128) |
| Validation | 103 | 103/103 at row 3 (L6), substep 50 | 9.572 mm (100%, n=122) | 154.476 mm (100%, n=128) |
| Diagnostic E05/E10/E15 | 44 | 44/44 at row 1 (L5), substep 50 | 1.623 mm (71.4%, n=248) | 3.818 mm (77.4%, n=106) |

Counts `n` contain terminal values whose exact clearance is within 5 mm of
the boundary. L7 has no terminal near-boundary samples. Under the frozen
material-bias gate, L5 and L6 are optimistic in every split. The link identity
of the exact false-safe changes between validation and the other splits, but
the common failure is terminal-horizon optimism. This supports a training
objective or flat-decoder failure, not a test-only state-coverage explanation.

## Geometry-signal audit

The exact central finite-difference geometry is highly accurate for actual
nearby controller trajectories:

- near-boundary local clearance linearization RMSE: `0.0496 mm`;
- signed clearance-delta cosine: `0.999738`;
- signed-delta sign agreement: `0.999966`;
- composed-vs-direct action-gradient median/p05 cosine:
  `0.999999 / 0.999959`.

However, two mandatory gates fail:

- candidate-resampled action-gradient median/p05 cosine is
  `0.941970 / 0.739169`, below `0.95 / 0.8`;
- using the nominal local Jacobian to approximate clearance error induced by
  the model's own joint-prediction error has `89.55 m` near-boundary RMSE.

The latter is a failed extrapolated linear estimate, not a physical clearance
measurement. It shows that the model's prediction error leaves the tiny local
region where `dh/dq` is accurate. Therefore the fitted local Jacobian cannot be
used as a trustworthy training signal for this flat decoder even though it is
excellent for actual small controller-action changes.

## Ensemble disagreement

Exact FK/ellipsoid minima were evaluated for each of the five members on all
174 false-safes and 104 unique same-state matched controls:

- false-safe median disagreement: `3.622 mm`;
- control median disagreement: `3.992 mm`;
- ratio: `0.907`;
- false-safe-detection AUROC: `0.344`.

False-safes do not have elevated ensemble disagreement. This is systematic
bias; ensemble uncertainty calibration is not supported as the next repair.

## Decision

Do not run the matched one-sided geometry-loss ablation, calibration, QP, or
closed-loop E05. Replace the flat 51-by-7 output decoder with a shared
time-conditioned execution model

\[
\hat q_k=F_\theta(x,A,k)
\]

and first train/evaluate it with the existing joint and sensitivity targets.
Rerun this residual audit before adding a geometry loss. Any final
generalization claim must use newly reserved complete episodes; E05/E10/E15
remain diagnostic failure cases only.

## Provenance

- Slurm job: `38072`, H100 `worker-2`, wall time `00:39:26`.
- Source commit: `8e18f249e0ac9a7a447b616e08d533f6f1a8f86f`.
- Records SHA-256:
  `c2c43f6c9fc0b2c6708af7766bcada78f5128ca955a0dccdb1f110d0a4ebae8e`.
- Result SHA-256:
  `8724b36b2c5abfe6d17b6bd230c4ce31aa3040346322662e3c07bfd25b4b31df`.
- Result payload SHA-256:
  `ec05e3af4cfc4897da7c024e4dd82ca73171fc715053ea2bcd13d613af432ec6`.
- Validation SHA-256:
  `9124409b40a94b4280375f13be58aae0266fa26d45091d16866a868c1b09d4ea`.
- Validation payload SHA-256:
  `fefc04db017a1302344df09b065b2e56c6b999f4a71154f1b681f71d5c676e67`.
