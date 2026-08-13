# E05 Moka compact-input memorization result

## Verdict

Input simplification was necessary and materially improved the model, but it
was not sufficient. The 25-input, one-state model nearly memorized the fitted
response field and represented the six near-active L5 witnesses well. It did
not predict fresh paired directions reliably. The same failure is present in
the local ridge teacher, so state inputs, a QP, and larger corrections remain
premature.

Clean H100 producer `39295` completed 65 cloned-OSC rollouts at original train
step 182 on `worker-2` from commit
`f3f7188956e3129287deba90e465345926b48659`. Independent H100 validator
`39298` reproduced every registered gate.

## Controlled change

The rejected model used 1,055 inputs. The ablation used only:

- nominal first-five XYZ actions: 15 values;
- three action-offset features;
- seven-way robot-row identity.

This produces 25 model inputs. Architecture, optimizer, seed, value loss,
paired-response loss, response weight, 32 directions, `+0.05/-0.05`
perturbation, geometry, and the 20-action outcome horizon were unchanged.

## Evidence

| Metric at step 182 | Original 1,055D MLP | Compact 25D MLP | Local ridge |
|---|---:|---:|---:|
| fit response cosine | 0.490 | **0.880** | 0.926 |
| fit sign accuracy | 0.759 | **0.932** | 0.949 |
| held-out response cosine | 0.506 | 0.409 | 0.450 |
| held-out sign accuracy | 0.719 | 0.736 | 0.782 |
| value RMSE | 32.75 mm | **0.70 mm** | n/a |
| all-row cosine to ridge | n/a | 0.862 | 1.0 |
| six near-active rows cosine to ridge | about 0.532 | **0.932** | 1.0 |

The compact model selected epoch 705 rather than epoch 8. It therefore had no
difficulty optimizing the value head and learned most fitted response rows.
This directly confirms that the old redundant input was a major source of
underfit.

The strict gate nevertheless failed. Fit cosine missed `0.9`, all-row cosine
to ridge missed `0.9`, and fit RMSE was `1.265` times the ridge RMSE. More
importantly, held-out cosine remained only `0.409`. The ridge teacher itself
reached only `0.450`, despite a full-rank design, zero critical primitive
switching in the prior audit, and useful sign accuracy. Its design condition
number was `480.85`.

## Root cause

The result separates two causes:

1. **Confirmed representation problem:** 1,055 redundant inputs prevented the
   old MLP from even fitting values and training responses. Compact inputs
   repair most of that failure, especially for the active L5 rows.
2. **Remaining target/locality problem:** a single affine response fitted from
   `+0.05/-0.05` changes over a 20-action OSC rollout does not transfer well to
   eight fresh directions, even without neural approximation.

This does not support adding `q`, `qdot`, geometry, a QP, or closed-loop
control yet. Those additions cannot repair a teacher that is unreliable on
fresh directions at one fixed physical state.

The next one-factor experiment should keep the compact representation and
repeat the same step-182 test at smaller paired perturbations, for example
`0.025` and `0.0125`, with independent fit and held-out directions. If ridge
held-out cosine improves substantially, retrain the compact model at the
validated local radius. If it does not, replace full 15D gradient regression
with direction-conditioned scalar response prediction or a nonlinear local
action-value model.

## Provenance

- Producer result file SHA-256:
  `a55fd62fe391724c662296ae7e59f67328f163c3980486b37465fdec7955aab1`.
- Producer result payload SHA-256:
  `5384c6893e0fefaa6e19bd59c8e13c210a1fe25e04a9e18eb00468325281ecae`.
- Validation file SHA-256:
  `f5ca20f14e70f16b5978067e95382eb1e9edf3424b39d78f5cbfab532656db39`.
- Validation payload SHA-256:
  `6c2a4c50561026d0c72276892653c948a3ed001949eccd402baf901d15609492`.

Attempt `39293` stopped before simulator construction and training because
`worker-0` exposed a zero-byte `nvidia-smi`. It produced no scientific result.
