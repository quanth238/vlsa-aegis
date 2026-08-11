# Direct-horizon OSC displacement result

## Verdict

**Strict NO-GO at the prediction gate.** Direct per-horizon displacement is a
substantial mechanism improvement over cumulative increments, but the current
state-conditioned MLP is not sufficiently accurate or conservative on the
new reserved episode groups. Residual calibration, QP correction, and closed
loop are therefore not authorized.

## Experiment

The model predicts each displacement independently,

\[
\Delta\hat q_k=F_\theta(x_{\rm OSC},A,k),\qquad
\hat q_k=q_0+\Delta\hat q_k,
\]

with architectural `Delta q_0 = 0` and no cumulative integration. It uses the
complete structured OSC state, both full 7D Cartesian actions, direct joint
displacement targets, finite-difference sensitivity supervision, and symmetric
near-boundary safety-normal supervision. Known MuJoCo FK and the frozen L5--L7
ellipsoids compute safety.

Training uses only the existing grouped train/validation episodes. The final
prediction test uses 25 states and 2,325 actions from newly reserved
goal-II-task-2 E05/E20/E25/E30/E35. The future task-3 intervention episodes
remain unopened.

## Validated results

| Metric | Direct horizon | Previous one-sided factorized | Cumulative increments | Required |
|---|---:|---:|---:|---:|
| Reserved false-safes | 64 | 0 | 192 | 0 |
| Exact-safe recall | 71.43% | 44.94% | 77.16% | >=90% |
| Recoverable-state support | 15/21 | 10/21 | 17/21 | 21/21 |
| Near-boundary RMSE | 9.650 mm | 21.721 mm | 75.502 mm | <=2.671 mm |
| Joint sensitivity cosine | 0.736 | 0.846 | 0.628 | >0.8 |
| Safety sensitivity cosine | 0.669 | 0.492 | 0.337 | >0.8 |
| Overall joint RMSE | 20.728 mrad | 33.811 rad | 381.675 mrad | <=12 mrad |
| Terminal joint RMSE | 29.494 mrad | 55.133 rad | 692.883 mrad | <=20.587 mrad |
| Horizon RMSE slope | 0.446 mrad/step | 923.410 mrad/step | 13.640 mrad/step | <=0.5 mrad/step |
| Exact q0 | Yes | No | Yes | Yes |

The previous one-sided model is numerically out of distribution on these new
episodes; its extreme joint errors make its zero false-safe count a rejection
effect, not a useful safety result.

The static-geometry decomposition passes: exact rollout joints evaluated
through the same FK/ellipsoid backend have zero false-safes, 100% safe recall,
and 0.251 mm near-boundary RMSE. The dominant failure is therefore predicted
OSC execution, not FK or the ellipsoid safety calculation.

Direct horizons solve the cumulative-drift pathology: terminal error falls
from 692.883 mrad to 29.494 mrad and the slope passes the preregistered limit.
They do not solve unseen-state boundary accuracy or action sensitivity.

## Independent validation

H100 job `38462` reloaded the immutable model and reproduced:

- all four stored prediction arrays with identical NaN masks and zero finite
  difference;
- all metrics and the complete gate decision exactly;
- all 2,325 reserved predictions.

The validated artifact identities are:

- model: `d9aaa72812fae83271a451aef136e8ed98a00fce7cc882f6f2bad67876b0ce29`;
- predictions: `191cbb857b00f9292ebbda9ec891432e2184d41cd58ee0a10e001cf6a67028fe`;
- result: `6ab198218aa3903698b516be5e9748dfeca7748456ac1d315df8b85abcc69544`;
- validation: `c676fd5fee6de2130047a7362f1a7f78b8d07c62abaf81318304b6a08a031c7b`.

## Decision

Stop this experiment here. Do not learn the optimistic-residual envelope, run
a QP, or execute closed loop. The evidence supports factorized execution as a
useful representation and direct horizons as the correct alternative to
recursive increments, but rejects the present plain MLP/loss/data combination
as a deployable safety predictor.
