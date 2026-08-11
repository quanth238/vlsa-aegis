# Structured bounded-orientation ablation result

## Verdict

Clean H100 job `38095` is an independently validated strict **NO-GO**.

The structured input fixes the confirmed numerical pathology: maximum
validation magnitude falls from `2e6` to `355.37`, and flat/time validation
terminal joint RMSE becomes physical at `33.39/37.35` mrad rather than roughly
`1,897/1,630` rad. This does not make either execution model safe on the
diagnostic test episodes.

| Test metric | job 38070 flat | structured flat | job 38076 time | structured time |
|---|---:|---:|---:|---:|
| False-safes | **0** | 64 | **0** | 63 |
| Exact-safe recall | **93.84%** | 63.05% | 47.29% | **63.05%** |
| Eligible-state support | **12/13** | 8/13 | 6/13 | **8/13** |
| Near-boundary RMSE | **2.671 mm** | 4.498 mm | 15.700 mm | **5.246 mm** |
| Terminal joint RMSE | **18.158 mrad** | 34.488 mrad | **22.889 mrad** | 29.287 mrad |

Structured time decoding improves job `38076` in boundary RMSE and support,
but worsens terminal joint error and introduces 63 false-safes. Structured flat
decoding is worse than job `38070` on every registered test safety gate.

The action sensitivity signal remains useful but is not conservative:

- flat joint/margin sensitivity cosine: `0.936/0.859`;
- time joint/margin sensitivity cosine: `0.910/0.912`.

## Interpretation

Two distinct findings must remain separate.

1. Duplicated scalar rotation standardization caused the impossible validation
   explosion. The bounded 6D representation repairs that implementation issue.
2. Stable inputs do not solve the execution-safety prediction problem. The
   current one-sided objective and either decoder still overestimate clearance
   on unseen diagnostic states.

The old flat model's zero false-safes cannot now be treated as evidence that
its representation was correct. Its checkpoint selection was exposed to the
exploding validation episode, so some of its conservatism may have been an
accidental consequence of the corrupted validation objective. Once validation
becomes numerically physical, the same loss no longer produces a conservative
test model.

The result rejects this package:

\[
\boxed{\text{bounded 6D orientation + current one-sided loss + plain flat/time decoder}}
\]

It does not reject factorized OSC execution learning in general. Both models
still learn useful action sensitivities, but neither can supply trustworthy QP
constraints.

## Next diagnostic

Do not train longer, calibrate, solve a QP, or run closed loop. First localize
the 63/64 structured-model false-safes by split, state, link, and substep, and
compare member disagreement and signed residuals with correctly rejected
near-boundary actions. This determines whether the stable-input failure is:

- the same terminal L5/L6 optimism already observed;
- a clean-validation versus diagnostic-test coverage shift; or
- inconsistent ensemble/checkpoint selection.

Only after this read-only localization should the next model objective be
chosen. If optimism is already present on supported train/validation states,
use a safety-constrained validation/checkpoint criterion or structured
dynamics model. If it appears only on unsupported diagnostic states, collect
new grouped boundary episodes before another architecture.

## Reproducibility

- Allocation: job `38095`, `worker-2`, one H100, `00:55:05`.
- Immutable run:
  `/mnt/data/quanth/experiments/vlsa-distal-factorized-structured-orientation/structured-orientation-20260811b`.
- Flat/time model SHA-256:
  `3f2ed45f269d8b43ef929ac75311758c974192d46370074e2ddb312414f11287`,
  `625061b23839669d1c36ad2a29cde232b5b06dfa9bb3636326bc0a45214ffb31`.
- Prediction/result/validation SHA-256:
  `3b06fe181996c9719f6ee50f704c9b2f55d0ac7689e30bb190bca8a663398eac`,
  `eeac18ca69aac9dc498b33b547418d987a6ac6fb572fbf02a5529dc771c070bd`,
  `cd170dcbfba441145983a3c3b6aaecbf6302fe250dc693337d1d3d313a3d77c9`.
- Result/validation payload SHA-256:
  `aef1a4353c8e134c464ab50d6a4c9bbdd9fd3f2cdec02920e9e5fe29bb95651b`,
  `5ce3a65712a1fd47d18bd9baae1eb95d9c8f72520d17bdb32b36d38baeb23f01`.
- Independent replay: all nine checks pass; maximum joint and margin
  differences are exactly `0.0`.

Job `38094` is apparatus-only: it stopped before training because a source
decision receipt was read at the result root instead of under `audit`.
