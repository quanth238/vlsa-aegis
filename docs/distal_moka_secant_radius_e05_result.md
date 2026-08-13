# E05 Moka compact-input secant-radius result

## Verdict

Strict **NO-GO**. Reducing the paired perturbation radius from `0.05` to
`0.025` and `0.0125` does not make the twenty-action action-to-safety
response reliably affine, even at the same E05 action-182 state with the
compact 25D input.

This experiment changed only physical perturbation magnitude. The exact 32
direction vectors, state, nominal action, first-five XYZ action coordinates,
twenty-action horizon, geometry, 24/8 fit/held-out split, model, loss,
optimizer, and seed were fixed. Directions were generated once using the
original radius-0.05 action-bound support mask and reused byte-for-byte at
both smaller radii.

## H100 result

Producer job `39305` completed on `worker-2` in `222.103 s` and evaluated
`129` cloned-OSC rollouts: one nominal rollout plus 32 positive/negative
pairs at each of two radii. Independent H100 validator `39307` reproduced
the hashes, direction binding, ranks, metrics, and preregistered gates.

| Radius | Method | Fit cosine | Fit sign | Held-out cosine | Held-out sign |
|---:|---|---:|---:|---:|---:|
| `0.05` | compact MLP reference | `0.880229` | `0.932143` | `0.408501` | `0.735714` |
| `0.05` | local ridge reference | — | — | `0.449661` | `0.782143` |
| `0.025` | compact MLP | `0.856972` | `0.935714` | `0.383901` | `0.748214` |
| `0.025` | local ridge | `0.905808` | `0.960714` | `0.646382` | `0.790179` |
| `0.0125` | compact MLP | `0.890541` | `0.929464` | `0.708877` | `0.824107` |
| `0.0125` | local ridge | `0.910353` | `0.927679` | `0.664620` | `0.769643` |

The preregistered held-out requirements were cosine at least `0.8` and sign
accuracy at least `0.85` for both the MLP and ridge teacher. No radius passed.
The six near-active MLP rows did agree with the within-radius ridge rows
(`0.963438` at `0.025`, `0.969130` at `0.0125`), and compact value RMSE
remained small (`0.652 mm`, `0.518 mm`). These favorable in-sample facts do
not rescue fresh-direction prediction.

The fitted response rows also changed materially between the two small
radii. Their mean cross-radius cosine was `0.592426` for ridge and `0.543534`
for the MLP; even the six near-active ridge rows averaged only `0.764505`.
Thus the failure cannot be assigned only to MLP capacity. The local affine
teacher itself is radius-sensitive and does not generalize to the eight
matched fresh directions.

## Interpretation

Input simplification was useful, but smaller finite secants do not repair the
full 140-row affine-response target. The `0.0125` MLP held-out cosine rose to
`0.708877`, which is meaningful progress, but its ridge ceiling was only
`0.664620` and both sign gates failed. This is evidence of a target/locality
or response-representation limitation, not authorization to enlarge the MLP,
add state inputs, or apply a QP.

The next clean prediction experiment should stop asking one locally affine
140-row field to explain all fresh directions. Test a direction-conditioned
scalar response model on the same immutable rollouts, or directly compare a
small nonlinear local action-value model against ridge. Keep the compact
physical context and evaluate fresh directions before any correction or
closed-loop experiment.

## Provenance

- Code commit: `b91d81d6815855de06ad0e4027ee80e1275cc21e`
- Producer: Slurm `39305`, `worker-2`
- Validator: Slurm `39307`, `worker-2`
- Result: `/mnt/data/quanth/experiments/vlsa-distal-moka-secant-radius-e05/secant-radius-20260813a/result.json`
- Validation: `/mnt/data/quanth/experiments/vlsa-distal-moka-secant-radius-e05/secant-radius-20260813a/validation.json`
- Result file SHA-256: `cb7e2aeb65a031f7b665b4aad2d5fa2840f313e51fa4ac52e0752a321dca79a6`
- Result payload SHA-256: `d74bd14ea877dd9e9b0ae350909691bb23425160d898868b30e8860139aada8e`
- Validation file SHA-256: `0a89b39a331443fec92bd4a945373865f861c271d8802797e565b23c8b8222a2`
- Validation payload SHA-256: `72e9392118ade961c937e50a742e74c63a693a4f3b7f5143671760f17f96e1f7`

No QP was solved, no corrected action was executed, and no task or safety
claim is made from this prediction-only one-state ablation.
