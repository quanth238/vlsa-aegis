# E05 Moka local nonlinear prediction result

## Verdict

Strict **NO-GO** for both candidate control representations. The
direction-conditioned scalar arm exposes a potentially useful paired-ranking
signal, but it does not provide a reliable link/time safety-response field.
The nonlinear absolute-margin arm is worse and should be rejected in its
current form.

Both models used the identical compact 25D witness representation plus one
15D direction or signed action-delta query. They shared the same 21,889-
parameter scalar architecture, optimizer, seed, E05 action-182 state,
radius-`0.0125` rollouts, geometry, and twenty-action horizon. Directions
0--19 trained, 20--23 selected checkpoints, and 24--31 remained untouched.

## H100 result

Producer job `39312` completed on `worker-2` in `130.628 s`, replaying `65`
cloned-OSC rollouts. Independent H100 validator `39313` reproduced the
source, reference, directions, split, parameter equality, metrics, hashes,
and every gate.

| Untouched-direction metric | Direction-conditioned scalar | Nonlinear action value | Required |
|---|---:|---:|---:|
| All-witness response cosine | `0.067122` | `0.386933` | `>=0.8` |
| All-witness sign accuracy | `0.653571` | `0.586607` | `>=0.85` |
| Near-active response cosine | `0.609441` | `0.063869` | `>=0.8` |
| Near-active sign accuracy | `0.833333` | `0.479167` | `>=0.85` |
| Safer +/- branch accuracy | **`7/8 (0.875)`** | `3/8 (0.375)` | `>=7/8` |
| Worst-margin RMSE | **`0.073 mm`** | `1.224 mm` | `<=1 mm` |

The direction-conditioned model learned the near-active training equations
well (`0.945768` cosine and `0.991667` sign), but this fell to `0.852751 /
0.666667` on validation and `0.609441 / 0.833333` on untouched directions.
It therefore learned some branch preference but not a transferable physical
response vector.

The small worst-margin RMSE is not evidence of a useful continuous safety
model. A read-only zero-response baseline that predicts the nominal worst
margin for both branches has `0.070 mm` RMSE on the same 16 branch outcomes,
slightly better than the direction-conditioned arm's `0.073 mm`. The
informative result is its `7/8` branch ordering, not absolute-margin fidelity.

The nonlinear value model fit training response moderately (`0.850427`
cosine), but its near-active response was already poor (`0.462749`) and
collapsed on fresh directions. Direct absolute-margin regression again
rounded away the small action-dependent difference that steering needs.

## Interpretation

Changing from a 15D affine row to a direction-conditioned scalar does not
solve response regression. Predicting absolute local values is still less
appropriate: large common margin structure dominates the small difference
between nearby actions. However, paired branch ranking may be easier and more
aligned with choosing a repulsive proposal than reconstructing all 140
response rows.

The next justified prediction-only experiment is therefore a matched paired-
preference test, not a QP or gradient controller: learn which of `+v/-v`
improves the hard or smooth future-risk aggregate, compare with random chance
and the analytical repulsion direction, and test on additional grouped
states. It must preserve quantitative margins for evaluation, but should not
claim a differentiable control field unless its continuous directional
response also passes.

## Provenance

- Code commit: `86d23589bf07f07d6bf4be372ca5f095b34325e5`
- Producer: Slurm `39312`, `worker-2`
- Validator: Slurm `39313`, `worker-2`
- Result: `/mnt/data/quanth/experiments/vlsa-distal-moka-local-action-value-e05/local-value-20260813a/result.json`
- Validation: `/mnt/data/quanth/experiments/vlsa-distal-moka-local-action-value-e05/local-value-20260813a/validation.json`
- Result file SHA-256: `6d9d49bb7511be8ee325ebbcf94956724ccc7fd3c3cdab0319688d48d7bbbec4`
- Result payload SHA-256: `95937f109842d3adf80a38db2501b81b2b947d7e6a050004ec55a3a94761bef7`
- Validation file SHA-256: `a691953a2f464738a8f1bc06a1c1fb5b21281ed78b2141a9fb9b229d450a7248`
- Validation payload SHA-256: `19cab581d87761445807dd512967aeeabd41255ddb64cef6882e6d5e5317a29b`

No QP was solved, no corrected action executed, and no task, collision-
prevention, state-generalization, or formal-safety claim is made.
