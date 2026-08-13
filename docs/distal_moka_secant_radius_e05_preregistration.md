# E05 Moka compact-input secant-radius ablation

The validated compact-input experiment confirmed that the old 1,055D input
was harmful, but both the compact MLP and local ridge teacher failed eight
fresh directions at perturbation radius `0.05`. This experiment changes only
the physical secant radius.

At the identical E05 step-182 state, generate the 32 direction vectors once
using the original `0.05` action-bound support rule. Reuse their exact values
for paired cloned-OSC rollouts at radii `0.025` and `0.0125`. This prevents a
smaller radius from silently changing the free action coordinates or sampled
direction subspace.

For each radius independently:

1. collect the same 24 fit and eight held-out directional responses;
2. fit the same ridge teacher;
3. train the same compact 25-input, 128-unit MLP with the unchanged value and
   paired-response losses, optimizer, seed, and checkpoint rule;
4. report fit and held-out cosine, sign, RMSE, row cosine, near-active L5 row
   cosine, ridge conditioning, and cross-radius row stability.

The action horizon remains 20 and the correction horizon remains the first
five XYZ commands. Geometry, base state, nominal actions, and all model
settings remain fixed.

A radius passes only when both the teacher and MLP have held-out cosine at
least `0.8` and sign accuracy at least `0.85`, the MLP fit cosine/sign reach
`0.9/0.85`, and near-active row cosine to ridge reaches `0.9`. If neither
radius passes, shrinking the secant does not rescue one affine 15D response;
the next representation should predict direction-conditioned scalar outcomes
or a nonlinear local action value. No QP or corrected action is authorized.
