# Action-conditioned execution-margin MLP result

## Verdict

**Strict NO-GO for the plain action-conditioned MLP.**

Clean H100 job `37807` completed on `worker-2` in `00:01:19`. The independent
validator accepted the artifact. The unseen off-grid gate failed, so no QP
proposal was executed and closed-loop E05 remained blocked.

## Result

| quantity | result | required |
|---|---:|---:|
| direct margin RMSE | 3.808 mm | diagnostic |
| current-clearance baseline RMSE | 16.513 mm | diagnostic |
| test false-safes | 21 / 1,440 | 0 |
| global accepted-set Jaccard | 0.7866 | at least 0.90 |
| worst-state Jaccard | 0 | at least 0.80 |
| states with accepted exact-safe support | 12 / 15 | 15 / 15 |
| valid learned QPs | 0 / 15 | 15 / 15 |
| fresh exact-safe selected QPs | 0 / 15 | 15 / 15 |

The value parameterization is materially better than the coefficient model:
RMSE is 4.34 times lower than the no-motion/current-clearance baseline, and
12 states retain safe actions. It is not conservative enough to deploy.

All 21 false-safes occur at the decisive E05 step 185: the exact regional
oracle accepts 20/96 actions, while the learned model accepts 41/96. At the
other difficult boundary states the opposite failure appears. It accepts no
action at E10 step 156 and E15 steps 149--150, despite oracle support of
21, 89, and 54 actions. Thus one validation-calibrated global MLP is
simultaneously overconfident at one unseen boundary and overly conservative at
three others.

## Interpretation

The experiment supports learning quantitative executed margins rather than
non-unique affine coefficients, but rejects this plain model as the safety
field used by the QP. Adequate context support and lower average regression
error do not guarantee a conservative per-state lower bound near zero.

OSQP and cloned execution are not the present failure: the registered protocol
correctly stopped before those stages because the learned constraints already
failed the unseen decision gate. Closed-loop E05 cannot be claimed.

No additional experiment is needed to reject a single larger global pad. A
nonnegative tightening can only remove currently accepted actions. Because the
model already has support in only 12/15 states at zero additional tightening,
no scalar can both remove the 21 false-safes and restore 15/15 support.

The next defensible direction is state-conditioned uncertainty or a
nonparametric local residual: E05 step 185 needs more tightening, while E10
step 156 and E15 steps 149--150 need less conservative bounds. Any such method
must be calibrated with leave-one-episode-out groups and pass the same untouched
E05/E10/E15 gate before QP or closed-loop execution.

## Immutable evidence

- run root: `/mnt/data/quanth/experiments/vlsa-distal-action-conditioned-margin/action-conditioned-margin-20260810c`
- result file SHA-256: `fc61cbf8b936447c591897ebf47f0606e135a8ff9df8d878d562bf37fc3a32da`
- result payload SHA-256: `9d345936fc6c4d1e3677de6223345db8e867a4571b3f449290fcf2aa208736d0`
- validation file SHA-256: `255df4eeb454e8d6de574517f0d1244d5d9588c18e9978222c93f113ec6ef704`
- model file SHA-256: `bfc1dd8dc4f3fcdb9c3268ce268a2f252fa31cadc86b72fb6c7d98f53c6346c3`
- source commit: `6579c0d8b8cc7c451854cf495a6b5c4804f3773f`
