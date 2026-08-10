# Distal local residual-bound result

## Verdict

**Strict NO-GO.** The state/action-local nearest-neighbor bound recovered
exact-safe support in all 15 unseen states, but it did not achieve the required
zero false-safe gate. Closed-loop E05 remains unauthorized.

## Validated H100 evidence

- Job: `37829`, `worker-2`, `00:13:35`, clean commit
  `9c0cb8b31d9104aab55252fae99d82ef9c06aac0`.
- Out-of-fold population: 12 complete training episodes, 60 held states,
  52,500 action/constraint residuals.
- Test population: 15 untouched E05/E10/E15 states and 1,440 off-grid actions.
- Direct action-margin RMSE: `3.808 mm`.
- Accepted exact-safe actions: `1,182`.
- Safe-support states: `15/15`.
- False-safe actions: `13`.
  - E05 step 185: `6`.
  - E10 step 155: `5`.
  - E10 step 156: `2`.
- Validation also failed conservatism: `40/960` false-safes and `9/10`
  supported states.
- Maximum/mean dangerous OOF residual: `10.023/-2.376 mm`.

The preliminary gate failed, so the registered protocol correctly ran zero
regional QP simulations and zero closed-loop actions. This result does not say
that no safe action exists: every test state retained accepted exact-safe
support. It says that the fixed 256-neighbor maximum plus 1 mm is not a valid
conservative lower bound.

## Interpretation

The useful mean predictor is retained as evidence, but the proposed uncertainty
model is rejected. Treating 52,500 correlated action/constraint residuals as a
large calibration population does not create 52,500 independent episode-level
samples. With only 12 training episodes, a claimed conditional 0.999 bound has
very weak group-level statistical support.

Before another model or closed-loop run, a no-training diagnostic should
separate two possible sources of the 13 decisions:

1. pointwise local lower margins are themselves false-safe; or
2. pointwise margins are conservative but the regional affine interpolation
   becomes false-safe between the fixed grid points.

Only after that attribution should the next estimator be frozen. A new margin,
neighbor count, or threshold must not be selected from E05/E10/E15 outcomes.

## Artifact identity

- Result file:
  `9b51577b5c3d87265422ea6be0697ea72bdacfe3df3572e3a3bd524f410cdf9d`
- Result payload:
  `bdd9b7a73e6e2a35325112a4f090aa5f510fead4de11f7ed27a29fcdf1f9acc9`
- Independent validation:
  `c55e3e40d784b60a0096195caedb1f2dfcc084271b3b1c723d405d64107e66bc`
- OOF residual artifact:
  `5d6feee53da4f9b800906372df70a0f886db4d15ff3e3eff49c1f0b946b20308`
- Immutable root:
  `/mnt/data/quanth/experiments/vlsa-distal-local-residual-bound/local-residual-bound-20260810b`
