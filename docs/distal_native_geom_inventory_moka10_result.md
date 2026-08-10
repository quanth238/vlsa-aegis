# Native MuJoCo L5--L7 geometry inventory result

## Verdict

**Strict NO-GO for the 1 m `mj_geomDistance` target definition.**

The compiled inventory itself is clean, but the proposed quantitative target
is not contact-consistent in MuJoCo 3.2.3 and cannot authorize target
collection or learning.

## Validated evidence

- H100 job: `37863`, `worker-2`, `00:07:29`.
- Clean commit: `2773e031e87d44991c06987db9b5986f0b17a0e4`.
- Population: 85 states from 17 complete episodes, split 60/10/15.
- Physical protected groups: exactly three and semantically identical in all
  episodes:
  - `robot0_link5_collision`;
  - `robot0_link6_collision`;
  - `robot0_link7_collision`.
- Active moka-pot collision geoms: 15.
- All distance queries were finite and below the 1 m cutoff.
- All 53 primary raw-contact observations mapped to registered physical pairs.

The strict gate nevertheless failed:

- only 56/85 starts had nonnegative queried distance;
- only 8/15 test starts had nonnegative queried distance;
- 29/85 starts, including E10 steps 155--156 and every registered E15 state,
  had negative queried distance;
- none of the 15 test nominal two-action rollouts reported raw protected
  contact, including those with negative queried distances;
- direct signed distance differed from individual solver contact distances by
  roughly 41--47 mm at the E05 witness, so the frozen 10 micrometre equality
  gate failed.

All 85 historical proxy-feature receipts differed under current canonical
replay (maximum feature-space absolute difference 1.743959). Therefore old
proxy margins must not be attached to newly collected physical labels.

## Interpretation

The equality check was too strong because one geom-pair query returns a global
minimum while `mjContact.dist` is attached to an individual solver contact.
More importantly, MuJoCo documents that `mj_geomDistance` with a large positive
`distmax` can be approximate and inaccurate for pairs handled by the general
convex collider. The installed version is 3.2.3, and the observed negative
distance/no-contact cases are direct evidence that the 1 m query cannot serve
as the authoritative native safety margin for this mesh/box population.

Consequently, the 29 negative starts must not yet be called physical
collisions. They are negative under the rejected query definition.

## Decision

- Do not collect physical targets using the 1 m signed-distance query.
- Do not train the action-conditioned model.
- Do not solve physical-row QPs or run closed-loop E05.
- Preserve the valid compiled inventory: three protected geoms and 15 obstacle
  geoms.
- Next test a contact-consistent near-boundary query using `distmax=0` for
  overlap and a preregistered sequence of small positive cutoffs for safe
  clearance. Require sign agreement with raw contacts and eliminate negative
  distance/no-contact states before collecting a dataset.

## Artifacts

- Result SHA-256:
  `555f463d26d54b0b8508ef78548a6e97bbefa2a7bdd6c33b9140d3de941b6553`.
- Result payload SHA-256:
  `a36d01a5165ae0edad56d41ac240b1f15e88209d8cdad1a7dbe718cb94317a33`.
- Validation SHA-256:
  `07787aa5594bbaa1a8656425a5715e3ef16b08221394a0b5c5a9954744111a68`.
- Preflight SHA-256:
  `cb8040628b2c83c25c4c869069795d4995196b1d9564cbb25c0341b725419dfe`.
- Run root:
  `/mnt/data/quanth/experiments/vlsa-distal-native-geom-inventory/native-geom-inventory-20260810e`.
