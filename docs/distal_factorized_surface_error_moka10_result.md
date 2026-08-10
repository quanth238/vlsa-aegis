# Factorized false-safe surface-error diagnostic: validated result

## Question

Do the 44 factorized false-safe actions contain larger L5--L7
obstacle-facing surface-position errors than same-state, boundary-matched
correctly rejected unsafe actions?

## Result

No. H100 job `38036` reproduced all 960 frozen E05/E10/E15 random-action
geometry records and all 44 source false-safes. The independent second pass
reproduced every record, metric, and decision exactly.

| Worst exact constraint metric | 44 false-safes | 44 matched unsafe controls | Difference |
|---|---:|---:|---:|
| Surface-position error, median | 4.538 mm | 4.639 mm | -0.101 mm |
| Center-position error, median | 4.468 mm | 4.561 mm | -0.093 mm |
| Signed surface clearance overestimate, median | 1.449 mm | 1.082 mm | +0.368 mm |
| Signed center clearance overestimate, median | 1.196 mm | 0.846 mm | +0.350 mm |
| Signed support/orientation contribution, median | -0.107 mm | -0.163 mm | +0.056 mm |
| Static minimum-margin overestimate, median | 2.351 mm | 2.060 mm | +0.291 mm |

The false-safe surface-position error won 72.7% of individual matched pairs,
and signed surface retraction correlated with margin overestimation at
Spearman `0.635`. Those two gates pass. However, the registered primary test
requires a population median excess of at least 0.5 mm. Instead, the
false-safe median is 0.101 mm smaller than the matched-control median.

All 44 false-safes have:

- worst protected body: `robot0_link5`;
- worst substep: `50`, the final state of the two-action horizon;
- predicted obstacle witness equal to the exact witness: `100%`.

Thus the errors are not caused by obstacle-witness switching, nor do the
false-safes form a population with uniquely larger Euclidean surface error.
The distinguishing signal is signed clearance bias at the final L5 horizon
state. Most of that bias comes from the link-center component, not the
ellipsoid support/orientation component.

## Decision

```text
surface_position_not_decisive_skip_surface_loss_and_audit_one_sided_uncertainty
```

The evidence does not authorize a generic symmetric surface-position-loss
pilot. It refines the next question toward terminal L5 one-sided error: why
does the model retract the predicted L5 surface from the obstacle by about
0.35--0.37 mm more than matched rejected actions at substep 50? A later
experiment may compare terminal L5 normal-direction supervision or a
state-conditioned one-sided bound, but no such training was performed here.

Poisson/SDF, calibration, QP, VLA inference, and closed-loop execution remain
unauthorized.

## Reproducibility

- Job: `38036`, `COMPLETED` in `00:12:16` on `worker-2`.
- Source commit: `231afc4357831603e85e7b12f23ecf95f1b00c09`.
- Records SHA-256: `b8856658c2dde292805b19b9902cf9b09fb8d9897f6d0100c2fe475149dc0187`.
- Result SHA-256: `263b51cf93cea9aee6b894962562229f8ad79ebd8ce1aa308bde873495e38b30`.
- Result payload SHA-256: `301e3c89ba802c46d1d131dc18f027a573f02bbe0177dc80267cebca4c26a0e7`.
- Validation SHA-256: `69b5e1b72ec17f8e5944ed719b9746b5c960a2f3e1e1328f165962a9409e70c6`.
- Validation payload SHA-256: `a96c4f1a060962610eaf8f0929b85920b6b9020ea2d7bd4fc2fbd8eb6da9ee58`.
- No new simulation label, training, surface loss, Poisson/SDF, calibration,
  QP, or closed-loop execution was performed.
