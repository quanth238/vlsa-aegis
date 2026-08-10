# Factorized OSC execution pilot: validated result

## Question

Can a state-conditioned model learn the complete two-action OSC joint rollout
and its action sensitivity more reliably than direct rollout-minimum margin
regression, then recover L5--L7 ellipsoid safety on unseen E05/E10/E15 states?

## Matched experiment

H100 job `37980` used immutable commit
`7062d70a5cc90f9689def2bfc4bf2bdd347b1c8e`. It collected 85 states from 17
episodes with a grouped 60/10/15 split; E05, E10, and E15 were test-only. Each
state contributed 93 complete two-action candidates: the nominal chunk,
positive/negative probes for all 14 action coordinates, and 32 antithetic
random pairs. Translation, rotation, and gripper were all varied. Every
candidate stored 51-by-7 joint positions and seven per-substep ellipsoid
margins. All 425 registered repeat rollouts were exact.

The direct arm predicted seven rollout-minimum margins. The factorized arm
predicted the complete joint trace and was evaluated through MuJoCo forward
kinematics plus the unchanged L5--L7 ellipsoids. Both arms shared states,
actions, features, splits, hidden widths, seeds, optimizer, and schedule and
received explicit finite-difference sensitivity supervision.

## Validated results

| Test metric | Direct margin | Factorized joints | Exact joints + static geometry |
|---|---:|---:|---:|
| Proxy false-safe actions | 0 | 44 | 0 |
| Exact-safe recall | 0.74% | 94.70% | 99.51% |
| States with accepted safe support | 1/15 | 13/15 | 13/15 |
| Near-boundary margin RMSE | 33.640 mm | 2.188 mm | 0.321 mm |
| Overall margin RMSE | 32.746 mm | 1.984 mm | 0.319 mm |

The factorized execution model additionally achieved:

- joint-trajectory RMSE: `11.560 mrad`;
- endpoint RMSE: `20.587 mrad`;
- joint absolute-error p95 / maximum: `26.337 / 78.529 mrad`;
- link-center RMSE / p95 / maximum: `3.538 / 6.300 / 13.288 mm`;
- joint-sensitivity cosine: `0.975`;
- action-space margin-sensitivity cosine: `0.913`.

The factorized predictor substantially improves margin accuracy, safe recall,
and support over direct minimum regression. It also learns useful local action
sensitivity. It nevertheless misses the 10 mrad joint-error gate, produces 44
ellipsoid false-safes, and lacks safe support in two test states. The direct
arm obtains zero false-safes only by rejecting almost every safe action.

Exact-joint recomposition has zero false-safes and 99.51% recall, so the
dominant remaining error in this pilot is joint-execution prediction rather
than the fixed-obstacle ellipsoid calculation. This does not establish that
the ellipsoids match every MuJoCo collision; raw contacts remain final physical
verification.

## Decision

The registered decision is:

```text
factorized_execution_pilot_NO_GO_or_inconclusive
```

The factorized research direction has useful mechanism evidence, but the
current state-conditioned MLP is not safe enough for intervention. Calibration,
QP construction, Poisson/SDF, VLA inference, and closed-loop E05 remain
unauthorized. The next model work should first localize joint/substep errors
behind the 44 false-safes and improve short-horizon execution prediction; the
geometry result does not justify adding Poisson/SDF now.

## Reproducibility

- Scientific job: `37980`, `FAILED` only because the original validation
  predicate rejected matching NaN masks; scientific result completed.
- Result SHA-256: `70c4dd81e7b20d03541d85ed392e0f552cd1e4b8b3e8f81d0c148f2fbe36b271`.
- Result payload SHA-256: `acbad04905abcd450b6e52cb7f2ee0c19a86e895016350dde3d07930b0c84477`.
- Validation-only job: `38025`, `COMPLETED` in `00:08:04` on `worker-2`.
- Validator commit: `059e8510878767439ac4edf4d10b991b90c93097`.
- Validation file SHA-256: `ed78d2f528010095f1bc4db85f73e160996db62e764f30122196d8fb8821ef1e`.
- Validation payload SHA-256: `e8a764e5b5be9e11807a64f2c773c363f9ca9ac4043a88915bc246d47206a9a4`.
- Recomputed maximum finite margin difference: `0.0 m`.
- Completed Table 1 artifacts were read-only.
