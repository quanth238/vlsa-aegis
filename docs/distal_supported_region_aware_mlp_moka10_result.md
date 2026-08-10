# Supported-state region-aware MLP result

## Verdict

**Strict NO-GO for the coefficient-output MLP; GO to preregister the
action-conditioned safety-value replacement.**

Clean H100 job `37795` completed on `worker-2` in `00:03:10` with exit code
zero. The independent validator marked the artifact scientifically valid.
Closed-loop E05 did not run.

## Decisive result

The immutable expanded population has full state support: all 15 held-out
E05/E10/E15 states passed the training-derived support and oracle-smoothness
gates in job `37771`. Nevertheless, the unchanged five-member regional
coefficient MLP accepted no held-out action:

| quantity | result | required |
|---|---:|---:|
| held-out actions | 1,440 | 1,440 |
| exact-safe actions | 1,236 | diagnostic |
| regional-oracle accepted actions | 1,235 | diagnostic |
| learned accepted actions | 0 | nonzero in every state |
| learned false-safes | 0 | 0 |
| supported held-out states | 0 / 15 | 15 / 15 |
| accepted-set Jaccard, global / worst state | 0 / 0 | at least 0.90 / 0.80 |
| valid learned QPs | 0 / 15 | 15 / 15 |
| fresh exact-safe QP rollouts | 0 / 15 | 15 / 15 |

The zero false-safe count is therefore vacuous: the learned lower bounds
reject every candidate. Validation-only regional/constraint calibration was
`1.000--10.563 mm` (mean `1.980 mm`), but no learned proposal existed even
after the input-support blocker was removed.

## Interpretation

This experiment isolates the parameterization. The same supported states and
actions are accepted by the validated fixed multi-region oracle, while the
network that regresses regional affine anchors and gradients has zero recall.
The failure is therefore no longer attributable to missing E05/E10/E15-like
state coverage, OSQP, or exact-rollout availability. It is evidence against
the current coefficient-output learning target, not against execution-aware
learning or the multi-region QP oracle.

The registered next method predicts each exact two-step L5--L7 safety value
directly as a function of state, constraint, and candidate action. Fixed
regions remain QP trust regions, but non-unique affine coefficients are not
supervised. Closed-loop E05 stays blocked until that model has zero held-out
false-safes, safe support in all 15 states, valid regional QPs, and fresh exact
safe rollouts.

## Immutable evidence

- run root: `/mnt/data/quanth/experiments/vlsa-distal-supported-region-aware-mlp/supported-region-aware-mlp-20260810a`
- result file SHA-256: `0c796022cc748a190f913780d7a722be68cd2233a13b05f0262ef9d63d648243`
- result payload SHA-256: `47eb8884050879f4fd95add3fe06d4739adf2792473c741932f4e5078549d70b`
- validation file SHA-256: `727dc69cf5ae5c08d1a11b07883132c8b21d19b67f984be476c269a0dc9afeb3`
- model file SHA-256: `1743cea3873b746389fc846bba49337cc14e3e281dd4e9731a3321232e0b7256`
- validation-only rollout payload SHA-256: `240378006c46360b6a4c4beda4e585b73b860ff4d4404b30c6de9ec53d50d93e`
- source commit: `728ff94c26618523dd8f6d57b89a8e4b3932fc7a`
