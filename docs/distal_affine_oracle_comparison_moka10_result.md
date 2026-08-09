# Affine oracle comparison result

## Verdict

The canonical finite-difference affine constraint passes; the direct
label-fitted half-space fails.

This is a representation GO for one local affine row per L5--L7 constraint
inside the registered action boxes. It is not a learned-model, QP deployment,
closed-loop task, or formal safety result.

## Experiment

H100 Slurm job `37649` ran on `worker-2` from clean commit
`52974146582a8e346d8728229e409091bc69b36c`. It completed in `00:23:51` and
executed 5,100 new cloned two-action OSC rollouts: 300 exact central-
difference probes and 4,800 fresh evaluation actions. The comparison covered
all 50 immutable states from job `37580`; every state retained its immutable
second released-AEGIS action and every internal OSC/MuJoCo substep was
measured.

The safety label used for the decision was the conjunction of nonnegative
exact L5--L7 ellipsoid margins, zero raw protected contact, and no more than
0.1 mm obstacle motion. Optimizer-side affine values and simulator-side
verification remained separate. The released AEGIS EE proxy was diagnostic
only because this test isolates the seven distal rows.

## Results

| Oracle representation | False-safe / 4,800 | Safe recall | Safe-support state coverage | Gate |
|---|---:|---:|---:|---|
| Direct label-fitted half-space | 30 | 99.85% (3,957/3,963) | 98% (49/50) | NO-GO |
| Canonical finite difference + one-sided error | 0 | 93.34% (3,699/3,963) | 96% (48/50) | GO |

The finite-difference arm had zero false-safe actions in every split. It
accepted 1,169/1,236 test, 2,080/2,275 train, and 450/452 validation safe
actions. Its two uncovered states contained only 7 and 1 fresh safe actions;
it rejected all of them conservatively and did not accept an unsafe action.

The direct half-space had 12 test, 17 train, and 1 validation false-safe
actions. Its high recall is therefore not sufficient: moving the fitted
threshold beyond all unsafe grid scores did not prevent optimistic
interpolation between the 125 fitted actions.

## Interpretation

The learned job-`37630` failure is not evidence that one affine QP row is
intrinsically insufficient at this trust-region scale. A canonical local
derivative with state-specific one-sided tightening represented the fresh
safe set conservatively and retained useful support. The failed object was
the non-unique minimum-L1 coefficient target and, separately, naive direct
binary half-space fitting without held-out tightening.

The next learned experiment should not regress the old minimum-L1 coefficient
selection. It should output local affine values/coefficients while training
directly on action-level one-sided lower-bound and safe/unsafe half-space
losses, anchored by canonical finite-difference information, then add
held-out state-group uncertainty tightening. It must repeat the zero-false-
safe and useful-recall test before any QP or closed-loop E05 run.

OSQP was not exercised or audited here because it cannot repair an incorrect
safe-set representation. Closed-loop E05 remains unauthorized.

## Artifact identity

- Result SHA-256: `5f34efe306cde24e1d2b20801015cb5f353f7e598f8afd606124da3bd0ff175f`
- Result payload SHA-256: `661273a27aa5a025b160f50e7b0526bb6d434a4b90fa1a8e63758ab0f2083ff6`
- Validation SHA-256: `8c5aa5c5f56e179a4603584d2e45da3e1de2ca571b1dc512e315c8a05c010b4f`
- Immutable source dataset SHA-256: `dc8d0dd89cf5dd3f736eb9c33dde884eb64ad1206e686504b4e42385f3ab6dc8`
- Remote root: `/mnt/data/quanth/experiments/vlsa-distal-affine-oracle-comparison/affine-oracle-cmp-20260810a`
- Local copies: `output/vlsa_distal_affine_oracle_comparison/affine-oracle-cmp-20260810a`
