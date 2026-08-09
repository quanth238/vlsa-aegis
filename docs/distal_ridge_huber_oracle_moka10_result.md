# Ridge-Huber affine target oracle result

## Verdict

Ridge-Huber fixes coefficient instability and improves safe-action recall, but
it does not pass the complete oracle gate. Therefore no state-conditioned MLP
was trained and closed-loop E05 remains unauthorized.

## Experiment

H100 Slurm job `37688` ran on `worker-1` from clean commit
`5576cc02c7072a6407eb48cb154c4255dbd9e62c`. It completed in `00:04:55`.
The job refit all seven rows at all 50 immutable states, used the same 16
deterministic 80% resamples for ridge-Huber and minimum-L1, evaluated both on
the untouched 4,800 off-grid exact actions from job `37649`, and executed 100
new cloned-OSC QP rollouts.

No training or closed-loop command was part of this job. Table 1 and all
source datasets remained read-only.

## Aggregate comparison

| Target | False-safe | Safe recall | States accepting safe action | Fresh safe QPs | Active-row stability |
|---|---:|---:|---:|---:|---:|
| Ridge-Huber | 0/4,800 | 95.76% (3,795/3,963) | 48/50 | 49/50 | 41/41 stable |
| Minimum-L1 | 0/4,800 | 87.86% (3,482/3,963) | 48/50 | 49/50 | 28/41 stable |

Ridge-Huber therefore solves the target-instability symptom: no active row
failed resampling, whereas 13 minimum-L1 active rows across 11 states were
unstable. It also retained 313 more exact-safe off-grid actions.

## Failed strict states

Both targets rejected every safe off-grid action at two sparse states:

- `vlsa-t1-spatial-i-t1-e00`, step 120: 7/96 actions were exact-safe;
- `vlsa-t1-goal-ii-t2-e00`, step 30: 1/96 action was exact-safe.

At goal-II-t2-e00 step 30, ridge-Huber's seven-row QP was primal infeasible.
The critical L5-part-2 nominal margin was `-3.638 mm`; the only exact-safe
off-grid action reached `+0.040 mm`, but the calibrated ridge-Huber lower
plane predicted `-1.638 mm`. Thus the stable global lower plane did not expose
the narrow safe component.

The minimum-L1 QP was safe at this state, but its comparator failed separately
at spatial-I-t1-e00 step 120 when OSQP reached 10,000 iterations. This numeric
stall does not change the ridge-Huber decision.

## Interpretation

The result identifies two distinct issues:

1. The minimum-L1 coefficient selection is genuinely unstable; ridge-Huber
   fixes that problem.
2. A single conservative affine lower plane over the complete registered box
   can still erase a small safe component. Stable targets alone are therefore
   insufficient for the required per-state QP gate.

The correct next oracle is piecewise/local: fit multiple affine regions or a
region conditioned around exact-safe support, then retain the same zero-
false-safe, per-state support, resampling, and fresh-QP tests. Merely training
the MLP on the current ridge-Huber coefficients would learn a stable but
known-incomplete safe set.

## Decision

- `ridge_huber_oracle_gate_pass=false`
- `learned_training_authorized=false`
- `learned_training_executed=false`
- `closed_loop_e05_authorized=false`

## Artifact identity

- Result SHA-256: `6efda2cb521416f9086dbd8ea99a07f46e32e9bb77b84cefe2efc0ea0ec1e8b3`
- Result payload SHA-256: `1076ef752d3b1ef0dda306a654c992292006c6fa6edb53d7e1f965138a91f2a3`
- Validation SHA-256: `c2d831aa87edae00918c4f8b020d6f1a53e3579994b51d0bf1ebaad39a826c4d`
- Remote root: `/mnt/data/quanth/experiments/vlsa-distal-ridge-huber-oracle/ridge-huber-oracle-20260810a`
- Local copies: `output/vlsa_distal_ridge_huber_oracle/ridge-huber-oracle-20260810a`
