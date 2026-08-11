# Recurrent nominal-plus-residual OSC execution gate

## Verdict

Independently validated H100 job `38702` is a **strict fitted-model NO-GO**.
The recurrent execution model improves trajectory and boundary accuracy over
the frozen direct-horizon model, and it recovers part of the missing action
response magnitude. It still does not learn a sufficiently correct or strong
action-to-joint response on its own training episodes. It therefore cannot be
used for VLA flow guidance, calibration, a QP, or closed-loop control.

The result stops before the five untouched task-3 episodes and before the
matched-random correction test, exactly as preregistered.

## Matched experiment

Both arms use the same immutable `7,905` two-action candidates, grouped
train/validation/old-diagnostic splits, complete OSC inputs, five ensemble
seeds, normalized paired-action secant supervision, symmetric safety-normal
supervision, and L5--L7 ellipsoid geometry.

- Frozen comparison: job-`38586` direct-horizon joint-displacement model.
- Experimental arm: causal GRU nominal-plus-residual model.
- Every recurrent output is a direct displacement from measured `q0`; no
  predicted increment is cumulatively integrated.
- Candidate residuals are centered by subtracting the zero-action-delta trace.
- The second action is architecturally unable to affect substeps `1--25`.

Checkpoint selection used validation secant fidelity among joint-error-gate
qualified checkpoints. This is a feasibility comparison, not a
recurrence-only causal ablation.

## Validation-set comparison

| Metric | Required | Direct job 38586 | Recurrent job 38702 | Recurrent pass |
|---|---:|---:|---:|:---:|
| Joint RMSE | <= 21.202 mrad | 19.275 | **15.987** | yes |
| Terminal joint RMSE | <= 35.068 mrad | 31.880 | **27.321** | yes |
| Horizon RMSE slope | <= 0.500 mrad/substep | 0.390 | 0.436 | yes |
| Near-boundary margin RMSE | <= 2.671 mm | 6.414 | **5.735** | no |
| False-safe actions | 0 | 95 | **62** | no |
| Exact-safe recall | >= 90% | 100.00% | 93.76% | yes |
| Recoverable-state support | 9/9 | 9/9 | **8/9** | no |
| Aggregate joint cosine | > 0.8 | 0.658 | 0.602 | no |
| Aggregate joint median gain ratio | 0.5--1.5 | 0.049 | **0.271** | no |
| Terminal joint cosine | > 0.8 | 0.838 | 0.821 | yes |
| Terminal joint median gain ratio | 0.5--1.5 | 0.018 | **0.094** | no |

For the common row-weighted safety-sensitivity metric, recurrence changes
validation aggregate cosine from `0.593` to `0.608` and median gain ratio from
`0.025` to `0.157`. These are improvements, but both remain outside the
registered useful-magnitude gate. The decision-level validation summaries are
also failing: aggregate joint/safety cosine `0.602/0.506`, median gain ratio
`0.271/0.492`, and mean relative norm error `0.721/0.560`. At the terminal
substep they are joint/safety cosine `0.821/0.605` with gain ratio
`0.094/0.099`.

## Root-cause localization

The failure is already present on training episodes:

- aggregate joint/safety cosine: `0.586/0.582`;
- aggregate joint/safety median gain ratio: `0.251/0.351`;
- terminal joint/safety cosine: `0.783/0.604`;
- terminal joint/safety median gain ratio: `0.050/0.064`;
- training false-safe actions: `214`;
- recoverable-state support: `50/52`.

Therefore this fitted result is not primarily an unseen-state coverage
failure. The recurrent representation and frozen training/checkpoint process
still underfit the physical action response. Recurrence partially repairs the
vanishing-gain behavior of the flat model, but it does not provide a reliable
correction direction.

The experiment does not distinguish architecture capacity, objective
conditioning, and optimization as separate causes. A per-member recurrent
audit would be needed before attributing the ensemble failure specifically to
member optimization or cancellation.

## Gate decision

Passed structural and trajectory checks:

- exact `q0`;
- validation joint and terminal RMSE;
- horizon slope and terminal/overall ratio;
- validation safe recall.

Failed scientific checks:

- train and validation aggregate joint/safety direction and magnitude;
- train and validation terminal magnitude and most terminal direction checks;
- zero validation false-safes;
- near-boundary RMSE;
- support in every recoverable validation state.

Consequently:

- untouched task-3 episodes opened: `false`;
- matched-random rollouts executed: `false`;
- flow guidance executed: `false`;
- calibration/QP/closed loop executed: `false`.

This is the preregistered stopping condition. The proposed full VLA-guidance
experiment is not authorized with either current learned execution model.

## Reproducibility

- Slurm job: `38702`, `worker-2`, NVIDIA H100 80GB HBM3.
- Allocation elapsed time: `02:08:09`, exit `0:0`.
- Clean source commit: `19d9b8b79a841e46333d9953ef2d751e2a30bff4`.
- Run root:
  `/mnt/data/quanth/experiments/vlsa-distal-recurrent-nominal-residual/recurrent-nominal-residual-20260812b`.
- Model SHA-256:
  `507176ad5467e1cffb8cc227f3cb152d8c657a563c00df804b0d2b21efa827d5`.
- Predictions SHA-256:
  `b9a2594d76d0355c5cad89381437c55a348d054f746bb68e2614a73ed74ce2a4`.
- Result file SHA-256:
  `b453277b69282388062353d367f398999311c68e39991ab87b94cf7e7a876b04`.
- Validation file SHA-256:
  `0fe6c7ab957df4f3c2e6fc54fa7fb40baea5ea8df565d3ce2db383de379209d9`.
- Validation status: `complete`, `valid=true`.
- The validator reloaded all five members, recomputed all `7,905` actions,
  reproduced five stored arrays with zero finite difference, and reproduced
  geometry, support, metrics, and the complete hard decision exactly.

Completed Table 1 artifacts were neither modified nor reinterpreted.
