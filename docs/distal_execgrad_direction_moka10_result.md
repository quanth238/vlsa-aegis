# ExecGrad controller-conditioned link-motion direction pilot

## Verdict

Independent H100 validation classifies the registered mechanism as
`NO_GO_direction_mechanism`. The link-JVP arm does not provide a useful local
Cartesian correction direction and must not be connected to flow guidance, a
QP, calibration, or closed-loop E05.

This rejects the tested combination of deterministic resolved-rate nominal,
direct residual trajectory decoder, and normalized link-center JVP loss. It
does not reject the broader idea that a different execution-gradient model may
learn a useful controller-conditioned direction.

## Registered comparison

Job `38744` evaluated the existing 85 states and 7,905 actions on one H100. It
used complete two-action chunks and paired translation/rotation directions;
gripper remained an input but was not treated as a steering dimension. The two
trained arms had the same direct nonrecursive residual architecture, seed,
data, optimizer, joint loss, symmetric safety-normal loss, and checkpoint
rule. Only `trajectory_plus_link_JVP` received the paired L5--L7 link-center
JVP loss.

The apparatus passed before training. The deterministic nominal repeated with
zero joint difference. On validation states, the MuJoCo point-Jacobian center
response reproduced the paired OSC link-center response with mean cosine
`0.999991` and scalar RMSE `0.012529 mm/action`, far inside the registered
`0.95` and `1 mm/action` limits.

## Validation direction result

| Model | Mean cosine | Wrong direction | Median exact gain | Mean random-rank p | Median gradient-norm ratio |
|---|---:|---:|---:|---:|---:|
| Trajectory + link JVP | **-0.366** | **9/10** | **-7.192 mm/unit action** | **0.849** | 0.185 |
| Trajectory only | 0.044 | 5/10 | -0.924 mm/unit action | 0.470 | 5.629 |
| Kinematic nominal | -0.017 | 5/10 | -1.353 mm/unit action | 0.526 | 5.962 |
| Frozen normalized joint-secant model | 0.625 | 1/10 | 11.647 mm/unit action | 0.076 | 0.030 |

The registered link-JVP gates were mean cosine at least `0.8`, wrong-direction
rate at most `5%`, mean random-rank p at most `0.05`, median gain at least
`0.1 mm/unit action` above trajectory-only, and validation joint RMSE no worse
than `1.1x` trajectory-only. All five decision tests failed.

The link-JVP arm also increased validation joint RMSE from `63.922 mrad` to
`287.309 mrad` and terminal RMSE from `87.084 mrad` to `289.898 mrad`. Its
absolute safety values remain diagnostic at this direction gate: it produced
95 validation false-safes, 88.26% exact-safe recall, and predicted-safe support
in 8 of 10 validation states.

## Root localization

The immediate failure occurs during fitting, not only on unseen states. On
training states, the link-JVP arm has mean direction cosine `-0.028`, wrong
direction on 27 of 59 nonzero-gradient states, and near-zero mean exact gain.
The physical center-JVP apparatus passed almost exactly, but the learned model
did not reproduce that response and its joint trajectory fit collapsed.

Therefore this experiment cannot yet test whether a correctly learned
link-motion JVP transfers to safety guidance. The first root-cause target is
the loss/optimization path—especially the train-RMS weights across action
coordinates and horizons and the checkpoint tradeoff—not E05 coverage or QP
linearization. A frozen, no-training decomposition should precede any revised
training. The older joint-secant model's positive direction is useful evidence,
but it still fails the preregistered cosine, wrong-direction, random-rank, and
magnitude gates and is not cleared for control.

## Scope and artifacts

- Scientific H100 job: `38744`, `worker-1`, `02:30:16`.
- Independent H100 validation: job `38764`, `worker-1`, six tests passed.
- Result: `/mnt/data/quanth/experiments/vlsa-distal-execgrad-direction/execgrad-direction-20260812c/result.json`
- Validation: `/mnt/data/quanth/experiments/vlsa-distal-execgrad-direction-validation/execgrad-direction-20260812c-validation-a/validation.json`
- Result SHA-256: `14af299f02932f14d57f9597970efebdaf189ba3d1fa2b109c0a70472b2d42b9`.
- Result payload SHA-256: `ab89434c6017320a0020cc0233e6dda837be72c93f453aa07dc352d0f7396e61`.
- Validation SHA-256: `ce78e6b7951c6d532b8a201816cb86c421ad267b3dbf3e4c0748b5b21e15cdc9`.
- Validation payload SHA-256: `1ff35ce9f80e05bc38e5221606e7224080c81d0a4dd311b3988f7a2bc9a0364a`.

No new episode was opened. No fresh cloned-OSC direction rollout, flow
guidance, calibration, QP, closed loop, Poisson/SDF, or classifier ran. Table 1
artifacts were not modified.
