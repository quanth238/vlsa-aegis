# Fixed obstacle-orientation joint-mask result

## Verdict

H100 job `38093` and its independent replay pass every registered gate. The
validation explosion is caused by duplicated, variance-floored scalar obstacle
rotation inputs.

This confirms the root of the physically impossible validation trajectories.
It does not repair the remaining normal-scale safety error or make either
execution model suitable for a QP.

## Causal result

The fixed union masks 288 scalar entries: obstacle-root rotation and the same
rotation repeated in every exact obstacle primitive. On the exploding
goal-II-t3-e35 validation episode:

| Model | Original terminal RMSE | Joint-mask terminal RMSE | Reduction |
|---|---:|---:|---:|
| Flat one-sided | 1896.791 rad | 0.017637 rad | 99.99907% |
| Time-conditioned | 1630.247 rad | 0.011109 rad | 99.99932% |

All six gates pass. The ordinary validation episode remains at
`0.042855/0.035057 rad`, and test terminal RMSE changes by only `-0.0375%` for
the flat model and `+0.0113%` for the time model. Masking every variance-floored
feature gives nearly the same result as masking only the duplicated rotations,
which further isolates the cause.

The physical setup is valid: E35 rotates the obstacle by 180 degrees. The
failure is the representation. Training sees individual rotation-matrix
entries essentially constant at `+1` or `-1`; scalar standardization with a
`1e-6` denominator maps a sign flip to about `2e6`. Repeating that representation
across 15 primitives amplifies it at the first layer.

## Consequence

Do not simply delete obstacle orientation or count this masked forward pass as
a corrected model. The next matched experiment should retain orientation using
a structured bounded representation—for example one root quaternion or 6D
rotation plus fixed local primitive transforms—leave presence bits
unstandardized, and avoid per-scalar division by near-zero variance. Compare it
against the original representation with the same data, model, loss, seeds,
and splits.

No simulation, training, changed normalization, calibration, candidate search,
QP, Poisson/SDF, or closed loop ran in this confirmation.

## Provenance

- Slurm allocation: `38093`, `worker-2`, one NVIDIA H100
- elapsed time: `00:00:11`
- source commit: `40059d5dd98e162eec2319745b1e27362c78fa62`
- result SHA-256: `1cd507f5b00ad4a5a243f957ae3466348680e821a9fab973caf133f73023cfc1`
- result payload SHA-256: `b7e34029375acf8be920b4ca67c6776a9f749a14b5760472df8da05168ebda6a`
- validation SHA-256: `b5b3931bc85021c849a4916e5e260b90c389b7a267931e9b9d8f0adb1ebfaa66`
- validation payload SHA-256: `21210c2f37faef35f10812c6e407b986cf0ce8993156454bda80cd71603f7020`
