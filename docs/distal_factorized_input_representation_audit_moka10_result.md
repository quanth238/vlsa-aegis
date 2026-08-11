# Factorized complete-input representation audit result

## Verdict

H100 job `38092` and its independent replay are valid. The registered
single-group root-cause gate is inconclusive: no one semantic group removes at
least 90% of validation terminal-joint error from both frozen models. The
result must therefore remain recorded as
`model_internal_instability_without_strong_input_group_cause` under the frozen
decision rule.

The measurements nevertheless localize the pathology to duplicated obstacle
orientation inputs, not positional time or ordinary OSC state.

## Evidence

- Fresh train-only means and standard deviations exactly match both models.
- Immutable job-`38070` and job-`38076` predictions replay exactly.
- `1,754/2,124` input dimensions use the `1e-6` standard-deviation floor.
- Validation reaches `2,000,000` standard deviations, versus `7.68` in train
  and `11.28` in diagnostic test.
- Both models first explode in their first linear layer: validation maxima are
  `540,957` and `540,812`, versus train maxima `7.20` and `4.56`.
- Only validation episode `vlsa-t1-goal-ii-t3-e35` explodes: flat/time terminal
  RMSE is `1896.79/1630.25 rad`. The other validation episode remains at
  `0.0429/0.0351 rad`.

The strongest features are raw obstacle rotation-matrix entries. Their train
mean is approximately `+1` or `-1` with raw standard deviation about
`2.9e-14`; E35 contains the opposite sign. Flooring the denominator to `1e-6`
turns this valid 180-degree setup change into a `2e6` z-score. The same
orientation is repeated in the root obstacle transform and all exact obstacle
primitive transforms.

Masking only the repeated primitive rotations reduces flat/time terminal error
by `59.34%/73.47%`, short of the frozen 90% gate. Masking only root orientation
is also insufficient because the other encoding remains. A fixed joint-mask
confirmation is therefore required before declaring the representation
artifact causal.

No simulation, training, changed normalization, calibration, candidate search,
QP, Poisson/SDF, or closed loop ran.

## Provenance

- Slurm allocation: `38092`, `worker-2`, one NVIDIA H100
- elapsed time: `00:00:26`
- source commit: `71ece3e004a334e023ae54cf1f0f1ff142a1dc0e`
- records SHA-256: `60d4539ef50e4539b912ba632dca0dc5e861e09f0034a6addaac645bc10ed2f1`
- result SHA-256: `055508e5309ae73d9a6a538ce65b4ab563af838e78ca941827c595efe9ea2f1f`
- result payload SHA-256: `5891d2318432bcd74c34eca757db41646140b9e08aab76e0842a1f06a1529b54`
- validation SHA-256: `656e6be4e034de13f9fc7820060624d64f8c4ab8ca3ee48da0cc9ffd79adb9e9`
- validation payload SHA-256: `d5db29e0f3f3b2e542835fbfda7ef4bc5ad1133dd77940219d46056899e4ac20`
