# R04B allocation-backed exact-resume evidence

R04B is the final one-case apparatus smoke. It proves that the opt-in saved
latent resume path can continue the exact deterministic sampler trajectory and
that the ordinary compiled baseline remains unchanged on the reviewed commit.
It is not a prediction, gradient-causality, steering, efficacy, novelty, or
transport result.

The raw 12.16 MB artifact remains on allocation-visible storage:

```text
/mnt/data/quanth/experiments/crfs-oracle/
  r04b-resume-parity-smoke-20260714a/
  crfs-93365b8b851365f2/r04-resume-parity.json
```

Its SHA-256 is
`977084df18cdccf1a8cdca42a1af1faaa4ef039ce2d83df716f14b8b43e3dae1`.
The compact checked-in record is `r04b-validation.json`.

- Source Slurm array task: `27558_0`, completed `0:0` on
  `worker-mig-3g40gb-0`.
- Dependency-backed OpenPI/PyTorch contract tests: 6/6 passed inside the source
  allocation before policy-server startup.
- Independent validator: Slurm job `27559`, completed `0:0` on `worker-2`,
  bound to source identifiers `27558/27558/0`, with zero errors.
- Reviewed source commit:
  `15b97b63f77b1b8b60fa53b5efec334ff2d32a3e`, clean.
- Frozen config SHA-256:
  `a0edb7edac86d2abd887218002d14e28e072cd472d631797d18d540f5140be33`.
- All zero resumes at steps 1--5 matched source feature fields and final
  normalized/physical actions by dtype, shape, and contiguous bytes.
- The two ordinary compiled calls were byte-identical and passed the unchanged
  compiled/eager physical limits.
- The one reset and 20 dummy settle controls produced the observation; no
  sampled-policy action or efficacy rollout was executed.

The reused case is permanently restricted to
`apparatus_only_never_train_calibrate_validate_test_or_claim`.
