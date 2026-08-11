# Explicit Local-Affine Execution-Jacobian Result

## Verdict

H100 job `38673` is an independently validated **strict NO-GO** for the
current explicit-J decoder and matched training protocol.

The architecture executed as intended:

- candidate secants match the stored explicit Jacobian to
  `4.42e-15 rad/action` maximum error;
- \(J_0\) is exactly zero;
- action-2 Jacobian columns are exactly zero through substep 25;
- all eight stored arrays reproduce with identical NaN masks and zero finite
  difference after reloading the five-member model.

The fitted execution model nevertheless fails every scientific prediction
gate.

| Metric | Required | Explicit J | Matched job 38586 |
|---|---:|---:|---:|
| Train aggregate joint cosine | >= 0.8 | 0.126 | 0.691 |
| Validation aggregate joint cosine | >= 0.8 | 0.132 | 0.658 |
| Train aggregate median gain ratio | 0.5--1.5 | 1.348 | 0.046 |
| Validation aggregate median gain ratio | 0.5--1.5 | 1.455 | 0.049 |
| Train terminal joint cosine | >= 0.8 | 0.070 | 0.902 |
| Validation terminal joint cosine | >= 0.8 | 0.060 | 0.838 |
| Train terminal median gain ratio | 0.5--1.5 | 0.781 | 0.017 |
| Validation terminal median gain ratio | 0.5--1.5 | 0.846 | 0.018 |
| Validation joint RMSE | <= 21.629 mrad | 24.787 mrad | 19.275 mrad |

Thus, explicit decoding repaired the gross gain collapse but not the learned
action direction. Translation sensitivity is especially wrong: train and
validation mean cosine are `-0.046/-0.041`; rotation reaches only
`0.298/0.305`. The resulting explicit model also has 166 train and 23
validation proxy false-safes. Those safety counts are diagnostic only; the
experiment was already stopped by the fitted sensitivity and joint-trajectory
gates.

## Interpretation

The local-affine construction is internally correct, but the frozen
multi-objective optimization does not learn a useful intercept and slope
together. This result rejects the claim that changing only the decoder to emit
an explicit Jacobian solves the job-38586 failure. It does not reject a learned
OSC execution model in general.

The matched comparison separates two failure modes:

- job `38586` retained useful direction but almost zero gain;
- job `38673` recovered gain magnitude but learned nearly random or reversed
  directions and worsened trajectory fit.

A calibrated optimism bound cannot repair either wrong action directions or
the failed trajectory gate. Therefore no residual-bound calibration, QP,
closed-loop control, new unseen-episode evaluation, Poisson/SDF model, binary
classifier, or new rollout collection was run.

The smallest next diagnostic is a no-training per-member audit of the frozen
explicit-J weights. It should determine whether direction is already wrong in
each member or is lost by ensemble averaging, and compare member direction to
checkpoint epoch. If all members fail, the next intervention must separate or
rescale intercept and Jacobian optimization/checkpoint selection; it should
not proceed to certification or control.

## Immutable evidence

- Slurm job: `38673`, worker-1 H100, elapsed `01:35:27`, exit `0:0`.
- Source commit: `bbe1cf7ffcb1385885d3bf2d3bfd89726aeae901`.
- Model SHA-256:
  `9d86649d74617316ce9fce7080d7227c970426e46c104d847b99f477fc3d7393`.
- Predictions SHA-256:
  `aa93567f28dde0fddf075e26e6982b68ed31fcb3591d41bbc167d546b3840d7c`.
- Result SHA-256:
  `944ea7e1aa02030b9be39002af0d73291e85d396c25fc836e6cbc7afff5b78cf`.
- Result payload SHA-256:
  `abc84fd53db6a031bc01bd8d1bee8cd1e01e3f484bdff4da304a53233a3350ae`.
- Validation SHA-256:
  `9e9aeaeb4825d13267885bd26104dca2463255d74f3c669ba294e433430795fe`.
- Validation payload SHA-256:
  `e4b53c288e57732d2b2a9d1080e4d58a228278dd0ddf9c01f8eb404883f7aeb6`.
- Preflight SHA-256:
  `5203edb9b97830c1884ab9dd92a932a7b20cd671c8e58ada0900e6309aa6348e`.

Exact next read-only command:

```bash
jq '{decision,explicit_jacobian_consistency,training:{member_audits:.training.member_audits},train:.metrics.sensitivity.train.all_horizon_trace.joint,validation:.metrics.sensitivity.validation.all_horizon_trace.joint}' /mnt/data/quanth/experiments/vlsa-distal-explicit-execution-jacobian/explicit-execution-jacobian-20260811a/result.json
```
