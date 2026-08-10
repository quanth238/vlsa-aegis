# State-conditioned region-aware MLP result

## Verdict

**Strict NO-GO for the registered learned model.** Clean H100 job `37722`
completed normally on `worker-2`, and the independent validator accepted the
artifact. The model was conservative on the 15 unseen states, but vacuous: it
accepted no action and therefore supplied no QP proposal. The conditional E05
closed-loop arm was not authorized and did not run.

This rejects the current 62-feature, five-member MLP and the current 30-state
training population. It does not reject the validated multi-region oracle or
prove that every state-conditioned representation must fail.

## Frozen experiment

The run used commit `deef507f0c2cf66c174b53f051a280d7b76ea665` and config
SHA-256 `976c8305b0b5d1e28803fa48ee936252c00a98c096ead0ae52e77d3239adc867`.
Complete groups remained separated: 30 training states, five validation-only
states, and 15 unseen E05/E10/E15 test states. The network predicted seven
lower affine constraints for each of 27 regions. Exact cloned-OSC labels were
evaluation authority; no test label entered training, early stopping,
calibration, or selection.

## Registered gates

| Test quantity | Required | Observed |
|---|---:|---:|
| Unseen off-grid actions | 1,440 | 1,440 |
| False-safe actions | 0 | 0 |
| Global accepted-set Jaccard | >= 0.90 | 0.00 |
| Minimum state Jaccard | >= 0.80 | 0.00 |
| States retaining safe support | 15/15 | 0/15 |
| Valid learned regional QPs | 15/15 | 0/15 |
| Fresh exact-safe QP rollouts | 15/15 | 0/15 |
| Released-AEGIS-EE-compatible QPs | 15/15 | 0/15 |

The exact multi-region oracle accepted 1,235 of the 1,440 actions, and exact
simulation classified 1,236 as safe. Thus, the zero learned support is not
caused by an empty test safe set.

## Failure audit

The QP solver did not fail. The preliminary neural half-spaces rejected every
candidate, so no learned QP was formed. A read-only post-outcome audit of the
frozen predictions also removed calibration and ensemble guards: the raw
ensemble-mean planes still accepted zero actions in all 15 states. Therefore,
the 1 mm padding and uncertainty guard are not the primary cause.

Before those guards, predicted regional lower anchors had `0.144200 m` RMSE,
`0.131562 m` MAE, and `-0.131545 m` mean error relative to the unseen oracle
anchors. The ensemble guard correctly reflected disagreement but was large:
minimum/median/maximum were `19.374/137.476/286.851 mm`. Validation calibration
was much smaller, with mean/maximum `1.064/5.375 mm`.

The grouped states are strongly out of the training feature distribution. The
maximum absolute standardized feature was `3.158` in training, `4.082` in
validation, and `10.182` in test; `4.73%` of all test feature scalars exceeded
five training standard deviations. The median test-row maximum was `8.139`.
The largest shifts included joint velocity 3 (`10.182`), nominal first-action
x (`9.342`), nominal second-action x (`9.010`), joint velocity 4 (`8.862`),
and joint velocity 2 (`8.258`). Member early stopping at epochs 13--88 with
best validation objectives ranging from `8.359` to `57.879` is consistent with
the same group shift.

## Interpretation

The validated multi-region oracle remains feasible and decision-stable when
its local rows are measured at the current state. The current MLP cannot infer
those rows on unseen robot/task states from the available grouped population.
Consequently, this run provides no evidence that learned receding QP replanning
prevents E05 collision or preserves SafeLIBERO completion.

A later experiment should first repair state coverage or representation and
repeat the same unseen accepted-set gate. A useful minimum prerequisite is to
keep test inputs inside a registered training-support envelope without leaking
test episodes. Absolute nominal actions and weakly covered joint velocities
should not be trusted as extrapolation coordinates. Closed-loop E05 must remain
blocked until the learned rows retain safe support and pass fresh exact QP
rollouts.

## Provenance

- Slurm: `37722`, one H100, eight CPUs, 64 GB, `32 s`.
- Result file SHA-256:
  `aa8dc377aeca50acdb102e721d3e483951aa9fd4ecd6477f2b8e46b2d3d0904a`.
- Result payload SHA-256:
  `298c11f7eb8f0533fdd27fb4f86d02134af3de286d2dde63fdd2dfe61a26d6f3`.
- Validation file SHA-256:
  `692a81ff96262d0a4d8c868ab914b39fe85e1390da8f967191eb5beee703c78f`.
- Model SHA-256:
  `4fd9425d8f4f404570603cf215ff6ecbb1aeb5d58a13729608f9c70ac253e7f0`.
- Local artifact root:
  `output/vlsa_distal_region_aware_mlp/region-aware-mlp-20260810a`.
