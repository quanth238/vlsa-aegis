# E05 Moka controller-conditioned response-field result

## Verdict

Strict **NO-GO** for the current one-task MLP. The frozen model did not
reproduce the exact local Moka avoidance response on either untouched state,
so no corrected action was executed and no QP or closed-loop run is
authorized.

## Immutable experiment

- Producer: Slurm `39230`, H100 `worker-1`, commit
  `a505434ea7aed7e55f596e0b61d389e5ac593ed1`.
- Independent validator: Slurm `39238`, same clean commit.
- Runtime: `1479.725 s`; cloned-OSC rollouts: `987`.
- State groups: actions `178--182` train, `183--184` validation, and
  `185--186` untouched test.
- Moka geometry: 15 compiled collision boxes represented independently;
  the released single obstacle MVEE and E38 are excluded.
- Result file SHA-256:
  `1a8f78a486812fd2cc22fe5e75a40198d13c074be1024abfcae160c36e09ed93`.
- Validation file SHA-256:
  `f7858a73123eb1f9215c75d698ec5172965113bdc228552bbba86e7e762c99e8`.

## Held-out prediction

| Test action | learned/exact direction cosine | paired-response cosine | sign accuracy |
|---:|---:|---:|---:|
| 185 | 0.677 | 0.392 | 0.679 |
| 186 | 0.313 | 0.057 | 0.518 |

The registered requirements were direction cosine at least `0.8` and sign
accuracy at least `0.75`. Neither test state passed.

## Exact correction response

All exact gains below are relative to the same `-58.272 mm` conservative
multi-primitive nominal minimum. Positive gain means improved clearance, not
collision prevention.

| Test action | radius | learned gain (mm) | fixed gain (mm) | learned - fixed (mm) | random p | gate |
|---:|---:|---:|---:|---:|---:|:---:|
| 185 | 0.10 | 0.346 | 0.606 | -0.260 | 0.0769 | fail |
| 185 | 0.25 | 1.302 | 2.031 | -0.730 | 0.0462 | fail |
| 185 | 0.50 | 2.561 | 4.088 | -1.527 | 0.0462 | fail |
| 186 | 0.10 | 0.290 | 0.309 | -0.020 | 0.0308 | fail |
| 186 | 0.25 | 0.945 | 0.721 | +0.224 | 0.0462 | row pass only |
| 186 | 0.50 | 2.059 | 1.631 | +0.429 | 0.0308 | fail |

Only the action-186/radius-0.25 row passed its per-radius criteria. The
state-level gate still failed because the learned direction and held-out
responses were inaccurate. Every learned candidate remained proxy-unsafe and
retained protected contacts; the largest learned gain was only `2.561 mm`
against a roughly `58 mm` violation under this conservative representation.

## Interpretation

This result rejects the present state/witness-conditioned MLP trained from
five nearby E05 states. It does not reject paired controller rollouts or the
exact multi-witness oracle: the experiment shows that the local physical
response can be measured, but the current model does not transfer it reliably
across nearby warning states. More correction magnitude, a QP, or closed-loop
execution cannot repair an incorrect learned response direction.

Before changing the architecture, the next diagnostic should separate:

1. response fit on train and validation states;
2. state-support distance for actions 185--186;
3. link/time/primitive witnesses responsible for the large nominal negative
   margin; and
4. whether fitting only near-active witnesses improves held-out response
   accuracy without reintroducing the single-worst-witness failure.

This remains one-task mechanism evidence, not cross-episode generalization,
task completion, a CBF, or formal safety.
