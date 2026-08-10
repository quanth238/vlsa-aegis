# One-sided geometry supervision v2: validated result

## Verdict

Strict NO-GO for deployment, with strong positive mechanism evidence.

H100 job `38070` trained the validation-adapted, link-generic one-sided
geometry-loss model and independently recomputed every saved joint prediction
and FK/ellipsoid margin exactly.

| Untouched E05/E10/E15 metric | Flat factorized baseline | One-sided geometry |
|---|---:|---:|
| False-safe actions | 44 | **0** |
| Exact-safe action recall | 94.70% | **93.84%** |
| States with accepted exact-safe support | 13/15 | **12/15** |
| Near-boundary RMSE | 2.188 mm | 2.671 mm |
| Joint-sensitivity cosine | 0.975 | 0.977 |
| Safety-margin sensitivity cosine | 0.913 | 0.921 |
| Joint-trajectory RMSE | 11.560 mrad | 10.864 mrad |

The asymmetric signed loss did what it was designed to do: it eliminated all
observed dangerous clearance overestimation on the untouched test actions
while preserving useful gradients and most safe actions. It nevertheless
failed the all-state support requirement. Three of fifteen held-out states
have no accepted exact-safe candidate, so a QP cannot act there.

Validation is also unstable: only 3/10 validation states retain safe support,
with one false-safe and 35.05% recall. This is evidence against the current
flat 51-by-7 output decoder, not evidence that another scalar buffer should be
added. A buffer can only reject more actions and cannot restore the three
unsupported test states.

## Interpretation

The broader factorized direction remains promising:

- the local signed geometry approximation is accurate on validation
  (`0.201 mm` RMSE, cosine `0.974`);
- joint and action-space safety sensitivities remain strong;
- false-safes fell from 44 to zero without a binary classifier; and
- exact-safe action recall remained above 93%.

The rejected method is now more specific:

\[
\boxed{\text{complete inputs + flat 357-output MLP + joint/sensitivity/
one-sided geometry loss}}
\]

It is not reliable across state groups. The next matched model should share a
time-conditioned execution decoder,

\[
\hat q_k=F_\theta(x,A,k),
\]

while retaining joint, sensitivity, and one-sided geometry supervision. That
tests whether explicit temporal parameter sharing prevents terminal and
episode-dependent error growth. Do not add a classifier or global buffer.

Calibration, QP, and closed-loop E05 remain blocked.

## Reproducibility

- Job: `38070`, `COMPLETED` in `00:38:38` on `worker-2`.
- Source commit: `5597dca239e562126e33465375fea942de150787`.
- Result SHA-256:
  `ee0f1296cdd155564a9cd040ae1cceca62a1005e1126b50b383614d6f1f12de0`.
- Result payload SHA-256:
  `e8763d699b7a20c3656a5b2587dad63ee76a95fa577d20bdb49bdc89d4f0b730`.
- Validation SHA-256:
  `1abb16135207cfc754c6d1a63373a01be69ed4930bdb1e69036ba3e4f3c1a3b3`.
- Validation payload SHA-256:
  `7dbb6ee1b6f080f99dbf6b6c882cc5c440554f185854ea22608d5de8fa44d334`.
- Model SHA-256:
  `5548b8e4b48fcf11aacbb4f26186e59f487c15ccaab8eb0882e0a9fcdf2f1a62`.
- Predictions SHA-256:
  `acf25b0467ce26669d9a0f57e0c192639bbad76692d09aaac81bc10666023396`.
