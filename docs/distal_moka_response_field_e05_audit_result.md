# Frozen E05 Moka response-model audit result

## Verdict

The primary failure is **training-response underfit caused by the current
input/target/loss representation**. Held-out state coverage is also poor,
especially at action 186, but it cannot explain a model that already fails on
its own training responses. Near-active primitive switching is not the cause.

No model was trained or changed. No correction, QP, or closed-loop action was
attempted.

## Evidence

Clean H100 producer `39265` replayed 585 deterministic cloned-OSC rollouts
from commit `f6355f490c5b5780e1bdc6ed9dcc0edeec85a993`. Independent H100
validator `39272` accepted the artifact.

| Split | MLP held-out cosine | MLP sign accuracy | local ridge cosine | local ridge sign accuracy | value RMSE |
|---|---:|---:|---:|---:|---:|
| Train | 0.312 | 0.573 | 0.663 | 0.877 | 34.29 mm |
| Validation | 0.340 | 0.627 | 0.427 | 0.808 | 33.27 mm |
| Test | 0.098 | 0.599 | 0.379 | 0.779 | 36.29 mm |

The MLP therefore fails before unseen-state evaluation. Its fit-direction
cosine on training equations is only `0.521`, and its mean fitted row-gradient
cosine is `0.319`. The selected checkpoint was epoch 8. The train value model
also produced 21 false-safe action-boundary witnesses.

## Representation diagnosis

The model receives 1,055 inputs but only five distinct physical training
states. Of those inputs, 875 dimensions are constant across training and are
assigned the unit fallback scale. The input contains complete simulator and
auxiliary internals, the absolute clock, controller state twice, the complete
20-action chunk, constant absolute obstacle geometry, and witness identity.

This is a high-dimensional, redundant representation with very little state
support. Validation chunks have 27.9--32.9% of action coordinates outside the
train range; test chunks have 31.4--33.6%. Test action-chunk nearest-training
distance rises to `3.617/4.346` RMS z at actions 185/186. Action 186 also has
an auxiliary-state distance of `19.198` RMS z. Obstacle geometry is fully
supported and constant, so obstacle-pose coverage is not causal here.

## Near-active witness diagnosis

All 54 rows within 2 mm of the state-wise worst value are the same
`robot0_link5` slab against compiled Moka primitive
`moka_pot_obstacle_1_g4`, over absolute actions 188--193. The closest
primitive never switches in any paired rollout: switch fraction is zero. The
second-closest primitive remains 5.15--6.65 mm farther away.

These critical rows are only 30 of 700 training witnesses (`4.29%`). The MLP
overestimates their safety by `48.03 mm` on average, and their mean
response-row cosine is only `0.509`. On training states, exact near-active
gradient norm is `15.49 mm/action`, while the model predicts
`6.26 mm/action`. Equal weighting over all 140 witnesses therefore permits
severe error exactly where the smooth field obtains most of its weight.

The local ridge teacher is better than the MLP in sign but not an exact
full-vector gradient: held-out cosine falls from `0.663` on train to
`0.427/0.379` on validation/test despite zero critical primitive switches.
This indicates finite-radius, long-horizon OSC nonlinearity and magnitude
error in addition to MLP underfit. Merely training the same model longer is
not justified.

## Root cause and next decisive test

The causal ordering is:

1. **Primary:** the 1,055D plain MLP and mixed unweighted value/secant loss do
   not fit the training safety response.
2. **Secondary:** shifted action chunks, controller state, and action-186
   auxiliary state are outside the five-state training support.
3. **Not causal:** Moka perception/pose in this run, near-active primitive
   switching, correction magnitude, or the QP.

The smallest next experiment should reuse the paired data and compare the
current representation with a compact, nonduplicated physical state:
`q`, `qdot`, OSC goal/error, the relevant Cartesian chunk, and relative
link--obstacle features. First require one-state memorization and aggregate
training response cosine above 0.8. Then use scale-normalized,
near-active-weighted paired response supervision and test validation. In a
separate matched arm, reduce the secant radius or condition the scalar response
on direction to determine whether the remaining ridge magnitude error is
local nonlinearity. No QP or closed-loop experiment is warranted until this
training-fit gate passes.

## Provenance

- Result file SHA-256:
  `19b754da4c26a9950038463ca1f49e5543f42042ea259764885a84e269450e27`.
- Result payload SHA-256:
  `5e0e80b6163c90e6c28a8b76ef61dd8dcfb112f95c9439322d5d62aff78e178d`.
- Validation file SHA-256:
  `803a7e09252a114bedfdc9b5a3a8b03d2cacb7db3034dff85f2f86eef932df50`.
- Validation payload SHA-256:
  `ca73b28abcc82b87c744100cc8a53d730bb795d3508d9850b904af24f3fe59c7`.

Attempt `39244` stopped before simulator startup on a missing isolated
`PYTHONPATH` and produced no scientific result. Earlier audit `39247/39261`
established the same underfit diagnosis but lacked the final local-teacher and
primitive-switch checks; `39265/39272` supersede it.
