# E05 Moka compact-input memorization ablation

The frozen response-model audit found a 1,055-dimensional input with only five
physical training states, 875 constant dimensions, duplicated controller
state, and poor training-response fit. This gate changes exactly one factor:
the model input.

Use only archived E05 action 182, the latest state in the original training
split. The physical context is the nominal first-five XYZ action chunk
(`15` values). The existing witness expansion adds three time features and a
seven-way robot-row identity, for exactly `25` model inputs. Raw simulator
internals, controller duplication, absolute clock, obstacle constants, action
tail, rotation, and gripper are excluded.

The architecture, AdamW optimizer, learning rate, weight decay, value loss,
paired-response loss, response weight, seed, epoch limit, and patience are
identical to the rejected model. The same deterministic `32` directions and
`+0.05/-0.05` cloned-OSC labels are replayed; the first `24` directions train
the response and the final `8` remain held out. The one training state is also
used for checkpoint selection because this is deliberately a capacity and
memorization test, not a generalization test.

The local ridge solution is the attainable linear-response reference. The
compact MLP must fit the training directional equations with cosine at least
`0.9`, sign accuracy at least `0.85`, and RMSE at most `1.1` times the local
ridge fit RMSE. Its predicted 15D rows must have mean cosine at least `0.9`
with the ridge rows. On the eight held-out directions, its cosine may be at
most `0.05` below the ridge teacher and its RMSE at most `1.1` times the ridge
teacher. Near-active L5 rows are reported separately but do not silently alter
the unchanged loss.

Interpretation:

- pass: the previous 1,055D representation was a material source of model
  underfit; the next independent gate may add exactly one physical state group;
- global pass but near-active failure: keep compact inputs and test weighting
  as the next single change;
- failure to match the ridge reference: input reduction alone is insufficient;
  next audit the mixed loss or decoder capacity;
- both MLP and ridge fail held-out directions: finite-radius or long-horizon
  nonlinearity limits the response target and must be tested separately.

This experiment cannot show state generalization because the nominal action
chunk is constant within one physical state. It trains no controller, solves
no QP, and executes no corrected action.
