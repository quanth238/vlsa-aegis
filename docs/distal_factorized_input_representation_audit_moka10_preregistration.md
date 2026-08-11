# Factorized complete-input representation audit

## Question

Why do both the flat one-sided execution model from H100 job `38070` and the
shared time-conditioned decoder from job `38076` produce physically impossible
validation trajectories while their diagnostic-test trajectories remain at
ordinary scale?

This is a frozen, no-training audit. It does not test another architecture and
does not change normalization, safety geometry, candidates, labels, QP, or
closed-loop control.

## Immutable population and models

The audit reuses the 7,905 two-action rollouts at 85 grouped states, their
2,110D complete OSC/state representation, both 7D actions, and the original
60/10/15 state split. It hash-binds the complete dataset, trajectory arrays,
both five-member model ensembles, their predictions, results, and independent
validations. Recomputed predictions must match the immutable arrays exactly.

## Audit

The 2,124 inputs receive their stored semantic names: 2,110 aligned
state/controller scalars plus 14 action coordinates. Training-row mean and
standard deviation are recomputed and must exactly equal the values stored by
both models. For every feature, state, episode, and split, the audit records
raw training variance and maximum absolute z-score. Features are classified as
presence, padded value, train constant, binary/categorical, action continuous,
or continuous physical.

Every linear and SiLU output is measured for every ensemble member on train,
validation, and test rows. The first module is called explosive only when its
validation maximum is at least `100` and at least `100` times its training
maximum.

Finally, the 24 fixed semantic groups with largest validation z-scores are
masked one at a time to their training mean, only for a read-only validation
forward pass. A group is a strong causal explanation only if masking it reduces
terminal-joint RMSE by at least 90% in both immutable models. This mask is not a
proposed deployment representation and cannot alter either completed result.

## Interpretation

- A representation artifact requires an explosive feature from a presence,
  padding, constant, or binary/categorical group and a strong causal reduction.
- A physical support shift requires a continuous physical/action group beyond
  10 training standard deviations and a strong causal reduction.
- If neither occurs despite internal activation explosion, the result remains
  a model-numerics or output-parameterization problem.

The audit authorizes no retraining. Its only output is the root-cause class and
the next matched experiment to preregister. E05/E10/E15 remain diagnostic.
Configuration SHA-256:
`e9e1b4d954696a704a28e1982270235ce1ff65d25166199fbf3eef4fa3805043`.
