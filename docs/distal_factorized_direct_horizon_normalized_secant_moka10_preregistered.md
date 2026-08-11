# Direct-Horizon Normalized Paired-Secant Gate

This matched ablation tests one hypothesis from validated job `38560`: the raw
secant objective underweights small and late-horizon action sensitivities.

The only change from job `38376` is

\[
L_{\mathrm{sec}}=
\frac{1}{|V|}\sum_{(d,k)\in V}
\frac{\|\hat s_{dk}-s_{dk}\|_2^2}{c_{dk}^2+\epsilon},
\]

where `c[d,k]` is the train-only RMS L2 joint-sensitivity scale. Cells below
`1e-5 rad/action` are excluded and inverse-scale weights are clipped to
`[0.001,1000]` times their training median.

Architecture, inputs, trajectory loss, symmetric safety-normal loss, data,
episode splits, seeds, optimizer, schedule, and early stopping stay fixed.
There is no one-sided loss, calibration, QP, closed loop, new simulation, or
new unseen-episode evaluation.

The fitted train/validation gate requires aggregate and terminal sensitivity
cosine at least `0.8`, norm ratio in `[0.5,1.5]`, mean relative norm error at
most `0.5`, and validation trajectory RMSE no greater than 1.1 times the
frozen direct model. A pass authorizes only a new grouped-unseen prediction
experiment; a failure selects a time-decoder change.

Config SHA-256:
`a9661a6ab8725374f509e61ffc11861216ba4809692f448cc9f44d0b731ab4df`.
