# Explicit Local-Affine OSC Execution-Jacobian Gate

Validated job `38586` shows that the candidate-conditioned trajectory decoder
can fit ordinary joint motion while shrinking action sensitivity almost to
zero. This experiment changes the decoder, not the loss or data.

For each state, candidate index zero is the nominal two-action anchor. The
network outputs a direct nominal displacement and an explicit local execution
Jacobian at every horizon:

\[
\hat q_k(A)=q_0+\Delta\hat q_k^0+
\hat J_k(A-A_0),\qquad k=0,\ldots,50.
\]

Both \(\Delta\hat q_0^0\) and \(\hat J_0\) are exact architectural zeros.
The rollout contains 25 controller substeps per action, so columns for the
second action are exactly zero through substep 25 and become active at
substep 26. A two-way action-phase encoding is added to the existing
positional time encoding.

The immutable 7,905-rollout population, grouped train/validation split,
complete structured OSC input, five seeds, optimizer, schedule, direct joint
loss, symmetric safety-normal loss, and train-RMS-normalized paired-secant
loss are unchanged. The secant loss now supervises the explicit Jacobian
directly. Candidate joint trajectories are evaluated through the existing
known FK and seven L5--L7 ellipsoid traces.

The fitted gate requires both train and validation aggregate and terminal
joint-sensitivity cosine at least `0.8`, median norm ratio in `[0.5,1.5]`,
mean relative norm error at most `0.5`, and validation joint RMSE no greater
than `1.1` times the frozen direct model. Stored candidate secants must match
the explicit Jacobian within `1e-5 rad/action`, and all causally forbidden
Jacobian entries must be exactly zero.

A pass authorizes only a separately preregistered prediction test on new
grouped unseen episodes. A failure stops for decoder/optimization diagnosis.
New labels, the unopened task-3 episodes, residual-bound calibration, QP,
closed loop, Poisson/SDF, and classifiers are forbidden in this experiment.

Config SHA-256:
`8aae74181f24db35c7b86c5fdd17ec84d62bb9fd05ab034d5b3b72a688eb79bf`.
