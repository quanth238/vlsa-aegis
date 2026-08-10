# Action-conditioned execution-margin MLP preregistration

## Question

Can a model that learns the quantitative two-step cloned-OSC L5--L7 margin at
one candidate action reproduce the validated regional oracle on the 15
supported unseen E05/E10/E15 states?

This is the registered replacement after job `37795` rejected the
coefficient-output parameterization. It is an unseen-state feasibility gate,
not a closed-loop or whole-arm safety claim.

## Immutable population

The job reuses job `37771`'s 85 states and exact labels without new data:

- train: 60 complete-episode-grouped states;
- validation: 10 complete-episode-grouped states;
- test: 15 E05/E10/E15 states, strictly test-only;
- fit grid: 125 exact two-action cloned-OSC actions per state;
- validation/test off-grid: 96 immutable actions per state.

Test states, actions, margins, contacts, and oracle decisions cannot influence
normalization, checkpoints, calibration, hyperparameters, or model selection.

## Direct value model

For one protected constraint row and candidate first action, the 56 inputs are
the validated 53D pair/state/geometry vector plus the candidate's normalized
coordinates in the current action box. The shared five-member MLP predicts a
scalar residual in millimetres:

\[
\hat m_r(s,u)=d_r(s)+f_\phi(s,r,u),
\]

where \(d_r(s)\) is current clearance and the target is the exact minimum
margin over both actions and every OSC/MuJoCo substep. No affine coefficient is
a training target.

Training uses boundary-weighted Huber value regression and a one-sided
overestimation penalty. Validation selects checkpoints. The conservative point
value is

\[
\underline m_r(s,u)=\operatorname{mean}_e\hat m_{r,e}
-2\operatorname{std}_e\hat m_{r,e}-c_r,
\]

where \(c_r\) is the maximum guarded overestimate on validation grid and
off-grid actions plus 1 mm. Calibration is per constraint and never sees test
labels.

## Regional QP bridge

At an unseen state the model is queried at the fixed 125-action grid. For each
of the 27 overlapping regions, the validated ridge-Huber routine fits seven
conservative affine rows to these guarded values. Thus the MLP learns values,
while deterministic online regionalization supplies the QP coefficients.

One bounded seven-row QP is solved per region. The selected proposal is the
valid solution closest to the nominal VLA action. Simulator outcomes are not
used for selection. If the off-grid gate passes, the selected action receives
one fresh exact two-step cloned-OSC rollout for evaluation only.

## Frozen gates

The model passes only with all of the following:

- zero false-safe decisions over 1,440 unseen off-grid actions;
- global/worst-state accepted-set Jaccard at least `0.90/0.80` relative to the
  validated regional oracle;
- accepted exact-safe support in all 15 test states;
- 15 valid regional QPs;
- 15/15 fresh exact distal-safe selected-action rollouts;
- 15/15 separate released-AEGIS-EE-compatible rollouts;
- selected-action shift from the oracle no more than `0.10` p95 and `0.25`
  maximum in normalized action units.

A full pass authorizes a separately preregistered receding-QP E05 run. A
failure rejects the plain action-conditioned MLP on this supported population;
closed-loop remains blocked.

Config SHA-256:
`b40ff3a14f4644ea91233e5bdcb69060ccce526ac2b3bca502715531d36792af`.

