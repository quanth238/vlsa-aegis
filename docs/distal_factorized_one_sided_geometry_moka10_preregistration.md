# Factorized one-sided geometry supervision: preregistration

## Question

Does a signed, horizon-aware geometry loss remove the terminal L5 optimism of
the factorized OSC execution model on unseen E05/E10/E15 states without
destroying safe-action support?

This tests one refinement of the validated factorized mechanism. It does not
test a QP or closed-loop controller.

## Frozen evidence and conditional audit

The immutable job-`37980` model has useful joint/action sensitivity but 44
test false-safes. Job `38036` localized all 44 to L5 at substep 50, with the
same obstacle witness and signed center-clearance optimism rather than unusual
Euclidean surface error.

Before retraining, the job evaluates the untouched validation episodes with
the immutable job-`37980` model and known FK/ellipsoid geometry. Training is
authorized only when:

- validation contains at least one false-safe random action; and
- at least 50% of validation false-safes have their exact worst value in an
  L5 row at terminal substep 50.

This prevents the test-only observation from selecting a loss that has no
corresponding validation signal.

## Differentiable signed geometry signal

The deployment output remains the predicted joint trajectory

\[
\hat Q=[\hat q_0,\ldots,\hat q_{50}].
\]

No safety head or binary classifier is added. For training only, the known
seven-row ellipsoid clearance is locally linearized in joint coordinates at
each state, substep, and constraint:

\[
h_{rk}(q+\Delta q)\approx h_{rk}(q)+J^h_{rk}\Delta q.
\]

Each `J^h` is fit with fixed ridge `1e-6` from the nominal plus the 28
registered full-action finite-difference candidates. Only train and validation
episodes receive such Jacobians; test labels are never used for fitting or
training. The 64 random-antithetic actions audit the linearization out of fit.
Validation random-action clearance-delta RMSE must be at most 2 mm.

The induced signed clearance prediction error is

\[
e^h_{rk}=J^h_{rk}(\hat q_k-q_k).
\]

Positive `e^h` is dangerous because it means predicted clearance exceeds the
executed clearance. The experimental loss is

\[
\mathcal L=\mathcal L_q+\lambda_G\mathcal L_{\mathrm{sensitivity}}
+10\,\mathbb E\!\left[w_{rk}
\left(\operatorname{Huber}_{2\mathrm{mm}}(e^h_{rk})
+4[e^h_{rk}]_+^2\right)\right].
\]

`w_rk` is five times larger within 5 mm of the boundary and increases smoothly
from 1 at substep 0 to 3 at substep 50 using fourth-power horizon weighting.
The baseline is the immutable job-`37980` factorized model. The experimental
arm uses the same dataset, grouped splits, complete inputs, architecture,
ensemble seeds, optimizer, batch size, epochs, patience, joint loss, and
sensitivity loss. The only new training term is the signed geometry loss.

At evaluation, safety is computed from predicted joints through the same known
MuJoCo FK and existing exact-box L5--L7 ellipsoid support-gap evaluator as the
original factorized pilot. The local Jacobian is not a deployment safety
output.

## Frozen gate

On the untouched 960 E05/E10/E15 random-antithetic actions, all conditions are
required:

- zero proxy false-safe actions;
- accepted exact-safe support in 15/15 states;
- at least 50% exact-safe recall;
- near-boundary RMSE no greater than 3.058 mm;
- joint/action sensitivity cosine at least 0.8;
- action-space safety-margin sensitivity cosine at least 0.8; and
- strictly fewer false-safes than the baseline 44.

A pass authorizes only a separately preregistered validation-only uncertainty
experiment. A failure directs the next model toward a shared time-conditioned
dynamics decoder rather than another classifier.

## Exclusions

The experiment collects no new controller rollout labels and does not add
generic surface-position loss, a binary classifier, uncertainty calibration,
Poisson/SDF, a QP, VLA inference, or closed-loop E05 execution. Raw MuJoCo
contact remains separate binary `D_sim`; ellipsoid margin remains `D_opt`.

Frozen config:
`configs/vlsa_distal_factorized_one_sided_geometry_moka10.v1.json`, SHA-256
`740ad469f4b2cba2d38d27526dbf9d9b1f35b3264b6d6cb1ae6a8df6811988b5`.
