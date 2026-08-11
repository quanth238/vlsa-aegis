# Factorized nominal-safety formula audit preregistration

## Question

Validate whether the learned execution model truly predicts joint
configurations and whether the observed safety error is introduced by the
learned joint rollout or by the subsequent forward-kinematics/ellipsoid
calculation:

\[
\widehat h_{ik}=\phi\!\left(p_i(F_{\theta,k}(x_t,\bar a_t)),\mathcal O\right).
\]

## Frozen construction

The model output is required to be a `51 x 7` joint-configuration trajectory.
For every predicted and exact joint vector, the existing evaluator writes the
seven joints to MuJoCo, calls `sim.forward()`, reconstructs the seven L5--L7
ellipsoids in world coordinates, and evaluates the analytic minimum support
gap to the same exact-box obstacle union. No QP or affine safety
linearization is used.

The immutable job-38072 records provide, for all 5,440 random candidates:

- the cloned-OSC rollout target \(h^{\rm dyn}\);
- the exact joint trace \(q\) and predicted joint trace \(\hat q\);
- \(h^{q}=\phi(p(q),\mathcal O_0)\);
- \(h^{\hat q}=\phi(p(\hat q),\mathcal O_0)\).

The error is decomposed exactly as

\[
h^{\hat q}-h^{\rm dyn}
=
\underbrace{h^q-h^{\rm dyn}}_{\text{exact-q FK/geometry recomposition}}
+
\underbrace{h^{\hat q}-h^q}_{\text{joint-prediction safety error}}.
\]

The first term may include real obstacle motion after substep zero because the
static evaluator deliberately freezes the obstacle at its initial pose. At
substep zero, that confound is absent and directly checks the `q -> FK -> phi`
implementation.

## Gates and interpretation

The exact-q chain must have at most 1 micrometre substep-zero maximum error,
at most 0.5 mm test near-boundary RMSE, zero false-safe actions, and at least
99% exact-safe recall. The predicted-q chain must still exhibit false-safes,
and its boundary error must be at least twice the exact-q recomposition error.

Passing attributes the dominant error to learned execution prediction, not
MuJoCo FK, ellipsoid attachment, the analytic safety function, or QP
linearization. Failing requires a deeper FK/geometry audit before changing the
MLP. New simulation, training, calibration, QP, closed-loop control, and
Poisson/SDF geometry are forbidden.
