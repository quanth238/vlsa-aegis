# Exact MuJoCo obstacle-box oracle

Status: preregistered after freezing the 8 mm outcome and before this H100
outcome is inspected.

## Question

Can accurate obstacle geometry plus the existing eight-row oracle-affine QP
produce a raw-safe action for the primary E05 false-safe transition?

This is the second user-requested test. It retains the exact job-`37109`
action-192 state, immutable prefix and nominal action, accepted seven L5--L7
robot slabs, released AEGIS EE ellipsoid, zero clearance target, all 87
candidates, 60-candidate trust region, affine calibration, QP tolerances, OSC
transition, and exact internal-substep verification. It does not carry the
failed 8 mm margin into this experiment.

## Sole geometry intervention

The frozen released-AEGIS obstacle MVEE is replaced by the 15 collision-active
moka-pot boxes compiled inside MuJoCo. Their identities and box-only type were
discovered by immutable clean H100 job `37163` and are hash-bound here. Each
box uses its exact compiled half sizes and its live MuJoCo position and
rotation at every internal step. There is no box-to-ellipsoid conversion and
no geometric padding.

For each of the eight robot/EE ellipsoids, clearance is the minimum over all
15 exact oriented boxes of the center-axis support gap

\[
h_i=\min_k \left(\lVert c_k-c_i\rVert-
\rho_i(n_{ik})-\sum_j a_{kj}|(R_k^Tn_{ik})_j|\right).
\]

Requiring every `h_i >= 0` requires a separating center-axis projection from
every box and is conservative for the registered proxy pair. Point coverage
of each raw contact is audited separately in exact box coordinates.

## Decision

The inherited ordered GO gate remains unchanged: contact geometry authority,
a jointly raw- and proxy-safe local candidate, zero calibrated affine
false-safes, a valid QP, and an exactly verified raw/proxy-safe QP transition.
Negative interval-start clearance is retained, not erased. The test may
therefore conclude that action 192 is too late for the exact-box/robot-proxy
barrier.

This is a privileged simulator-mesh oracle (the object happens to be compiled
from boxes, not mesh geoms). It tests mechanism feasibility only. It is not a
deployable perception method, population result, or safety certificate.

Immutable discovery:

- job `37163` result file SHA-256:
  `3db37092b2c5573e70cfc604bbbf01f1361be822ec87c8733d2c80b84685b0fd`;
- result payload SHA-256:
  `2eef9d767354f6692d36f57899bd511c261733e05394d00c997eacc7252fc051`;
- registered exact-box config SHA-256:
  `cc568a85c2acf215beda1cef4abcc31a92b3f6d3772d6c36147410afd471bf9f`.

No geometry, margin, candidate, solver, or decision setting may be tuned after
outcome inspection.
