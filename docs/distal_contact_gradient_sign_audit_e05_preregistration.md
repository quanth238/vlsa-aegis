# E05 contact-gradient sign and alignment audit

## Question

Does the immutable job-`37416` implementation actually evaluate and execute
the negative gradient of its learned contact-risk scalar, or did a sign,
normalization, or action-indexing error invalidate the previous result?

This is an implementation audit, not a new learned method. No model is trained
and closed-loop execution is forbidden.

## Frozen pair

At held-out E05 states 187 and 190, let

\[
R(u)=\max_{\ell\in\{L5,L6,L7\}}p_\phi(\mathrm{contact}_\ell\mid s,u),
\qquad
\hat g=\frac{\nabla_uR(u)}{\|\nabla_uR(u)\|_2}.
\]

For each `epsilon` in `0.0001, 0.001, 0.01, 0.05, 0.1`, construct without
clipping

\[
u^- = u-\epsilon\hat g,\qquad u^+ = u+\epsilon\hat g.
\]

Only Cartesian XYZ entries may change. The remaining four action entries must
be bitwise identical, and the last three feature entries must equal the
candidate XYZ action.

## Model consistency

The implementation passes only if `R(u-) < R(u+)` for every state/radius pair
by more than `1e-12`. At the smallest epsilon, the central finite-difference
slope

\[
\frac{R(u^+)-R(u^-)}{2\epsilon}
\]

must match the autograd norm within 1% relative error. This jointly checks the
sign, normalization chain rule, and XYZ feature indexing.

## Simulator alignment

Every paired action receives a fresh complete cloned OSC transition from the
same state. No mesh distance is used. Simulator preference is determined by:

1. zero raw L5--L7 contact over all internal substeps;
2. fewer recorded contact events;
3. smaller summed MuJoCo contact penetration.

The result reports whether the simulator prefers `u-`, `u+`, or ties. A
correct internal model ordering with any simulator preference for `u+` is
direct evidence of supervision/model misalignment. Ties are inconclusive.

## Decision

- Any internal ordering, finite-difference, or indexing failure: inspect and
  fix implementation before changing supervision.
- Internal audit passes but simulator disagrees: implementation is consistent;
  pointwise contact supervision is inadequate for steering.
- Internal audit passes with simulator agreement/ties: no implementation bug
  is found, but job `37427` still shows that the gradient is not exceptional
  relative to random directions.

No outcome authorizes closed-loop testing. If implementation passes, the next
model may use paired directional-ranking supervision and must beat the same
matched-random gate on unseen states.

