# Distal two-step oracle affine safe-set preregistration

## Question

Before training another neural model, determine whether the proposed online
representation is capable of expressing a useful safe action.  For each
registered test state, can seven conservative affine lower envelopes fitted
with privileged exact two-step labels produce a bounded minimum-deviation QP
action that remains safe under a fresh exact OSC rollout?

This is a representation feasibility test.  It is not learned generalization,
closed-loop E05 efficacy, population efficacy, or a formal certificate between
the sampled actions.

## Frozen evidence and cases

The experiment reuses the immutable job-37270 dataset and only records whose
`source` is `grid`.  It does not recollect or reinterpret Table 1.  The cases
are the complete registered held-out group: E05, E10, and E15.  Each must
contain exactly 512 actions at one selected pre-collision state, seven exact
L5--L7 minimum-substep margins over two OSC actions, and separate raw contact
and obstacle-motion evidence.  The second action remains the immutable
released-AEGIS action.

## Oracle affine certificate

Safe grid actions are ordered first by Euclidean correction from the nominal
first action and then by immutable grid index.  For a candidate and each of
the seven rows, solve a linear program for

\[
\ell_i(a)=b_i+G_i^\top(a-a_0)
\]

such that, for every one of the 512 sampled actions,

\[
\ell_i(a_j) \le m_i^{\mathrm{exec}}(a_j)-1\,\mu\mathrm m,
\]

and the candidate is certified by \(\ell_i(a_k)\ge0\).  Among feasible
coefficients the LP minimizes \(\lVert G_i\rVert_1\).  The first candidate
admitting all seven rows defines the state certificate.  A postcheck requires
zero sampled false-safe actions and no lower-envelope violation above
\(10^{-8}\) m.

The seven coefficients enter one hard QP with the original action limits and
minimum-deviation objective.  The QP is not allowed to execute its registered
safe grid target directly; it must solve for its own continuous action.

## Exact authority and decision

The QP action is executed in a cloned simulator through both real OSC
transitions and every MuJoCo substep.  A case passes only if:

1. a grid action is both distal-proxy and raw-simulator safe;
2. all seven sampled-grid conservative affine certificates exist;
3. the QP has seven input rows and is valid;
4. every exact L5--L7 margin and the unchanged released-AEGIS EE proxy margin
   remains nonnegative over both actions;
5. there is no protected raw contact and obstacle motion stays below the
   registered 0.1 mm per-step threshold.

`representation_go=true` requires all E05/E10/E15 cases to pass.  A GO only
authorizes collection of state-to-affine-coefficient targets under a new data
gate.  A NO-GO rejects this registered single-affine sampled-grid construction,
not all piecewise-affine, nonlinear, or simulator-search safe sets.  No neural
training and no closed-loop task rollout are part of this experiment.
