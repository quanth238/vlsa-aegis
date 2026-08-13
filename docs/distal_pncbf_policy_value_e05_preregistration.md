# Exact fixed-backup policy-value gate

This gate evaluates one deterministic, input-admissible backup policy before
any neural approximation.  It retains the validated H100 E05 rollout setup:
the immutable state and proposal ledger, seven L5--L7 rows, five-action
receding windows, Schmitt-trigger warning, fixed normal-repulsion magnitudes,
deterministic selection, and fail-closed behavior.

The augmented state includes the complete simulator/controller state, the
warning latch, the proposal-ledger position, and the current five-action
proposal.  Thus the registered policy is Markov on the augmented state even
though its physical-state projection alone is not.

For each selected internal-substep clearance, define

\[
h_j=d_{\mathrm{safe}}-d_j,\qquad d_{\mathrm{safe}}=1\ \mathrm{mm}.
\]

At every receding decision state, construct the exact finite-horizon value

\[
V_j(z_t)=\max_{n\ge t}h_j(z_n)
\]

over the remaining registered backup rollout plus a 25-action verified
terminal hold.  If hold is unsafe, use the preregistered obstacle-normal
retreat.  No terminal invariant-set or infinite-horizon claim is made.

Pass requires:

- all seven exact values are nonpositive at every recorded decision state;
- zero protected MuJoCo contact and CAR pass;
- exact Bellman recursion
  \(V_j(z_t)=\max\{\max_{k\in\mathrm{prefix}}h_j(z_k),V_j(z^+)\}\);
- a verified +1 mm terminal tail;
- independent action-ledger replay.

This gate authorizes only later value approximation.  It does not authorize a
neural-CBF, QP, invariance, task-generalization, or learned-control claim.
