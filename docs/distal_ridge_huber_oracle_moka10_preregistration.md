# Ridge-Huber affine target oracle preregistration

This experiment tests target stability before any new neural training or
closed-loop execution. It preserves the job-37580 fit dataset, the job-37649
off-grid exact labels, all complete episode splits, and all Table 1 artifacts.

At each of the same 50 states and for each of seven L5--L7 rows, the nominal
value is fixed to the exact two-step OSC margin `b=m(u0)`. Only the gradient
is fitted. Responses are expressed in millimetres and use boundary weights
`1 + 9 exp(-|m|/5 mm)`, a 2 mm Huber transition, and ridge strength `1e-4`.
The one-sided error is the maximum fitted-action affine overestimate plus
1 micrometre. No minimum-L1 coefficient enters this fit.

Gradient stability uses 16 deterministic 80% candidate resamples. An active
row is one with an unsafe fitted action or a fitted margin within 5 mm of the
boundary. Every active row must retain cosine at least 0.9 and relative norm
difference at most 0.25; near-zero rows use a fixed absolute threshold.
The minimum-L1 comparator is refitted on the identical 16 subsets, so target
stability is compared independently of action sampling.

Ridge-Huber and the immutable minimum-L1 targets are evaluated on the same
4,800 off-grid job-37649 actions and exact labels, which neither method used
for fitting. Each method also receives a new seven-row QP and exact two-step
cloned-OSC rollout at every state. The released AEGIS EE proxy is diagnostic;
the gate is exact L5--L7 proxy safety plus zero raw protected contact and at
most 0.1 mm obstacle motion.

Ridge-Huber authorizes the later learned gate only if all conditions hold:

- zero off-grid false-safe actions globally;
- at least one accepted exact-safe off-grid action in every state;
- every active row passes the resampling stability gate; and
- every state has a valid QP whose newly executed rollout is exactly safe.

If authorized, the later model preserves the 30/5/15 grouped state split and
adds an action-level one-sided false-safe penalty to coefficient loss. It must
have zero held-out false-safes, mean gradient cosine at least 0.8, feasible
QP output, and fresh exact safe rollouts on all 15 test states. Closed-loop
E05 remains forbidden until every learned gate passes.
