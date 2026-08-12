# E05 learned repulsive-force direction result

Verdict: **strict local NO-GO for learned force weighting**.

Clean H100 producer job `38789` ran on `worker-2` from commit `4b4765b`.
Independent H100 validation job `38791` recomputed the registered norms,
gains, p-values, source and config hashes, and the fail-closed continuation
decision.

At untouched E05 step 185, the nominal two-action hard L5--L7 margin was
`-9.313104 mm`.  The learned and fixed directions had cosine `0.997199`.

| Action radius | Learned gain | Fixed gain | Learned - fixed | Random p | Learned hard margin | Safe |
|---:|---:|---:|---:|---:|---:|:---:|
| 0.10 | 1.003028 mm | 1.006700 mm | -0.003673 mm | 1/257 | -8.309539 mm | no |
| 0.25 | 2.509951 mm | 2.516463 mm | -0.006512 mm | 1/257 | -6.801919 mm | no |

The learned physical field is much more useful than random steering: none of
256 matched random directions achieved as much soft-min gain.  However, the
fixed analytical soft-min field was slightly better at both radii, and the
registered local correction region contained no exact-safe learned, fixed,
or random action.  Therefore the MLP added no demonstrated control value and
the conditional two-action continuation correctly did not execute.

This does not show that obstacle repulsion is useless. It shows:

1. the paired cloned-OSC pullback supplies a meaningful push-away direction;
2. learning only monotone row weights reproduces that analytical direction;
3. direction weighting cannot repair inadequate safe support in the local
   two-action candidate region.

The result does not justify enlarging the learned model or running closed
loop. The clean next scientific comparison would test fixed physical
repulsion with a preregistered larger/iterative trust region or an earlier
intervention state. Only after exact safe support exists is it meaningful to
ask whether a learned task-conditioned force improves over the fixed field.

Immutable artifacts:

- producer result:
  `/mnt/data/quanth/experiments/vlsa-distal-repulsive-force-direction-e05/repulsive-direction-20260812c/result.json`
- result SHA-256:
  `7ddaf40e25ce7a31e7687229bc86890583d87b604cd615c58cfdadb34a6ce0a2`
- validation receipt:
  `/mnt/data/quanth/experiments/vlsa-distal-repulsive-force-direction-e05/repulsive-direction-20260812c/validation.json`
- validation SHA-256:
  `7f64d148dd3861d94e196be063ec73ec654381f64de5c9f2024bee58604ae38a`
