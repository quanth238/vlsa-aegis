# Fixed overlapping multi-region affine oracle result

## Verdict

**NO-GO under the frozen all-gates criterion.** The regional representation
substantially improves zero-margin safe-set coverage and QP feasibility, but it
does not recover both registered sparse-support states with stable fits. The
+1 mm arm also has one freshly verified false-safe QP proposal.

No MLP was trained and no closed-loop E05 experiment ran.

## Paired off-grid comparison

Both methods used the same 50 states, 4,800 immutable fresh actions, exact
two-action OSC labels, and L5--L7/raw-contact gates.

| Clearance | Method | False-safe | Safe recall | Accepted / exact safe | Accepted-support states | Exact-safe selected QPs |
|---|---|---:|---:|---:|---:|---:|
| 0 mm | 27-region | 0 | 99.85% | 3,957 / 3,963 | 49 / 50 | 50 / 50 |
| 0 mm | single affine | 0 | 95.76% | 3,795 / 3,963 | 48 / 50 | 49 / 50 |
| +1 mm | 27-region | 0 | 99.71% | 3,796 / 3,807 | 48 / 49 | 49 / 50 |
| +1 mm | single affine | 0 | 94.83% | 3,610 / 3,807 | 48 / 49 | 49 / 50 |

The regional union therefore recovers 162 additional zero-margin safe actions
and one additional zero-margin support state relative to the single-affine
oracle. This is evidence that local regions improve representational capacity.

## Required sparse states

- State index 4, `vlsa-t1-spatial-i-t1-e00`, step 120: recovered. The regional
  oracle accepted 6/7 exact-safe zero-margin off-grid actions (single affine
  accepted 0), all active fits were stable, and its chosen QP rollout had
  minimum distal clearance `0.04896 mm` with zero raw contact.
- State index 19, `vlsa-t1-goal-ii-t2-e00`, step 30: not recovered under the
  frozen support gate. Both methods rejected its one immutable safe off-grid
  action. Regional QP search nevertheless found three exact-safe zero-margin
  proposals; the closest required a `0.5879` normalized-action correction and
  had minimum distal clearance `0.2795 mm`. This state's active fit was
  unstable and no +1 mm regional QP was valid.

Thus multiple regions improve QP feasibility even in the hardest state, but
the registered conservative support-recall and stability requirements do not
pass.

## Stability and positive buffer

Thirteen active region/constraint fits failed across five states. Every failure
was L5-part-2. Direction remained above the cosine threshold (minimum `0.9139`),
but resampled gradient norm differed by as much as `50.18%`, exceeding the
registered `25%` limit. Localizing to 27 samples per region reduces bias but
raises coefficient-variance sensitivity near the boundary.

The +1 mm arm had zero false-safes on all 4,800 off-grid actions, but one of
its valid regional QP proposals was false-safe at E15 step 150. The predicted
active rows were on the 1 mm boundary; the fresh rollout reached
`0.999489 mm`, missing the buffer by `0.000511 mm` while remaining collision-
free with zero obstacle motion. A different proposal was exactly +1 mm safe,
but the frozen gate audits every proposal and therefore fails.

## Execution evidence

- Slurm job: `37690`, H100 `worker-1`, completed in `00:14:50`.
- Fresh cloned-OSC rollouts: `2,567`.
- Source commit: `61d6e6641521e17454c5ee1f7ea06a4e05e27c45` (clean).
- Remote root: `/mnt/data/quanth/experiments/vlsa-distal-multi-region-affine-oracle/multi-region-affine-20260810a`.
- Local copy: `output/vlsa_distal_multi_region_affine_oracle/multi-region-affine-20260810a`.
- Result file SHA-256: `a14c8f3c3f06d881a4ed83db94edf139f4ca5f2a9f53381f3b182a7e28ac2a5f`.
- Result payload SHA-256: `a80b343a4810d746086ef933eead6e86f1ec61a0f123ee08b1a459c7274de265`.
- Validation SHA-256: `a24175762be7fecee2308c7f8a9dcb6f9bc82cf2ed2edaaf4949e53776e46e13`.

## Interpretation

The experiment rejects this fixed 27-region ridge-Huber oracle under the
requested strict gate, so it does not authorize MLP training or closed-loop
E05. It does not reject piecewise-affine execution-aware safety in general:
zero-margin selected QPs were exact-safe in all 50 states and coverage improved
strongly. The remaining issue is a sparse boundary component plus local
coefficient variance, not an inability of regional QPs to find any safe action.
