# Cumulative joint-displacement execution model result

## Verdict

Clean H100 job `38240` is an independently validated strict **NO-GO**.

The model satisfies the structural requirement that the observed initial
configuration is exact, `q_hat_0=q_0`. It does not predict the future OSC
rollout accurately enough for safety. Integrating 50 learned per-step joint
increments compounds signed error through the horizon.

| Diagnostic metric | Required | Job 38240 |
|---|---:|---:|
| Test false-safes | 0 | **0** |
| Validation false-safes | 0 | 30 |
| Test exact-safe recall | >=90% | 25.99% |
| Test eligible-state support | 13/13 | 5/13 |
| Test near-boundary RMSE | <=2.671 mm | 33.724 mm |
| Test terminal joint RMSE | <=20.587 mrad | 410.690 mrad |
| Joint-sensitivity cosine | >=0.8 | 0.746 |
| Margin-sensitivity cosine | >=0.8 | 0.185 |
| Initial joint error | exactly 0 | **0** |

Zero test false-safes is not a safety success here. The model accepts only
211 of 812 exact-safe test actions and supports five of thirteen recoverable
states. It is mostly conservative because its predicted rollout has drifted
far from the physical rollout. Validation also retains 30 dangerous
false-safes.

## Root cause localized by the experiment

The test joint RMSE grows almost monotonically from `8.441 mrad` at substep 1
to `410.690 mrad` at substep 50. Validation grows from `16.633 mrad` to
`908.280 mrad`. This is the signature of accumulated increment bias. It is
already severe on validation, so it is not explained only by the diagnostic
E05/E10/E15 distribution shift.

The exact-q geometry audit from job `38155` remains controlling evidence:
exact joints passed through the same MuJoCo FK, ellipsoid attachment, and
analytic support gap have `0.321 mm` boundary RMSE, zero false-safes, and
`99.51%` recall. Therefore job `38240` does not implicate FK, the ellipsoid
formula, or QP linearization. No QP was used in this experiment.

The rejected package is:

\[
\boxed{
\text{per-substep increment prediction}
\rightarrow
\text{50-step cumulative integration}
\rightarrow
\text{explicit geometry}
}
\]

This does not reject factorized execution prediction. The previous flat
one-sided model remains the best current learned safety result: zero test
false-safes, `93.84%` recall, `12/13` eligible-state support, and joint/margin
sensitivity cosines `0.977/0.921`. Its remaining failure is state-specific
support, not cumulative drift.

## Consequence

Reserved E01/E11/E21/E31/E41 labels were not collected. Trust-region
sequential QP correction and closed-loop E05 are not authorized.

The next prediction hypothesis should distinguish displacement from
increment integration: predict each horizon displacement
`Delta q_k=q_k-q_0` directly, keep `Delta q_0=0` exact, and supervise
multi-horizon endpoints without recursively summing predicted increments.
That is a new preregistered model gate, not a threshold repair to job `38240`.

## Reproducibility

- Allocation: job `38240`, `worker-2`, one H100, `00:32:09`.
- Source: commit `c1d130369465293c535be3b6cdda6d65696d6e58`, clean worktree.
- Immutable run:
  `/mnt/data/quanth/experiments/vlsa-distal-factorized-cumulative-delta-q/cumulative-delta-q-20260811b`.
- Model SHA-256:
  `511810494a12e15c8ef7b1d839738499cb4f194f4dc19c8bcfa1e13731140c1b`.
- Predictions SHA-256:
  `af1ec13590f2dab5531e3274c5b5d4619dea28159fe782d4beaa09f949f76736`.
- Result/validation SHA-256:
  `78243f0c874780565cecb8f9e74a185141fa67930de182e2b68d812818ac2435`,
  `b30c9dc4a2ddc0a01286a9a92fa626bd81176662b2a00286b61d8aeef9342bcb`.
- Result/validation payload SHA-256:
  `6e35fb4a53389e1f94cdd72cb406a0addec818545a606d1bdae4d2818774cf5d`,
  `4b2e5f7457c7dc94a25d30e2b1b436992334009fce9c053d3a404b24ba75a4be`.
- Independent replay: all eight checks passed; maximum joint, margin, and
  exact-static-margin differences were exactly `0.0`.

Job `38239` remains an apparatus-only failed attempt before training; none of
its outputs were reused.
