# Multi-region decision-stability audit preregistration

Job 37690 remains a strict NO-GO. This follow-up asks whether its 13 unstable
raw L5-part-2 coefficients materially change the safety decisions used by the
regional controller. It contains no MLP training and no closed-loop E05 run.

For every one of the same 50 states and 27 fixed regions, 16 deterministic
80% resamples select 22 of the existing 27 regional grid actions. One subset
is shared across all seven L5--L7 rows so each replicate defines a coherent
seven-row QP. The exact region-center value remains fixed. Gradients are fitted
on the subset, while the one-sided error is recalibrated on all 27 regional
labels. This isolates coefficient variation without weakening the immutable
sampled lower-bound check.

At zero margin, each replicate is evaluated on the same 96 immutable off-grid
actions per state. Accepted safe-set decisions are compared with the complete-
fit job-37690 decisions. One QP is solved per region; the nearest valid proposal
is selected without using the simulator outcome, then that single action is
freshly executed through the exact two-action cloned OSC. This produces 800
selected-action rollouts rather than re-verifying all 21,600 regional QPs.

Decision stability requires:

- zero false-safe decisions across 76,800 replicate/action evaluations;
- global accepted-set Jaccard at least 0.98 for every replicate;
- state/replicate accepted-set Jaccard at least 0.80;
- safe support retained in all 49 complete-fit supported states;
- all 800 selected QP actions valid and fresh exact-safe;
- selected-action L2 shift from the complete fit at most 0.05 at p95 and 0.15
  maximum.

Passing means the observed coefficient non-uniqueness is decision-harmless for
this sampled zero-margin population and may authorize a later grouped MLP
preregistration. It does not change job 37690 to GO, establish formal safety,
or authorize closed-loop E05.
