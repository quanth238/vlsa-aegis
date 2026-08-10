# Fixed overlapping multi-region affine oracle preregistration

This no-training experiment tests whether the two sparse-support failures of
the single ridge-Huber affine oracle come from representing the complete
`L_inf <= 0.5` action box with one half-space per L5--L7 constraint. It reuses
the immutable 50-state, 125-action job-37580 fit data and the untouched 4,800
off-grid exact actions from job 37649. Table 1 is read-only.

Each state action box is normalized to `[-1,1]^3`. Every axis has three fixed,
overlapping intervals `[-1,0]`, `[-0.5,0.5]`, and `[0,1]`, producing 27 boxes.
Each box contains exactly 27 points from the existing five-point grid and has
an exact grid point at its registered center. Seven independent boundary-
weighted ridge-Huber rows are anchored at that center and tightened by the
maximum fitted overestimate plus 1 micrometre.

For each state and clearance arm (0 mm and +1 mm), one seven-row QP is solved
in every region using that region's hard action bounds. Every valid QP proposal
is executed in a fresh cloned two-action OSC rollout. The chosen action is the
exactly verified-safe proposal with minimum L2 correction from the nominal,
breaking ties by region index. Affine acceptance without exact safety is
counted as a false-safe proposal; proposals are never silently discarded.

The same immutable off-grid actions and exact labels evaluate the multi-region
union and the job-37688 single-affine comparator. An action is predicted safe
by the multi-region model only when at least one containing region accepts all
seven rows. Resampling uses 16 deterministic 80% subsets within each region;
every active region/constraint fit must meet cosine 0.9 and relative-norm
difference 0.25.

Each clearance arm independently requires zero off-grid false-safes, zero
false-safe regional QP proposals, accepted exact-safe support in every state
where such immutable support exists, and an exactly safe selected QP rollout
at every state. The 0 mm arm must specifically recover state indexes 4 and 19,
the two safe-support misses from job 37688. The overall oracle passes only if
both 0 mm and +1 mm arms pass. No MLP is trained and no closed-loop E05 run is
performed by this experiment regardless of outcome.
