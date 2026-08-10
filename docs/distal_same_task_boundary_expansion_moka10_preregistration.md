# Same-task boundary coverage expansion preregistration

## Question

Does adding complete boundary episodes from the same SafeLIBERO task remove
the measured state-support failure for the immutable E05/E10/E15 test states,
while preserving the validated fixed multi-region safety oracle?

This is a coverage diagnostic. It does not train an MLP, execute a learned QP,
run closed-loop E05, or establish formal/whole-arm safety.

## Frozen episode split

- Additional training episodes: goal-II-task-0 E00 and E20.
- Additional validation episode: goal-II-task-0 E30.
- Immutable test episodes: goal-II-task-0 E05, E10, and E15.
- The split unit is a complete episode. No state from E05/E10/E15 may affect
  collection, oracle fitting, normalization, or thresholds.
- This first establishes same-task unseen-episode coverage only; it is not an
  unseen-task claim.

## Frozen collection and oracle

For each added episode, register five states at offsets `[-4,-3,-2,-1,0]`
from its first two-step exact-proxy crossing. At each state, evaluate the same
125 Cartesian residual actions in the existing `L_inf <= 0.5` box through two
complete cloned OSC steps, recording all internal substeps and all seven
L5--L7 ellipsoid constraints.

Fit the existing 27 fixed overlapping regional ridge-Huber lower planes on
each new state without changing partition, loss, regularization, boundary
weighting, or one-sided padding. The original 50 state records and regional
targets remain byte-derived immutable inputs; the 15 new records are appended.

The expanded split is 40 training, 10 validation, and 15 test states.

## Decision gate

Recompute the frozen 62D feature-shift audit and the 43D state-context support
metric. Thresholds remain training-only:

- support distance: p95 nearest cross-episode training distance;
- per-feature support: at most five training standard deviations;
- oracle smoothness: training p95 union-margin RMSE and p05 accepted-set
  Jaccard at matched normalized action coordinates.

Current region-aware MLP retraining may be separately preregistered only if all
15 immutable test states pass both state support and regional-oracle
smoothness. If support remains incomplete, collect more complete episodes. If
support passes but the retrained MLP still fails the zero-false-safe/safe-
support gate, replace coefficient output with an action-conditioned
conservative safety-value model.

Closed-loop E05 remains blocked under every outcome of this experiment.

## Execution and provenance

Simulation and fitting run only inside one H100 Slurm allocation. The source
commit must be clean; Table 1 artifacts are read-only. The evaluator writes an
additional dataset, full collection provenance, expanded dataset, expanded
regional oracle, result, and independently recomputed validation receipt.

Frozen config SHA-256:
`8a81876fe4420b8ab09dcb628c215a830e60d4122bd0f81d1b53bc4fdea94c0d`.

Frozen selected-manifest SHA-256:
`63110fe7778be984ddd6a62582fdbf30ff8acba43ae277b78263c1bde23a87f8`.
