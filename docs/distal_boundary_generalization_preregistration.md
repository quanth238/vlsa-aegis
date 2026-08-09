# Grouped distal boundary generalization pilot

This E02 pilot follows the accepted action-188 local-capacity result without
reusing E05 for model fitting or calibration. It remains an
outcome-conditioned moka-pot mechanism test, not an unbiased SafeLIBERO
population estimate and not a formal safety certificate.

The immutable ten-case selection contains only Table-1 AEGIS episodes with
the same active `moka_pot_obstacle_1` collision geometry. Episodes that were
already in protected-link contact after settling are excluded. Complete task
groups are assigned once:

- train: three `spatial-I-t1` and three `goal-II-t2` episodes;
- validation: one unseen `goal-II-t3` recovery-stress episode;
- test: three `goal-II-t0` episodes, including primary E05.

For each episode, the collector replays the immutable AEGIS action ledger and
selects the first exact seven-proxy nominal crossing in the 20-action window
ending at the registered collision. From that state it evaluates a fixed
8-by-8-by-8 Cartesian grid in the existing `L_inf <= 0.5` trust region,
selects the 16 safe and 16 unsafe candidates closest to the +/-5 mm boundary,
and evaluates six central finite differences per anchor. Exact labels cover
every internal OSC/MuJoCo substep and keep optimizer gaps separate from raw
contact and obstacle-motion verification.

The exact 15-box union is the obstacle authority for every recorded
clearance. Some older Table-1 PCA MVEEs contain a reflective basis that the
ellipsoid constructor correctly rejects. Because the probe constructor still
requires an obstacle object even when the exact-box union overrides it, the
validated primary-E05 MVEE is reused only as a proper-rotation placeholder;
it does not enter any recorded clearance, witness, label, or acceptance gate.

Training is blocked until all ten episodes have a recoverable crossing,
balanced anchors, and at least one witness-stable active-row gradient. The
architecture, paired margin-only and gradient-supervised arms, validation
calibration, and bounded seven-row projection are unchanged from the accepted
local-capacity experiment.

The gradient arm authorizes closed-loop E05 only if every held-out test
episode has active-boundary RMSE below its current-clearance baseline, zero
conservative false-safe candidates, mean active-row gradient cosine at least
0.8, and a valid seven-row QP whose exact cloned transition has nonnegative
seven-proxy gaps, zero protected raw contact, and at most 0.1 mm obstacle
motion. Failure stops the sequence before closed-loop E05.
