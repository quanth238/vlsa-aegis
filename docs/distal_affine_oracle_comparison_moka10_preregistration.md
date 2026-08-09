# Direct half-space versus finite-difference affine oracle comparison

This is the no-training decision experiment requested after learned
coefficient job 37630. It asks whether the failure comes from the non-unique
minimum-L1 regression target or from the one-local-affine-row assumption.
It does not train the MLP, solve a closed-loop task, or audit OSQP.

The immutable job-37580 dataset supplies 125 fitted actions at each of the
same 50 paired simulator states. For every L5--L7 row, the direct arm fits a
class-balanced logistic half-space from only the sign of the exact two-step
OSC margin. Its threshold is then moved just beyond the largest unsafe fit
score, guaranteeing zero fit false-safe actions without using the old
minimum-L1 coefficients.

The comparison arm measures a canonical central finite-difference gradient
at the exact nominal action with epsilon 0.02. Its exact nominal value and
gradient are tightened by the maximum one-sided overprediction on the same
125 fit actions, plus 1 micrometre. Thus both arms see the same fitted actions
but construct their affine rows differently.

Evaluation uses 96 new deterministic uniform actions inside each state's
registered box: 4,800 fresh two-action cloned-OSC rollouts, plus 300 finite-
difference probe rollouts. The immutable second AEGIS action is unchanged.
Every internal OSC substep is measured. A fresh action is truly safe only if
all seven L5--L7 proxy margins are nonnegative, raw protected contact is zero,
and obstacle motion is at most 0.1 mm. The released AEGIS EE proxy is retained
only as a diagnostic because this experiment isolates the seven distal rows.

For each method, the frozen GO gate is:

- zero false-safe fresh actions globally;
- aggregate true-safe recall at least 0.20; and
- at least one accepted true-safe action in at least 0.80 of states that have
  fresh true-safe support.

If either method passes, a local affine representation is supported and a
future MLP may be trained with a direct one-sided half-space loss and held-out
uncertainty tightening. If neither passes, the one-row local representation
is inadequate at this trust-region scale; the next experiment must reduce the
trust region or use multiple local regions. No outcome authorizes closed-loop
E05 in this experiment.
