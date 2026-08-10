# Shared time-conditioned execution decoder preregistration

## Question

Does replacing the flat 357-output trajectory head with a decoder shared over
time reduce terminal OSC-execution bias while preserving the one-sided
geometry loss that removed all 44 test false-safes?

The three-state audit has already closed the missing state-coverage branch:
all three states pass the frozen complete-state support test. It also showed
that E05/E10 contain no exact-safe action among the 64 registered candidates,
whereas E15 contains 44 exact-safe actions rejected by every model member.
This experiment tests the execution model only. It does not expand candidates.

## Matched arms

The immutable baseline is job `38070`, which maps one complete OSC state and
two complete 7D actions to all `51 x 7` joint residuals through one flat head.
The experimental model uses

\[
\hat q_k=q_0+\tau_k D_\theta(E_\theta(x,A),\eta(k)),
\qquad \tau_k=k/50,
\]

where `E` is a shared state/action encoder and `D` is one shared decoder. The
fixed time encoding contains `tau` and sine/cosine features at frequencies
1, 2, 4, and 8. Multiplication by `tau` enforces the known initial condition
exactly.

Dataset, complete inputs, grouped 60/10/15 splits, targets, seeds, optimizer,
schedule, Huber losses, finite-difference sensitivity supervision, and
one-sided L5--L7 geometry supervision are unchanged. To bound CPU cost, each
training update uses terminal substep 50 plus three registered random interior
substeps; checkpoint selection and evaluation always use all 51 substeps.

## Gate

The decoder is only a diagnostic mechanism GO if it simultaneously achieves:

- zero validation and test false-safes;
- accepted safe support in every validation/test state that actually contains
  an exact-safe registered candidate;
- validation/test safe recall at least 50%/90%;
- test near-boundary RMSE no worse than 2.671122 mm;
- joint and margin sensitivity cosine at least 0.8;
- strictly lower terminal joint RMSE on both validation and test than job
  `38070`;
- strictly more test-state support than job `38070`.

E05/E10 states with zero exact-safe registered candidates are reported but are
not falsely counted as decoder rejection. Even a complete mechanism pass only
authorizes evaluation on newly reserved complete episodes. It does not
authorize calibration, a QP, exact action execution, or closed loop.

No new controller rollout labels, candidate expansion, classifier, surface
loss, Poisson/SDF, uncertainty calibration, QP, or closed-loop execution is
permitted in this gate. E05/E10/E15 are diagnostic only because they have
already influenced model selection.
