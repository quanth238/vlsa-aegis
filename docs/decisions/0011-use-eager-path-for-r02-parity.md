# 0011 — Compare parity on the eager intervention path

Status: accepted on 2026-07-14 after apparatus failure `27381` and before any
R02 policy outcome.

## Context

ADR-0010 required the converted PyTorch compiled default, eager trace-only
sampler, and independently instrumented sampler to be byte-identical.  Slurm
job `27381` was the first execution of that rule.  It wrote a schema-valid
failed artifact (SHA-256
`b6be9eb8a31f272940c81ab27ced91b19226240aeb1cb021abb616585bf0e879`);
it did not run an R02 intervention or simulator outcome.

The public JAX instrumented loop reproduced its default exactly.  Every
registered cross-framework numerical condition also passed.  In particular,
the public-JAX versus compiled-PyTorch final first-five translation error was
0.0039634 maximum and 0.0016541 RMS, below the frozen 0.010/0.005 bounds.  All
ten latent and velocity comparisons passed.  The sole failed assumption was
compiled-PyTorch versus eager-PyTorch byte identity: maximum normalized error
0.0138757 and RMS 0.0015238.  The independently instrumented PyTorch loop had
the same discrepancy relative to the compiled output, consistent with the
compiled and eager kernels following the same Euler equations with different
floating-point implementations.

R01 used the eager trace-only sampler.  Every R02 intervention also requires
the eager path because it returns trace dictionaries and applies a correction.
Therefore compiled/eager byte identity is not the pairing identity needed for
the causal R02 comparison.  It is a baseline diagnostic.  The required exact
identity is eager trace-only versus an independently instrumented eager loop.

## Decision

Supersede only ADR-0010's compiled-versus-eager `array_equal` acceptance rule.
Do not change any numerical tolerance after observing job `27381`.

The replacement parity gate must:

1. require public-JAX instrumented final versus public-JAX default to be
   `array_equal`;
2. require PyTorch eager trace-only final versus the independently instrumented
   eager loop to be `array_equal`;
3. use public JAX versus PyTorch eager for the ten pre-update latent/velocity
   comparisons and the final normalized/physical R02-path comparisons;
4. also retain public JAX versus compiled PyTorch final comparisons as the
   ordinary-baseline diagnostic;
5. retain compiled PyTorch versus eager PyTorch final comparisons as a separate
   path-seam diagnostic; and
6. require every final comparison in items 3--5 to pass the unchanged ADR-0010
   limits: normalized 0.10/0.025 maximum/RMS, physical first-five translation
   0.010/0.005, and physical `10 x 7` actions 0.050/0.015.

The ordinary no-envelope OpenPI call remains compiled and unchanged.  R02 uses
the eager frozen request as its paired nominal because that is the exact path
used by R01 and by all intervention arms.  A compiled request is executed and
reported before and after the paired sequence, but numerical rather than byte
identity to eager is the registered requirement.

The failed `27381` artifact remains failed and is never used to authorize R02.
A new immutable allocation artifact must pass the revised validator before an
R02 case is run.

## Consequences

- This is a correction to backend-path identity, not post-outcome tolerance
  tuning; all quantitative bounds remain unchanged.
- R02 can test the intended intervention mechanism without claiming that
  `torch.compile` and eager kernels are bitwise implementations of one another.
- Any eager/instrumented mismatch, cross-framework tolerance failure, or
  compiled/eager tolerance failure still blocks R02 interpretation.
