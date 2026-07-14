# 0014 — Use numeric portability for derived R02 reconstruction

Status: accepted on 2026-07-14 after replacement smoke `27403`, before any
20-case R02 population allocation.

## Context

The replacement smoke completed and wrote a final paired artifact.  Its runner
validated that artifact on the allocation host.  An independent verifier on a
different CPU runtime reconstructed the same seeded-random and analytic
directions with maximum absolute differences of `2.220446049250313e-16`.
The historical-versus-fresh direction angle differed by
`7.416192659981391e-12` degrees because `acos` amplifies a one-ULP cosine
difference near one.  No stored action, simulator outcome, arm correction, or
gate differed.

Requiring byte equality between independently reduced float64 derivations is
therefore not a portable artifact condition.  It is separate from the exact
within-allocation requirements on observations, noise, eager traces, policy
replay, executed actions, and simulator repeats.

## Decision

Keep every stored array and diagnostic content-bound.  Keep the exact binding
from each stored model direction to the float32 correction sent to the policy,
and keep the analytic diagnostic direction exactly equal to the stored applied
direction.

For only the independent reconstruction checks, accept finite float64 values
when their absolute difference is at most `1e-12`.  Preserve mapping keys,
sequence lengths, integer indices, strings, booleans, and nulls exactly.  For
the derived angle in degrees, accept at most `1e-10` absolute difference to
cover the observed conditioning of `acos`.  Meaningful direction, selected
geometry pair, source action, norm, bounds, or outcome changes still fail.

## Consequences

- The completed smoke can be validated on both the allocation and independent
  verifier hosts without changing any scientific outcome.
- The tolerance is nine orders of magnitude below the registered `1e-3`
  direction-tamper regression; a separate boundary test rejects reconstruction
  differences above the tolerance.  It does not relax action bounds,
  clearance, progress, pairing, or statistical gates.
- The population may launch only after the updated validator, regressions, and
  local gate pass from a clean commit.  The old smoke remains immutable; a new
  smoke ID must verify the updated commit before the population launch.
