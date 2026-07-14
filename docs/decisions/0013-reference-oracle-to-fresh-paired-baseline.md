# 0013 — Reference the oracle direction to the fresh paired baseline

Status: accepted on 2026-07-14 after apparatus failure `27393` and before any
R02 simulator action or outcome.

## Context

The first R02 smoke allocation, job `27393`, stopped before executing any
action.  It produced only `launch-failure.json`; no `r02-paired.json` exists.
The policy's two eager requests and their midpoint traces were byte-identical,
but the runner rejected two historical-pairing checks.

The branch check compared a fresh `ReachSnapshot.to_dict()` containing Python
tuples with the JSON-loaded R01 snapshot containing lists.  Their serialized
values can be identical while direct Python equality is false.  This is an
apparatus representation bug, not state evidence.

The second check required a fresh eager action chunk to be byte-identical to
the action chunk serialized by the earlier R01 allocation.  R01 did not freeze
the rendered observation tensor, and byte identity across independently
constructed simulator/policy allocations was never established.  As a
diagnostic, the already accepted parity allocation's eager first-five
translation differs from the raw R01 nominal by at most `0.0035328335` action
units (RMS `0.0014743872`), despite using the same registered case and noise.
Byte identity across allocations was not established.  ADR-0010 did,
independently and before any R02 outcome, freeze numerical parity limits for
the same physical action fields.

The causal R02 comparison requires exact pairing among the arms in the current
allocation.  It does not require the current frozen action to equal an earlier
allocation's output.  Retaining the historical nominal as the oracle reference
would also make the correction target stale: `A_current + (A_witness -
A_R01_nominal)` need not equal the immutable safe witness.

## Decision

Canonicalize the fresh branch snapshot through JSON-compatible values before
requiring exact equality to the content-bound raw R01 branch snapshot.  No
coordinate tolerance is introduced.  A genuine branch-state difference still
fails before any rollout.

Keep the raw R01 nominal actions and their content hash in every artifact.
Treat fresh-eager versus R01-nominal byte equality as a diagnostic only, but
fail before simulator execution unless the historical drift satisfies the
already-frozen ADR-0010 limits:

```text
first-five translation: maximum <= 0.010, RMS <= 0.005
first-five x seven actions: maximum <= 0.050, RMS <= 0.015
```

These limits are reused, not fitted to job `27393`.  The required current-run
policy pairing is:

1. duplicate fresh eager actions and midpoint traces are byte-identical;
2. every intervention arm uses that same state, observation, explicit noise,
   horizon, and pre-intervention trace;
3. the final frozen eager replay is byte-identical after all intervention
   requests; and
4. the current frozen collision and immutable direct witness are both
   reconfirmed by repeated simulator rollouts.

For each eligible case, define the applied oracle direction from the current
paired baseline and the immutable R01 direct witness:

```text
Delta_current_phys[0:5,0:3]
    = A_R01_witness[0:5,0:3] - A_current_eager[0:5,0:3]
Delta_current_model = Delta_current_phys / action_scale
```

All other entries are zero.  Conversion remains scale-only.  Preserve
`A_R01_witness - A_R01_nominal` separately as a source diagnostic, but never
apply it.  The seeded-random and analytic-geometry controls use exactly
`||Delta_current_model||_2`; all existing mask, midpoint, residual schedule,
bounds, state, observation, noise, and executed-horizon controls remain
unchanged.

The case artifact validator must independently reconstruct both the historical
diagnostic and the applied current direction from raw actions, including their
norm and angle difference.  A historical drift-limit failure, or a missing,
zero, nonfinite, mistargeted, or tampered direction, fails closed.  Nominal or
direct reconfirmation failure remains a terminal population mismatch and is
not repaired by changing the reference.

## Consequences

- This fixes an invalid cross-allocation byte-identity premise before observing
  an R02 simulator outcome; it reuses stricter pre-outcome numerical limits and
  does not relax a safety, progress, bounds, or statistical threshold.
- The oracle arm tests whether the current frozen flow can move toward a known
  action that is directly reconfirmed from the same branch.
- Historical policy drift remains visible and reportable, rather than being
  hidden behind a post-hoc tolerance.
- The failed `27393` smoke ID remains immutable.  Any rerun uses a new ID and a
  clean commit containing this decision and its regression tests.
