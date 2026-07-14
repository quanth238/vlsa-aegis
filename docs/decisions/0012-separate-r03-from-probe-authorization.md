# 0012 — Separate oracle steerability from learned-probe authorization

Status: accepted on 2026-07-14 after sampler parity job `27389` and before any
R02 policy or simulator outcome.

## Context

R03's registered oracle-versus-random criteria test whether the frozen flow can
realize a known endpoint-free safe-progress direction.  Passing that test does
not show that a learned continuation-clearance probe is needed.  The analytic
geometry arm has the same correction norm, mask, intervention step, residual
schedule, state, observation, noise, and execution horizon.  If it explains
the gain, this controlled pilot supports geometry guidance rather than a
learned safety model.

The feasible-conditioned population has only 17 complete groups.  Rates move
in increments of `1/17`; an oracle-minus-random difference of at least `0.20`
therefore requires at least `4/17`.  Four oracle-only discordances and no
analytic-only discordance have a one-sided exact paired probability of
`0.0625`, so a percentile interval alone is not a sufficiently conservative
authorization for training.

## Decision

Keep the preregistered R03 scientific gate unchanged on all 17 complete paired
groups:

```text
SPSR_oracle >= 0.50
SPSR_oracle - SPSR_random >= 0.20
grouped 95% bootstrap LCB(SPSR_oracle - SPSR_random) > 0
```

Report a separate analytic-versus-random matched gate using the same three
conditions.  This is an explanatory diagnostic; it does not replace the R03
oracle gate.

Set `learned_probe_authorized=true` only when all of the following hold:

1. the R03 oracle-versus-random scientific gate passes;
2. the analytic-versus-random matched gate does not pass;
3. the grouped 95% bootstrap lower bound for
   `SPSR_oracle - SPSR_analytic` is strictly above zero; and
4. the exact one-sided paired label-swap test on oracle-versus-analytic
   discordances has `p < 0.05`.

For the exact test, if `b` groups pass only under oracle and `c` pass only under
analytic, report

```text
p = sum_{k=b}^{b+c} choose(b+c, k) / 2^(b+c)
```

with `p=1` when there are no discordances.  Bounds failures, direction
failures, and reconfirmation-terminal cases count as zero in their fixed
denominators; no arm-specific exclusion is permitted.  Bootstrap complete
state/episode groups with the already registered seed and at least 10,000
replicates.

Do not add an oracle-minus-analytic `0.20` threshold: there is no independent
calibration for it, and the exact paired test supplies the stricter small-sample
authorization.  `learned_probe_authorized` permits proceeding to R04; it is
not evidence that ECG is effective or novel.

Comparable intervention budget means identical per-case model-space L2 norm,
translation mask, intervention step, remaining residual schedule, paired
state/observation/noise/horizon, bounds policy, and one midpoint direction
evaluation.  Record inference latency and required geometry inputs separately;
equal correction norm is not a claim of equal computational cost.

## Consequences

- R03 can pass while learned-probe training remains unauthorized.
- If analytic geometry passes its matched gate, the current pilot does not
  justify a learned safety method.  A learned method would need a separately
  preregistered missing-geometry or deployment-efficiency hypothesis.
- Five oracle-only discordances and no analytic-only discordance are the
  smallest no-loss pattern that passes the exact `p < 0.05` rule.
