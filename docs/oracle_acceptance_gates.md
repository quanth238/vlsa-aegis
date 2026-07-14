# Oracle acceptance gates

These are pilot decision criteria, not deployment guarantees.

| Gate | Evidence | Pass condition |
|---|---|---|
| H03 measurement | calibrated primitive, 50 unique states x 5 repeats, substep transforms, contacts, visualization | variation below 0.1 mm; 126 samples/replay; no contact at positive conservative clearance |
| H04 calibration | +/-axis response on 10 states; 20 held-out random prefixes | median endpoint error <=5 mm; 95th-percentile absolute D_opt/D_sim error <=10 mm; zero false-safe cases at a 10 mm predicted margin |
| H05 projection | all colliding nominal prefixes | feasible fraction at least 0.60; direct teacher safety at least 0.95; median endpoint error at most 5 mm |
| H06 direct replay | feasible projection population | repaired `A+` remains safe when replayed from identical state |
| H07 sampler | converted checkpoint, fixed-noise trace tests | conversion parity established; correction scale/sign/integral verified |
| H08 paired arms | same state/noise direct/random/oracle/bridge | schema-valid, no pairing violations, no hidden exclusions |
| H09 oracle analysis | clustered held-out population | oracle rescue at least 0.50; oracle-minus-random at least 0.20; paired 95% CI strictly above zero; median endpoint error below 1 cm |

If H05 five-step feasibility is below 0.40, H=10 may be tested once. If feasibility remains below 0.60, pivot to longer-horizon planning. If the midpoint intervention fails, only `t_s` 0.3 and 0.1 may be tested on validation. If none significantly beats equal-norm random, reject this intervention representation.

The stopped H sequence never authorized a learned steering network. Under the
endpoint-free pivot, R03 is the replacement oracle-analysis prerequisite.

## Endpoint-free pivot gates

| Gate | Evidence | Pass condition |
|---|---|---|
| R00 reach calibration | complete nominal first-five artifacts on disjoint calibration groups | at least 50 safe, phase-valid, positive-progress chunks; frozen `inverted_cdf` Q25 artifact |
| R01 endpoint-free feasibility | frozen 20 H05 cases; bounded proxy search plus repeated direct simulator verification | at least 12/20 reproduce the nominal collision and have a changed-action witness with `D_sim >= 5 mm`, no contact or >1 mm substep scene motion, and reach progress at least `p_min` |
| R02 oracle intervention | same state/observation/noise/horizon direct, random, analytic, and endpoint-free oracle arms | schema-valid paired artifacts; direct witnesses remain separate from flow outcomes |
| R03 oracle analysis | complete episode groups and clustered intervals | oracle direction beats equal-norm random under preregistered safety-progress criteria |
| R04 learned ECG | continuation-matched labels, false-safe audit, gradient causality, closed-loop test | authorized only after R03 passes and ADR-0012 rejects analytic geometry as an explanation using grouped oracle-minus-analytic inference and an exact paired test |

R01 search exhaustion is not a certificate. Report model candidates, verified
witnesses, proxy false-safe/false-negative cases, safe-without-calibrated-
progress cases, and invalid states separately. Never tune `p_min`, search
budget, guidance time, or guidance strength on the frozen 20-case population.
