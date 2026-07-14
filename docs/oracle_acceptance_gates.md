# Oracle acceptance gates

These are pilot decision criteria, not deployment guarantees.

| Gate | Evidence | Pass condition |
|---|---|---|
| H03 measurement | calibrated primitive, 50 unique states x 5 repeats, substep transforms, contacts, visualization | variation below 0.1 mm; 126 samples/replay; no contact at positive conservative clearance |
| H04 calibration | +/-axis response on 10 states; 20 held-out random prefixes | median endpoint error <=5 mm; 95th-percentile absolute D_opt/D_sim error <=10 mm; zero false-safe cases at a 10 mm predicted margin |
| H05 projection | all colliding nominal prefixes | five-step feasible fraction at least 0.40 initially; direct teacher safety at least 0.95; median endpoint error at most 5 mm |
| H06 direct replay | feasible projection population | repaired `A+` remains safe when replayed from identical state |
| H07 sampler | converted checkpoint, fixed-noise trace tests | conversion parity established; correction scale/sign/integral verified |
| H08 paired arms | same state/noise direct/random/oracle/bridge | schema-valid, no pairing violations, no hidden exclusions |
| H09 oracle analysis | clustered held-out population | oracle rescue at least 0.50; oracle-minus-random at least 0.20; paired 95% CI strictly above zero; median endpoint error below 1 cm |

If H05 five-step feasibility is below 0.40, H=10 may be tested once. If feasibility remains below 0.60, pivot to longer-horizon planning. If the midpoint intervention fails, only `t_s` 0.3 and 0.1 may be tested on validation. If none significantly beats equal-norm random, reject this intervention representation.

No learned steering network is authorized before H09 passes.
