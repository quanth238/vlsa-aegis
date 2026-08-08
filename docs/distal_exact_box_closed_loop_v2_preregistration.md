# Distal-only 8 mm closed-loop v2 preregistration

Version 2 is the narrow semantic correction identified by frozen H100 job
`37185`.  Every policy, state/noise, action-horizon, candidate, QP, cloning,
raw-contact, obstacle-motion, and success setting from v1 remains unchanged.

Only the obstacle paired with the released EE row changes:

- L5--L7 rows: the exact 15 live MuJoCo moka-pot boxes with `8 mm` targets;
- released EE row: the original frozen AEGIS perception MVEE and released EE
  proxy with a `0 mm` target.

Thus exact simulator geometry and the warning margin are now genuinely
distal-only.  The nominal action still comes from live pi0.5 followed by the
unchanged released AEGIS EE QP.  Every row is evaluated over the interval
start and every internal MuJoCo state, and exact raw L5--L7 contact plus
`0.1 mm` within-step obstacle-motion vetoes remain mandatory.

The run continues until native task success, the registered horizon, or an
explicit no-verified-candidate failure.  `primary_problem_solved=true` still
requires task success, no L5--L7 contact, no paper CAR, and exact clone/main
agreement for every execution.  Scope remains one privileged E05 feasibility
test, not deployable perception, population efficacy, whole-arm safety, or a
formal certificate.
