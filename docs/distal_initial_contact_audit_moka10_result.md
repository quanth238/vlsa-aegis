# Distal initial-contact audit result

## Verdict

**PASS: all 85 registered states are prevention states.**

Clean H100 job `37882` completed on `worker-2` in `00:06:47`; independent
validation passed. The canonical archived prefixes reconstructed all 17
episodes and all grouped states:

| split | prevention | recovery |
|---|---:|---:|
| train | 60 | 0 |
| validation | 10 | 0 |
| test (E05/E10/E15) | 15 | 0 |
| total | 85 | 0 |

Every state had zero registered protected L5--L7 contact at `k=0`. Zero-cutoff
query sign matched raw contact exactly: zero negative-without-contact, zero
contact-without-negative, and zero unregistered pairs. Positive native distance
was not queried.

The rejected 1 m `mj_geomDistance` experiment's 29 apparent recovery states
were therefore measurement artifacts, not initial collisions. Including `k=0`
in the two-step ellipsoid rollout target does not make the current 85-state
learning population impossible by definition.

This result authorizes only a separately registered prevention-cohort model
experiment. It does not authorize QP execution or closed-loop E05.

## Consequence for the next ordered experiment

The existing action-conditioned model already used the same 60/10/15 grouped
states, 53D complete q/qdot/controller-relative proxy geometry plus candidate
action, and quantitative two-step ellipsoid rollout margins. It improved RMSE
to 3.808 mm but failed with 21 proxy false-safes and support in only 12/15
states. The initial-state audit eliminates unsafe `k=0` as an explanation for
that failure. Rerunning the unchanged model would be redundant; calibration
and QP remain blocked until a prediction-only model passes.

## Immutable evidence

- source commit: `514986cc0b1793db8edcbbe4e3ae5fee63d5b9b4`;
- run root:
  `/mnt/data/quanth/experiments/vlsa-distal-initial-contact-audit/initial-contact-audit-20260810b`;
- result SHA-256:
  `b6f57361a7259de49790061b1d04c1c74a1e6f80e18bc04ab09433f607fb034a`;
- result payload SHA-256:
  `f424932506c2e2b808ec3be151638fe6c1ce69a3ce12b090e8f7875c4406170a`;
- validation SHA-256:
  `40e1e8946eb06296032475b2e238cd53fba2e848758cfd602292cad949fe6fdc`;
- preflight SHA-256:
  `2d178fc92fc5659e1c68012eed748e3dad6d143b8eed580e8211087ec9fe3e8a`.
