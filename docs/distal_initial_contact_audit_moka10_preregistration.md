# Distal initial-contact audit preregistration

## Question

Which of the 85 paired L5--L7 states are contact-free prevention states at
`k=0`, and which are already-contacting recovery states?

## Measurement

Replay the canonical archived action prefix for every complete episode on one
H100. At each registered state, discover the compiled protected and active
obstacle collision geoms, record raw protected contacts, and query only
`mj_geomDistance(..., distmax=0)`. Require exact pair-level agreement between
negative zero-cutoff sign and raw contact. Positive native distance is
forbidden.

Classify a state as prevention when no protected L5--L7 contact exists at
`k=0`; otherwise classify it as recovery. Preserve all 60/10/15 episode-grouped
states and report both cohorts without discarding either.

## Gate

The 85-state audit must have one semantic inventory, finite zero-cutoff
queries, no unregistered contacts, and no raw-contact/sign mismatch. A pass
authorizes only preregistration of training on prevention states. Recovery
states, if present, require a future-only metric excluding the fixed unsafe
`k=0`. No model, QP, or closed-loop E05 runs in this gate.
