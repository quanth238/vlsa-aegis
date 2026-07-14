# 0003: Keep two intervention semantics distinct

Status: accepted — 2026-07-14

The primary CRFS intervention applies `-Delta/t_s` to the velocity at every remaining Euler step. Because integration uses negative `dt`, the additive contribution is `+Delta` at the final action under a constant base field.

A separate one-shot bridge edit adds `(1-t_s) Delta` to `x_t` and then resumes the unmodified base flow. The nonlinear policy can respond differently to these operations, so the harness reports them as separate arms and never treats one as an implementation of the other.
