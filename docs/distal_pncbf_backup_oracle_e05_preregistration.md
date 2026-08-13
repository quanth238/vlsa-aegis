# Fixed policy-value backup oracle for E05

The PNCBF construction requires a fixed continuation policy before its
maximum-over-time violation value can be learned. This gate defines that
policy without training.

Replay immutable AEGIS through action 181 and execute the registered
contact-free five-action prefix at 182--186. Thereafter, frozen pi0.5 proposes
five executed actions from each measured observation and original released
AEGIS filters only the end effector. A cloned five-action L5--L7 rollout
updates a hysteresis warning: activate below +1 mm and release only above
+5 mm.

When active, generate only the current obstacle-normal repulsive residual,
constant over five actions, at correction norms
`0.25/0.5/0.75/1.0/1.5/2.0`. No tangent, route mode, endpoint constraint, or
task-return objective is allowed. Choose the smallest raw-contact/CAR-safe
candidate reaching +1 mm; if conservative geometry cannot reach +1 mm,
choose the raw-contact/CAR-safe candidate with greatest strict clearance
improvement. Fail closed if none exists. A nominal rollout with raw protected
contact or excessive CAR forces backup evaluation even if the conservative
proxy warning did not activate. Execute all five selected actions, observe,
and repeat.

Primary success is native task completion with zero executed L5--L7 contact
and CAR at most 1 mm. Proxy clearance is a warning/selection signal and is
reported separately from physical acceptance. Passing would define a usable
oracle policy for collecting policy-conditioned maximum-future-violation
labels; it would not yet be a learned or formally verified CBF.
