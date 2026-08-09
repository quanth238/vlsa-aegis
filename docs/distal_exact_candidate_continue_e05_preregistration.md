# Unsafe continuation after the E05 empty safe set

## Question

What happens if the exact-candidate controller does not stop when its sampled
safe set is empty at action 187?

This is not a safety method. It deliberately executes actions after the safety
filter has declared that no registered safe action exists.

## Frozen execution

- Hash-bind clean H100 job `37294` and its independent validation.
- Reconstruct actions 0--186 from its accepted action ledger and require every
  dynamic-state and obstacle-displacement receipt to match.
- At action 187, latch `safety_filter_bypassed=true`.
- Execute the immutable released-AEGIS Table-1 nominal actions 187--236 without
  candidate search, QP, stopping on a proxy violation, stopping on raw contact,
  or stopping on paper CAR.
- Continue until native task success or exhaustion of the 237-action plan.
- Measure all eight proxy rows at every internal OSC/MuJoCo substep and retain
  raw protected contacts, obstacle displacement, and native goal progress.

The config SHA-256 is
`f1cec8e8479be6b71f54e2239cbf1c21a5054be40aaa610753e469e42c6386c2`.

## Interpretation

Report native task success and the first proxy violation, protected contact,
and paper-CAR steps. `safety_success` is fixed false because the controller
knowingly bypasses an empty safe set. A later single-context replay must match
the complete accepted action/state ledger before its video is presented.
