# Omitted-variable sufficiency audit

## Goal

Determine whether the historical 56D input is provably insufficient because
it omits goal orientation, rotation commands, gripper commands, or controller
memory, and separately recheck whether the already trained complete-input MLP
passes its held-out prediction gate.

## Test 1: deterministic target

Reuse and independently validate the immutable duplicate replay population:
10,625 complete snapshot/action pairs, each replayed twice. All seven float64
margins, canonical raw-contact receipts, and both next-state hashes must match
exactly. Any mismatch invalidates the audit.

## Test 2: controlled omissions

Audit all 85 states. At each state, select the closest safe and closest unsafe
candidate (or the closest absolute-margin candidate if one class is absent).
From the exact same restored state and translation action, independently vary:

- goal orientation by plus/minus 15 degrees around each axis;
- each first-action rotation component to minus/plus one;
- both gripper commands to minus/plus one;
- one controller-memory field at a time: update flag, relative orientation,
  orientation reference, or previous torque memory.

Recompute and byte-hash all seven old56 rows after every intervention. The
historical input is provably insufficient only if an identical old56 hash
produces both an all-seven-safe and an unsafe two-action rollout margin. Raw
MuJoCo contact conflicts are reported separately and are not substituted for
the proxy-margin definition.

The complete-input MLP is not retrained. Its immutable E05/E10/E15 metrics
must have zero false-safe actions and safe support in 15/15 states to pass.
The QP remains frozen; calibration and closed-loop E05 are forbidden.
