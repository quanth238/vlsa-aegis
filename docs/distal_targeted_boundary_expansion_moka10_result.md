# Targeted same-task boundary expansion result

## Verdict

**GO for a separately preregistered retraining of the current region-aware
MLP.** The expanded training population supports all 15 immutable E05/E10/E15
states, and the fixed regional oracle is smooth for all 15 matched
comparisons. No MLP or closed-loop controller ran in this gate.

## H100 evidence

- Slurm job `37771`, `COMPLETED` on `worker-2` in `00:13:17`.
- Clean source commit:
  `cda0897a08c25ce9722c1dd84f03d9227b8ea848`.
- Allocation: one NVIDIA H100 80 GB HBM3.
- Four complete additional training episodes: E25/E35/E40/E45.
- New labels: 20 states and 2,500 two-step cloned-OSC grid rollouts.
- Expanded grouped split: 60 train, 10 validation, 15 immutable test states.
- Independent recomputation: `valid`.

Every episode selected its own closest nonnegative five-state L5--L7 boundary
window. The anchor clearances were 17.274 mm (E25), 28.523 mm (E35),
49.293 mm (E40), and 0.119 mm (E45). No E05/E10/E15 feature or label selected
these states.

## Support result

| Measure | Job 37742 | Job 37771 |
|---|---:|---:|
| Supported test states | 7/15 | 15/15 |
| Smooth oracle pairs | 15/15 | 15/15 |
| Maximum test feature shift | 5.601 z | 3.555 z |
| Features above 5 z | 2 | 0 |

The expanded cross-episode support threshold is `1.111101` RMS z. Test-state
nearest-neighbor distances are `0.588950--0.872968`, and every test state also
passes the five-z featurewise bound. E05 is now supported primarily by E25;
E10 and early E15 are supported primarily by E45. The largest remaining
feature shifts are nominal second-action x (`3.555 z`) and nominal
first-action x (`3.409 z`).

## Separate compatibility failures

The inherited single-affine/all-eight collector gate remains false, but this
does not remove a state or invalidate the seven-row regional support result:

- E25 step 186 has no exact-safe sampled two-step candidate in the local box.
- E35 steps 71--75 have valid distal certificates, valid distal QPs, positive
  L5--L7 margins, zero raw contact, and zero obstacle motion, but the separate
  released AEGIS EE proxy is negative by 1.372--10.992 mm.

All 20 raw state records and regional labels are retained. The learned method
must continue to report the original EE proxy as a separate compatibility
diagnostic rather than treating it as one of the seven learned L5--L7 rows.

## Decision

Preregister and retrain the unchanged region-aware coefficient-output MLP on
the 60/10/15 split. It must pass all of the following before closed-loop E05:

- zero conservative false-safe test actions;
- accepted safe-action support in all 15 test states;
- valid seven-row regional QPs;
- fresh exact cloned-OSC verification of every selected QP action.

If the supported-state model fails, the coefficient-output architecture—not
missing same-task data support—is the next rejected component. The following
model should then predict conservative safety values conditioned on state,
region, and candidate action.

## Artifact identity

- Result file/payload SHA-256:
  `68be3d07d0d616037ff90e25299c8309ee7d2b5e9353e181f67887b81098b428` /
  `5879931851f8991d9bae09f896c7962c2d2790e5e9d2ec9d4d5212f50cd24472`.
- Validation file SHA-256:
  `68a84e6b5caddbe12e9f490f798822e606ca8a31bb5167d9e78af4d49183c620`.
- Expanded dataset file/payload SHA-256:
  `98158b3ea85993945ddc5170747af8a14a929597becddcce92d484f343d95c80` /
  `629142dd9b7710b7f26f00d001fd5ee3781cc1a000fef82c9d9fca9aa79ab41b`.
- Expanded oracle file/payload SHA-256:
  `5ac8efe88fed934ade3ec36512aa2a0d870142bd91ae09923e7cae577aea77b7` /
  `0716aae9cba8ef1bed5ce3b0800733593bfe8c550c599da68e5106729a72aebe`.

Remote immutable root:
`/mnt/data/quanth/experiments/vlsa-distal-targeted-boundary-expansion/targeted-boundary-expansion-20260810a`.
