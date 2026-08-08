# Exact MuJoCo obstacle-box oracle result

Clean H100 job `37180` completed on `worker-1` in 46 seconds from commit
`e8ae3e30265d7cb808886aab297b5f92b3f472f5`. The allocation tests and
independent validator passed. The 15 box identities were bound to immutable
discovery job `37163`, and Table 1 remained read-only.

## Outcome

The exact boxes repair obstacle contact authority but do not make the existing
action-192 QP solve the case.

- All 15 obstacle primitives are the exact compiled MuJoCo boxes, with zero
  inflation and live pose updates.
- Every nominal L6/g12 raw contact is correctly witnessed: exact source-box
  containment is `0.991--0.999`, the accepted robot slab containment is
  `0.903--0.910`, and the minimum pair support gap is
  `-4.315` to `-4.679 mm`.
- Removing the Loewner inflation materially tightens the interval-start rows,
  but `L5_part_1=-11.009 mm` and `L6_part_0=-3.892 mm` remain negative.
- Because every candidate shares that interval-start state, none of the 87
  candidates is proxy-safe. Raw-safe local candidates still exist.
- OSQP reports `primal infeasible` after 50 iterations; no QP action is
  executed or exactly verified.

Therefore, accurate obstacle geometry is necessary but insufficient at this
late state. The exact-box proxy recognizes danger before the nominal raw L6
contact begins at internal substep 11, but the hard minimum-substep constraint
also demands that the already-negative interval start be nonnegative. No
current action can change the past start state. The negative L5 row without a
raw L5 contact additionally exposes remaining conservatism in the robot-slab
and center-axis proxy.

The retained conclusion is scoped: exact simulator obstacle geometry plus the
existing one-state QP cannot solve action 192. It does not prove that an
earlier exact-box activation cannot solve the episode. The scientifically
next test must locate the first exact-box crossing in the immutable action
ledger and invoke the same QP one transition earlier, before `h<0`; that test
requires separate preregistration.

## Artifact identity

- Result file SHA-256:
  `d57850d8d8b04e3601d0cb870cca84f15bc41684eae71dc2cb57c9414c1d10fb`
- Canonical result payload SHA-256:
  `30d48753b457d1b52f935a9d5a5d4912f9b2aad651751f76b1e9edf6ab442f11`
- Validation file SHA-256:
  `146710ecd82255b5652dec895c2767ae3a7bedcce4319c409629a98c0dedd224`
- Evaluation preflight SHA-256:
  `a1bbb6e989505bbdc56c60d1c93c1b0ef052fef0af9dd3ec72aa91152955f8ec`
- Remote result:
  `/mnt/data/quanth/experiments/vlsa-distal-oracle-exact-box-obstacle-e05/exact-box-20260809a/result.json`
- Local evidence copy:
  `/Users/quanth238/personal/Research/probe_vla/output/vlsa_distal_oracle_exact_box_obstacle_e05/exact-box-20260809a/result.json`
