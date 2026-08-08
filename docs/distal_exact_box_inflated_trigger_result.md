# Exact-box 8 mm rounded-shell first-crossing result

Clean H100 job `37183` completed on `worker-1` in 43 seconds from commit
`df97f3e5bc54be34ce4bef01ed88552638666a2f`. Allocation tests and the
independent validator passed, and all 13 immutable prefix actions reproduced
their exact cloned next-state hashes.

## Registered outcome

The first-crossing QP test passes at action 13.

- Current exact clearance remained outside the shell: the minimum was the
  released EE row at `11.256 mm`.
- The nominal transition predicted that row would fall to `7.354 mm`, crossing
  the `8 mm` rounded shell. All seven distal rows remained at least
  `105.715 mm` clear.
- There were 21 jointly raw- and inflated-proxy-safe registered candidates and
  zero calibrated affine candidate false-safes.
- The valid eight-row QP changed XYZ from
  `[-0.3132,-0.1425,-0.6578]` to `[-0.4134,-0.3588,-0.3911]`, an action-space
  correction of `0.3578` L2.
- Exact OSC execution had zero L5--L7 contacts, zero obstacle displacement,
  and minimum EE clearance `9.554 mm`; all distal rows remained above
  `107.720 mm`.
- QP total wall time was `2.230 ms`; the exact verification step took
  `110.057 ms`. The complete scan and audit took `40.631 s`.

## Interpretation

This proves the intended narrow mechanism: enlarging the exact boxes with a
rounded `8 mm` shell and activating before the crossing makes the QP feasible
and exactly safe for that transition. It also exposes an important limitation:
the deterministic first trigger is caused entirely by the released EE proxy,
not an L5/L6/L7 row. Therefore this result does **not** demonstrate avoidance
of the later primary L5/L6 collision or preservation of task completion.

The next direct distal experiment would keep the released EE target at its
base value and apply the `8 mm` warning shell only to L5--L7 constraints, then
run the resulting filter closed-loop. That is a link-specific margin rather
than a single globally enlarged physical obstacle primitive and requires a
separate protocol.

## Artifact identity

- Result file SHA-256:
  `1591a3bc1774be86f79f54941ac53cf622281041630038b6d1b0001ca3e8efe3`
- Canonical result payload SHA-256:
  `26eea3c6044b77e4f75b0c6ef18b77a8d398e78ab233788cff97dc7573edd32e`
- Validation file SHA-256:
  `72a9d5aff2dbf5b018f2a75f6262162e3962c42aa31ba528e209328e30811a58`
- Evaluation preflight SHA-256:
  `bf04ffd7de9aa0e215b61f8a9c315474674dbfec48b29ea3ce4d900308098fc9`
- Remote result:
  `/mnt/data/quanth/experiments/vlsa-distal-exact-box-inflated-trigger-e05/inflated-trigger-20260809a/result.json`
- Local evidence copy:
  `/Users/quanth238/personal/Research/probe_vla/output/vlsa_distal_exact_box_inflated_trigger_e05/inflated-trigger-20260809a/result.json`
