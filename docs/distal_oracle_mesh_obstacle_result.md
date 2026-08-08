# E05 conservative obstacle-primitive oracle result

## Verdict

The conservative live primitive union **repairs geometry authority**, but its
one-ellipsoid-per-box representation is already negative at the action-192
interval start.  The registered result is therefore NO-GO with
`no_jointly_raw_and_proxy_safe_action_in_local_trust_region`, not another
false-safe QP execution.

## H100 evidence

- Job `37163`, worker-1, clean commit
  `c8ca7914a7ebd5228948ecf5bef91da3ea2227e0`.
- Active obstacle: 15 collision-active boxes, zero meshes.
- Certificates: 15/15 verified closed-form Loewner box enclosures.
- Raw nominal: 15 L6/g12 contact substeps and `1.557 mm` obstacle motion.
- Geometry witness: all contact points inside the accepted robot slab and the
  exact contacted obstacle primitive; minimum pair gap was negative.
- Interval start: four of eight barriers already negative; minimum
  `-33.621 mm`.
- Local candidates: raw-safe candidates exist, but zero proxy-safe candidates
  because every trace includes the same negative start state.
- QP: invalid with `uncontrollable_constraint` on row 3; no unsafe action was
  proposed or executed.

## Interpretation

This run establishes that live physical obstacle geometry corrects the missing
contact label.  It does not yet establish that the physical filter must trigger
earlier, because each exact MuJoCo box was wrapped in one enclosing ellipsoid.
The Loewner box ellipsoid scales every half-axis by `sqrt(3)`, adding avoidable
empty volume.

The next smallest test uses the 15 live oriented boxes directly.  Exact box
support and point containment are conservative with respect to the simulator
collision geometry and strictly tighter than the ellipsoid wrapper.  The
state, candidates, robot/EE bounds, affine calibration, QP, and decision rule
remain unchanged.  If exact boxes are also negative at interval start, the
method must intervene earlier in the immutable action ledger.

## Artifact identities

- Result file SHA-256:
  `3db37092b2c5573e70cfc604bbbf01f1361be822ec87c8733d2c80b84685b0fd`.
- Result payload SHA-256:
  `2eef9d767354f6692d36f57899bd511c261733e05394d00c997eacc7252fc051`.
- Validation file SHA-256:
  `ede301e1c764e6362fdfa99f61d5b4516c7bff91c726f76f7a3216609a1dafe5`.
- Allocation preflight SHA-256:
  `314e7785c31007352f35f3e2f94a21adf7f793154ec981b06bb24ab4a2de8e12`.

This is single-state mechanism evidence only.  `E02` remains active and neural
training remains blocked.
