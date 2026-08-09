# E05 contact-gradient implementation audit result

Clean H100 job `37441` found no sign, normalization, or XYZ-indexing fault in
the immutable job-`37416` model.

- `R(u-epsilon*g_hat) < R(u+epsilon*g_hat)` for all 10 state/radius pairs.
- All feature/action indexing receipts passed.
- At `epsilon=0.0001`, central-difference/autograd relative error was
  `1.31e-9` at state 187 and `9.81e-10` at state 190.
- Cloned OSC preferred the negative-gradient sign in all 10 pairs and never
  preferred the positive sign or tied under the registered contact burden.

The simulator preference was only local improvement. Every tested command up
to radius `0.1` still contacted the obstacle. At state 187, the negative sign
reduced contact events from 5 to 2 relative to the positive sign at radius
`0.1`; at state 190 it reduced them from 16 to 11. Thus the implementation is
coherent, but this does not overturn job `37427`: many random directions still
reach safety as early or earlier at the larger radii required for recovery.

The correct next model test is therefore paired directional supervision, not
an implementation repair and not merely more pointwise classification data.
Closed-loop E05 remains unauthorized.

Evidence:

- Result SHA-256: `63483c4cf87bcbd3b64349ce6bf5077fe8a34df958e8451d77445beea7d88b88`
- Validation SHA-256: `2d74f1faead6b39cf52f18c9b4c0d115e355ce155c689c67142237b21f15639c`
- Result payload SHA-256: `da036c04f6ba8a7fe26c8b862d8deb1f16331e6acbc123bfbd012e1a27eaee57`
- Local artifacts:
  `/Users/quanth238/personal/Research/probe_vla/output/vlsa_distal_contact_gradient_sign_audit_e05/gradient-sign-audit-e05-20260810a/`

