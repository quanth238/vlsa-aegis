# Distal ellipsoid/contact boundary audit result

## Verdict

**PASS as a conservative observed-contact proxy; not an exact physical
clearance.**

Clean H100 job `37879` completed on `worker-2` in `00:00:03`; the independent
validator accepted the result. Across all 13,025 paired two-action cloned-OSC
rollouts at the zero ellipsoid threshold:

- 466/466 protected MuJoCo-contact actions were rejected;
- no contact action was accepted (zero observed false-safes);
- 10,648 contact-free actions were accepted;
- 1,911 contact-free actions were rejected (false-unsafes);
- contact-free safe-action recall was 84.78%.

Inside the fixed absolute 5 mm proxy-boundary band, all 301 contact actions
were rejected, but 1,413 contact-free actions were also rejected. Therefore the
current ellipsoids are conservative relative to raw contact. They can reduce
safe support, but the observed population does not show them hiding L5--L7
contact.

The frozen threshold sweep is diagnostic only. A -2 mm threshold still had
zero false-safes and increased contact-free recall to 90.53%, whereas -5 mm
accepted 301 contact actions. These values were observed on the audit
population and must not be converted into a tuned deployment threshold.

The unseen test split contained no raw-contact candidate actions. Its zero
false-safe count is therefore vacuous for contact detection; training and
validation supplied all 466 contact examples. A separate initial-state audit
is authorized, but MLP training, QP execution, and closed-loop E05 remain
blocked.

## Interpretation

The proxy geometry can contribute conservative rejection, but it is not the
source of an observed unsafe label in this population. The earlier
action-conditioned MLP's false-safe predictions against the ellipsoid target
remain a model/calibration failure; replacing the target with binary contact
would also remove the smooth margin needed for steering.

## Immutable evidence

- source commit: `d78d4e740b0deaffa2aca079dbd88faf81371032`;
- run root:
  `/mnt/data/quanth/experiments/vlsa-distal-proxy-contact-boundary-audit/proxy-contact-boundary-20260810a`;
- result SHA-256:
  `6d9d5e1b68d21542f4311d19d45b3256c9882745119c17afb003a1a670867def`;
- result payload SHA-256:
  `5dad22090eb0c830b5e086b95b55a11945a7832a3dd77e00530298453d8c5617`;
- validation SHA-256:
  `d741b4b24949ab03d18f82512c435710093ce755c234185cc86a119f95fc6309`;
- preflight SHA-256:
  `dfe5dd154e583951f928c7798c8c950d3544ec47b7223e597eb5f911cedde44b`.
