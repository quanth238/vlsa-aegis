# E05 paired directional contact model result

Clean H100 job `37463` produced a partial improvement but failed the complete
unseen-state gate.

The new objective successfully fit the registered finite pairs:

- train informative-pair accuracy: `28/28`;
- validation informative-pair accuracy: `8/8`;
- held-out test informative-pair accuracy: `30/30`.

At radius `0.1`, the revised gradient was then compared with the same 256
random directions using fresh cloned OSC transitions and the registered raw
safety/contact-count/penetration ordering:

| State | Revised contact count | Random at least as good | Add-one value | Gate |
| --- | ---: | ---: | ---: | --- |
| 187 | 2 | 1 / 256 | 0.00778 | pass |
| 190 | 10 | 41 / 256 | 0.16342 | fail |

The negative sign was better than the positive sign at both states, but both
negative-gradient actions still contacted. Test contact-risk AUC improved from
`0.698` to `0.739`, while false-safe actions changed only from 52 to 51. The
training labels still contain 124 L5 positives and zero L6/L7 positives.

## Interpretation

Paired contact-count supervision materially improved the direction at state
187 and reduced state-190 burden, but it did not generalize into an exceptional
continuous gradient at both unseen states. Perfect ordering on 30 held-out
finite pairs is therefore not enough: ordering sparse endpoints does not
constrain the derivative at the nominal action strongly enough.

Per the preregistered decision, this is `mechanism_go=false`. Stop before
closed-loop testing. A future experiment would need dense small-radius
directional/gradient supervision or a continuous execution-burden target,
rather than another weight or epoch adjustment to this finite-pair loss.

Evidence:

- Model SHA-256: `c477a9af90b9bd1498492b90ac597890f83c1183505e6bddcdb7adcad546fea3`
- Result SHA-256: `cd186104aa245767cc346bb16b5e62796612fa946c7037cccf1f2cb01a4f35b4`
- Validation SHA-256: `aed056699ce110d1e2598e5408f00df29c95923a11ed268e681280ee5b5e8df7`
- Result payload SHA-256: `588887035abbda58d867d752cc49112092db55b0003df8e7c0dda92dca1479d8`
- Local artifacts:
  `/Users/quanth238/personal/Research/probe_vla/output/vlsa_distal_contact_directional_ranker_e05/directional-ranker-e05-20260810a/`

