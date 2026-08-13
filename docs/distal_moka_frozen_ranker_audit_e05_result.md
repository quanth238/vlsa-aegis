# E05 Moka frozen ranker audit result

## Verdict

Strict **NO-GO** for the preregistered avoidance-ranking gate. The frozen
direction-conditioned model performs significantly better than random sign
selection, but its accuracy is below the required level and its selected
branches do not improve mean near-active L5 clearance.

This result refines the supervisor question:

- paired directional supervision contains a statistically real relative-
  ordering signal at this one state;
- the current frozen model does not convert that signal into reliably
  positive avoidance;
- the result does not support a continuous potential, QP gradient, or robust
  Best-of-N controller.

## Frozen H100 result

Producer job `39319` completed on `worker-2` in `217.824 s` and replayed one
nominal plus 64 positive/negative cloned-OSC pairs (`129` rollouts). All test
directions were disjoint from the 32 development directions up to sign.
Independent H100 validator `39324` reproduced every registered metric and
hash.

| Primary metric | Result | Required | Pass |
|---|---:|---:|:---:|
| Safer-branch choices | `49/64` | at least `55/64` | No |
| Branch accuracy | `0.765625` | `>=0.85` | No |
| Wrong-direction rate | `0.234375` | diagnostic | — |
| One-sided binomial p vs 50% | `1.218e-5` | `<0.05` | Yes |
| Near-active L5 branch accuracy | `0.765625` | diagnostic | — |
| Mean selected near-active L5 gain | `-0.002563 mm` | `>0 mm` | No |
| Positive-gain rate | `0.515625` | diagnostic | — |

The random-sign baseline's mean gain was `-0.046616 mm`, so the model is
better than random on average but still not beneficial in absolute mean. The
largest gain available among the selected branches was only `+0.093389 mm`;
the worst was `-0.161786 mm`.

## Best-of-N observation

The model's top candidate was exact-best for `N=16`, `32`, and `64`. At
`N=64` it also matched the exact best of all 128 signed physical branches,
with zero ranking regret. This is encouraging evidence that its score may be
useful for proposal ranking.

It is not an avoidance success. The exact best branch still had margin
`-58.178324 mm`, compared with the nominal `-58.271713 mm`: only
`+0.093389 mm` improvement and still deeply proxy-unsafe. At `N=4/8`, the
predicted top candidate ranked only `2/4` and `4/8`. One exact top-1 hit at
one state is a diagnostic, not evidence of robust Best-of-N selection.

## Scientific interpretation

The original `7/8` result was not pure chance: the larger frozen test remains
highly significant against 50%. But it overestimated reliability. At 76.6%
accuracy, nearly one in four corrections chooses the physically worse sign.
More importantly, correct pair ordering does not guarantee useful magnitude:
many paired differences are tiny and the mean selected clearance change is
not positive.

Therefore the current learned object should not be used as a safety filter.
For the advisor, the defensible statement is:

> Direction-conditioned paired supervision yields statistically detectable
> local ranking information, but the present model and radius do not produce
> reliable or sufficiently strong avoidance. An integrable scalar potential
> remains unvalidated, while verified Best-of-N ranking remains a hypothesis
> requiring grouped states and candidate families with meaningful safe
> support.

A later ranking experiment should move to grouped physical states and include
structured directions whose exact outcomes span meaningful clearance gains.
Simply collecting more near-zero `+/-0.0125` pairs at this state would mostly
improve confidence about a weak effect, not demonstrate collision avoidance.

## Provenance

- Code commit: `5ee79369593c49180c0841c6a0201efc097b825a`
- Producer: Slurm `39319`, `worker-2`
- Validator: Slurm `39324`, `worker-2`
- Result file SHA-256: `c841ea3cbae7cc8cb02e286ec1dfa2fc94c85cb106d6d263fe956ac5a30e4f03`
- Result payload SHA-256: `8b97733bba6491563cb1b92ad48b613704fd38e4f35ebffbe8ab5e49031333e9`
- Validation file SHA-256: `caa06f768357493c97139c77325dce881bdfd9736dd1b2be336ccb1eb3e379d0`
- Validation payload SHA-256: `fb4c67621bd3c68d7f580865afd07a1b7378728dd5f778cc28eabf35cff94ba3`

No model was trained, no QP was solved, and no selected correction executed.
