# E05 Cartesian controllability and detour-manifold oracle result

## Verdict

The controllability audit **passes**, but the registered task-relative
five-action manifold is a strict **NO-GO** for selector training.  Cartesian
commands affect the endangered links early and strongly enough, and the
archived continuation contains genuine MuJoCo contact.  Nevertheless, neither
continuous arm nor any of the 54 fixed candidates satisfies physical/proxy
safety and task compatibility together.

This distinguishes the failure from the frozen ranker's earlier `49/64`
result: better ranking cannot solve the current candidate-support problem.

## Controllability audit

At the measured state before archived action 182:

| measurement | result |
|---|---:|
| compiled-box Loewner proxy at initial state | `+17.229 mm` |
| initial exact robot-ellipsoid/compiled-box overlap | no |
| initial protected MuJoCo contact | `0` |
| first command-induced L5--L7 center change | internal sample `2` |
| first Loewner-proxy violation | internal sample `30` |
| first exact robot-ellipsoid/compiled-box overlap | internal sample `132` |
| maximum pre-overlap link-center change from basis probes | `14.244 mm` |
| nominal exact-overlap samples | `311` |
| nominal protected-contact samples | `170` |
| nominal minimum Loewner proxy | `-58.287 mm` |

Thus, intervention is not too late: action changes affect L5--L7 about 130
internal samples before the first exact overlap.  The Cartesian interface has
material authority.  The future violation is also not solely a conservative
proxy artifact because raw protected contact occurs in the nominal rollout.

## Oracle comparison

H100 producer `39333` evaluated 702 cloned-OSC rollouts from clean commit
`eba0d8baff3f94369012166038f084dc7c5f194c`.  It measured 25 internal MuJoCo
substeps per action for the nominal trajectory, ten authority probes, ten
finalists per continuous arm, and every one of the 54 finite-library actions.

| arm | search | raw contact-free | CAR pass | terminal pass | exact ellipsoid/box nonoverlap | full pass |
|---|---:|---:|---:|---:|---:|---:|
| soft/free 5D | 374 boundary evaluations; 10 finalists | `10/10` | `10/10` | `0/10` | `0/10` | `0` |
| endpoint-preserving 4D | 189 boundary evaluations; 10 finalists | `0/10` | `0/10` | `10/10` | `0/10` | `0` |
| frozen library | 54 internally verified candidates | `0/54` | `0/54` | `48/54` | `0/54` | `0` |

The best soft/free finalist used correction norm `1.482` and:

- removed every protected MuJoCo contact;
- passed CAR with negligible obstacle displacement;
- retained task progress `1.117`;
- improved the Loewner proxy from `-58.287` to `-29.860 mm`;
- improved normalized exact-overlap slack from `-0.1061` to `-0.0510`;
- but retained 202 exact robot-ellipsoid/compiled-box overlap samples;
- and ended `20.926 mm` from the nominal terminal EE position, beyond the
  registered `15 mm` task threshold.

The best endpoint-preserving finalist had only `2.496 mm` terminal EE error,
but retained 140 protected-contact samples and displaced the obstacle by
`17.832 mm`, far above the 1 mm CAR limit.  The least-contact finite-library
candidate still had 103 protected-contact samples.

The soft differential-evolution arm exhausted its registered eight
generations rather than converging; therefore the correct claim is that the
**registered bounded search found no support**, not that no coefficient in
the continuous manifold can exist.  The endpoint arm did converge.  All 54
finite candidates were exhaustively internally verified.

## Independent validation

Independent H100 job `39338` recomputed all gates and freshly replayed the
best candidate from each of the two continuous arms and the finite library.
All stored margins, contacts, CAR, task metrics, and pass/fail decisions were
reproduced within `1e-10`.

- Producer result SHA-256:
  `ddc676f775b19cdf507e88359d086b019f8aea2e3191e6a4c0023cb1109c6d21`
- Producer payload SHA-256:
  `3ac21494b469de9a7d60f702e9b6a19f12830ee73e2b2412f309c446496c7e98`
- Validator file SHA-256:
  `086bd244db935905554e92018fea21cb086c0d4b671ef81c76dc1e7566860aac`
- Validator payload SHA-256:
  `d4713cefffed96c1f8de67b8f46dbf1d335e594c20877faa90fad186f99dfa71`

## Scientific interpretation

The current root problem is not perception alone, intervention timing, or
absence of Cartesian authority.  It is the conflict induced by a five-action
detour followed by the unchanged fixed suffix:

1. A strong free detour can eliminate native contact, but it has not escaped
   the conservative L5--L7 ellipsoid/compiled-box intersection and does not
   rejoin the nominal terminal state within five actions.
2. Strict five-action endpoint preservation retains task geometry but removes
   the freedom needed for avoidance.
3. The frozen library is too weak for this state; it contains no contact-free
   candidate.

Do not train the ordinal selector or absolute safety gate yet.  The next
no-learning experiment should change the **candidate horizon/continuation**,
not the network: execute a contact-free soft prefix, reobserve/requery the
frozen VLA (or optimize a longer recovery tail), and check whether it can
restore task compatibility while maintaining internal-substep safety.  Only
policy-consistent safe candidates should become selector training data.
