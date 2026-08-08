# E05 oracle-affine mechanism result

## Verdict

**NO-GO for training a Physics NN to predict the current AEGIS proxy
clearances.**  The exact OSC transition was not the decisive error in this
state.  The frozen AEGIS obstacle MVEE did not cover the observed moka-pot/L6
contact points, so the eight-row barrier supplied a false-safe label even when
evaluated at every internal MuJoCo step.

This does not reject Physics-informed learning in general.  It rejects the
current learning target.  The next target must use authoritative obstacle
geometry or raw collision-distance/risk evidence.

## Registered H100 evidence

- Job: `37137`, worker-1, one H100, 8 CPUs, 64 GiB, 41 seconds.
- Source: clean commit `73682dc5f649e57ec0f8f4103f5f0975ddd32278`.
- State/action: immutable job-`37109` actions 0--191 and action 192.
- Candidate set: 87 total; 60 inside the registered `L_infinity <= 0.5`
  affine trust region.
- Nominal raw result: 15 L6 contact substeps and `1.557 mm` maximum obstacle
  motion.
- Proxy result: all eight minimum-substep gaps positive; minimum `1.725 mm`.
- Contact coverage: L6 robot-slab quadratic `0.903--0.910` (inside), frozen
  obstacle-MVEE quadratic `1.247--1.268` (outside).  This audit does
  not separate initial point-cloud under-coverage from proxy-pose staleness.
- Support gap during contact: falsely positive `5.763--6.679 mm`.
- Local control authority: five candidates were both raw-safe and proxy-safe.
- Oracle affine fit: zero registered candidate false-safes after the frozen
  one-sided calibration; maximum per-row absolute fit error `0.290 mm`.
- QP: valid, `1.257 ms`, XYZ correction L2 `1.416e-9`; exact execution repeated
  all 15 contacts.

## Interpretation

The network proposed in the reviewed direction would learn
`action -> proxy minimum clearance`.  Here that quantity is positive for the
colliding action because the frozen obstacle proxy excludes the observed
contact points.  A more accurate transition model, a neural QP approximation,
or additional
L5--L7 robot ellipsoids cannot make an absent obstacle surface appear in the
constraint.

The fastest valid next experiment is:

1. bound the active obstacle collision mesh with a conservative union of tight
   primitives, keeping the accepted L5--L7 slabs unchanged;
2. replace the single obstacle-MVEE pair rows with the corresponding minimum
   pair clearances;
3. rerun this exact frozen oracle audit; and
4. train only if contact implies a nonpositive proxy gap and an exact QP action
   is raw-safe.

An alternative learned direction is to predict raw signed collision distance
or a conservative raw-contact residual, rather than reproduce the released
AEGIS MVEE support gap.

## Artifact identities

- `result.json` file SHA-256:
  `6a5f9842f04b57a42b3232aab6cc135da4ddfa8584bf869a46918c6d98fdbc58`.
- Canonical result payload SHA-256:
  `39001fcf5d3d686f5fc5be0527a7f4133ac0dc040b4268e770f326ee0cf7a749`.
- `validation.json` file SHA-256:
  `1c710822bdb19429d34ababf6c2dfe00315a7f09ae1abc7fe43d44df6f1adbbb`.
- `evaluation-preflight.txt` file SHA-256:
  `a89543cd2a54c4907eb567b1be995d2e97a261c47e962eba9e2866d050d57ef5`.

The claim is a one-state mechanism diagnosis, not population safety, policy
efficacy, neural generalization, or a safety certificate.  Gate `E02` remains
active and `E03` training remains blocked.
