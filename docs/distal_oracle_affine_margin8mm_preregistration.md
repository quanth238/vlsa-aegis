# Fixed 8 mm clearance-margin intervention test

Status: preregistered before H100 outcome inspection on 2026-08-09.

This is the first test in the user-specified order. It retains the immutable
E05 action-192 simulator state from job `37109`, the accepted seven L5--L7
slabs, the released AEGIS end-effector ellipsoid, and the frozen released
AEGIS obstacle MVEE. It does not use the later MuJoCo obstacle primitives.

The sole scientific intervention is

\[
h_{\mathrm{corrected}}=h_{\mathrm{AEGIS}}-0.008\ \mathrm{m}.
\]

Requiring `h_corrected >= 0` is implemented exactly as an unmodified proxy
clearance target `h_AEGIS >= 0.008 m` for all eight QP rows. The candidate
set, OSC transition, 60-candidate affine fit, one-sided `1e-6 m` calibration,
QP tolerances, action bounds, and complete internal-substep verification are
unchanged from the base oracle-affine protocol.

The registered question is narrow: does the valid eight-row QP produce an
action whose exact cloned OSC transition has (1) no raw L5--L7 contact, (2) no
more than `0.1 mm` obstacle displacement, and (3) all eight frozen-proxy
minimum-substep clearances at least `8 mm`? The test passes only if all three
conditions hold. A pass demonstrates that a fixed early-warning margin can
activate the existing QP at this single state. It does not repair or validate
the known incomplete obstacle MVEE, prove trajectory safety, or establish a
deployable margin.

After this outcome is frozen, the second test replaces only the obstacle MVEE
with the exact collision geometry already compiled inside MuJoCo. That later
simulator-geometry test is privileged feasibility evidence and must not be
interpreted as deployable perception.

Immutable inputs:

- Table 1 remains read-only.
- False-safe ledger: job `37109`, action 192, file SHA-256
  `d79a28585e74cece727fbc3d3e6a72eb1647a782cae4962ec3af9045f448e603`.
- Base settings are identical to
  `configs/vlsa_distal_oracle_affine_e05.v1.json` except schema, protocol,
  claim scope, and the fixed clearance target.
- Registered config:
  `configs/vlsa_distal_oracle_affine_margin8mm_e05.v1.json`.

No setting may be changed after inspecting the H100 outcome.
