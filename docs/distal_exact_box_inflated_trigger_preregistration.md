# Exact-box 8 mm rounded-shell first-crossing test

Status: preregistered before H100 outcome inspection on 2026-08-09.

## Motivation

Jobs `37175` and `37180` activated at action 192, after registered proxy rows
were already unsafe. Enlarging a primitive at that same state can only worsen
infeasibility. This follow-up tests the useful heuristic: enlarge every exact
MuJoCo obstacle box by an isotropic warning shell and trigger before the
nominal transition crosses that shell.

## Geometry and trigger

Every one of the 15 exact live oriented moka-pot boxes is Minkowski-summed
with a sphere of radius `8 mm`. Its support function is

\[
\rho_{B\oplus S_{0.008}}(n)=\rho_B(n)+0.008,
\qquad h_{\mathrm{inflated}}=h_{\mathrm{exact}}-0.008.
\]

This is a rounded box, not an `8 mm` increase to every half-axis. It therefore
adds exactly `8 mm` in every direction without diagonal overinflation.

The immutable job-`37109` action ledger is replayed from step 0. At each state,
one exact cloned OSC transition evaluates the nominal action. The selected
trigger is deterministically the first step where:

1. all eight current exact clearances are at least `8 mm`; and
2. at least one nominal minimum-substep exact clearance is below `8 mm`.

All earlier actions execute exactly as recorded and must reproduce their
cloned dynamic-state hashes. No trigger step is chosen after inspecting the
scan.

## QP test

At the selected state, the existing 87-candidate, 60-candidate-trust-region
oracle-affine apparatus is reused with `clearance_target_m=0.008`. Robot
L5--L7 slabs, released EE proxy, OSC controller, action bounds, affine error
calibration, and QP tolerances are unchanged. The QP proposal is executed only
in the synchronized clone and must:

- have zero raw L5--L7 contact;
- move the obstacle by at most `0.1 mm`; and
- keep all eight minimum-substep exact clearances at least `8 mm`.

The test also requires a jointly raw/inflated-proxy-safe registered candidate
and zero calibrated affine candidate false-safes.

This is a single-transition privileged oracle test. Even a pass does not show
closed-loop task completion, deployable perception, population efficacy, or a
safety certificate. It only tests whether earlier activation makes the
inflated exact-box QP mechanism feasible at the first predicted crossing.

Frozen config SHA-256:
`b47e78ab32aff8587f5f001ab63c27301134d1ada9d92e5abb109ae63a26c5df`.
