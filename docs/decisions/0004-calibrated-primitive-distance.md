# 0004 — Use the calibrated primitive metric as D_sim

Status: accepted after H03 on 2026-07-14.

## Context

The initial runner used `mj_geomDistance` between Panda gripper contact meshes and the active obstacle's box geoms. Job 27091 measured -5.63 cm for the collision-enabled `gripper0_hand_collision` / `moka_pot_obstacle_1_g7` pair while MuJoCo reported no pair contact. The same failure occurred in 16 of the 50 strict H03 states. This is consistent with the upstream open distance-query bug for non-plane and box geometries.

Equation (3) in `main.tex` already defines the controlled pilot as a conservative EEF sphere against known obstacle geometry. Job 27096 separately calibrated sphere--box distance around the zero-clearance boundary against MuJoCo contacts.

## Decision

`D_sim` for the controlled pilot is the analytic EEF-sphere-to-oriented-box signed distance computed from MuJoCo's physics-substep site and geom transforms. It is evaluated across every collision box in the active obstacle union. Physical EEF--obstacle contacts are retained as a one-way conservatism check: contact with positive proxy clearance is forbidden, while a negative conservative clearance before physical contact is expected.

Raw mesh--box `mj_geomDistance` remains in every audit artifact but is advisory only. It cannot drive projection, labels, safety rates, or gate decisions.

## Consequences

- H03 matches the proposal's differentiable primitive geometry and is reproducible on 50 unique saved states.
- H04 must still establish independence between the optimization model and held-out simulator trajectories.
- The 6 cm sphere is a controlled-pilot abstraction, not a full-arm or deployment safety guarantee.
