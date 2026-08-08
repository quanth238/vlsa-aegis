# Distal-only 8 mm closed-loop v3 preregistration

V3 implements “retain original EE” literally. Live pi0.5 actions first pass
through the unchanged released AEGIS EE QP. The new simulator-in-the-loop
filter then imposes only seven hard minimum-substep targets: `8 mm` on the
three L5, two L6, and two L7 slabs against the exact 15 live MuJoCo boxes.

The original EE proxy/MVEE gap remains recorded as a diagnostic, but no second
discrete EE target is added. All v2 policy seeds, five-action replanning,
candidate set, finite differences, QP tolerances, exact clone checks, raw
L5--L7 contact veto, `0.1 mm` within-step obstacle-motion veto, registered
horizon, and native task criterion remain unchanged.

Passing still requires task completion with no protected contact, no paper
CAR, and exact clone/main agreement. This is a single-case privileged
feasibility test, not a deployable, population, whole-arm, or formal result.
