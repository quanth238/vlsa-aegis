# 0006 — Test endpoint-free reach feasibility before ECG

Status: accepted for the next experiment on 2026-07-14.

## Context

H05 proved that the registered zero-sum correction cannot repair any of the 20 frozen cases because it preserves an endpoint that already violates the safety margin. That result rejects exact local endpoint equivalence; it does not establish that no safe task-progressing action exists or that the frozen policy flow is unsteerable.

The proposed Early Clearance Guidance formulation removes endpoint equality and replaces it with task progress. Its draft describes a transport phase after stable grasp, but the frozen H05 population contains initial task states queried immediately after settling. For SafeLIBERO Spatial task 0, these are pre-grasp reach states. They cannot provide transport evidence, and there is no preceding replanning state for the proposed fallback.

## Decision

Preserve H05 and start a new opt-in gate sequence.

For the same-case diagnostic, define the branch-start target as the position of `akita_black_bowl_1` and measure committed-prefix reach progress as

\[
\Delta\Phi_{\mathrm{reach}}
=
\lVert p_{\mathrm{eef},0}-g_0\rVert_2
-
\lVert p_{\mathrm{eef},5}-g_0\rVert_2.
\]

The target remains fixed at its branch-start position so moving the bowl cannot manufacture progress. Safety and progress are evaluated over the five actions actually executed before replanning. The ten-action policy output remains recorded as model context but its discarded tail does not define the primary label.

R00 first freezes `p_min` as the lower quartile of positive baseline reach progress among simulator-verified non-colliding calibration chunks. Calibration groups remain disjoint from the frozen 20 evaluation groups; if the 30 available groups do not yield 50 valid chunks, additional fixed policy-noise seeds may be used only within those calibration groups, with episode grouping retained and every exclusion reported.

R01 then searches all five translational commands without a zero-sum or endpoint constraint. `D_opt` from the frozen H04 model is only a search signal. A case counts only when paired direct MuJoCo replays verify all of:

- conservative `D_sim >= 5 mm` over the branch point and all 125 physics substeps;
- `DeltaPhi_reach >= p_min`;
- declared action bounds;
- no forbidden EEF--obstacle contact;
- no target-bowl or active-obstacle displacement beyond the registered tolerance.

Report safe witnesses at both `p = 0` and `p = p_min`. A finite optimizer miss is `not_found_within_budget`, never an infeasibility certificate. Keep model-feasible, simulator-verified, proxy-false-safe, proxy-false-negative, safe-without-progress, and invalid-phase outcomes separate.

If at least 12 of the 20 frozen cases have verified safe-progress witnesses, advance to an endpoint-free oracle flow-steerability test. Compare the exact planner direction and an analytic geometry direction with equal-norm random guidance under identical state, observation, policy noise, and execution horizon. Do not train a probe before the oracle intervention and grouped oracle analysis gates pass.

A separate transport study requires newly collected, immutable post-grasp branch states and cannot reuse these 20 identities as transport cases.

## Consequences

- The revised idea is tested against the actual H05 failure population without pretending those states are transport states.
- Exact physical feasibility and local flow reachability remain separate questions.
- Progress loss due to a necessary lateral detour can be diagnosed by comparing `p = 0` with the calibrated threshold.
- H04 extrapolation is recorded for any action component outside its calibrated `+/-0.15` range; every candidate is still judged by `D_sim`.
- A planner failure alone cannot terminate the broader research direction.
