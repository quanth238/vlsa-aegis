# ADR-0065: Use Codex-frozen semantic labels for the AEGIS diagnostic

## Status

Accepted by explicit user direction on 2026-07-17; supersedes only the
GLM-4.5V selector requirement in ADR-0064.

## Decision

The public VLSA/AEGIS repository cannot run its semantic obstacle selector as
shipped because `main/utils.py` contains an empty ZhipuAI key.  The user has
explicitly directed Codex to perform that setup instead of configuring a GLM
API credential.

R06 therefore becomes a two-stage diagnostic:

1. an allocation-backed capture reconstructs the registered branch, confirms
   the frozen $\pi_{0.5}$ collision, and saves the same-state 1024-pixel AEGIS
   perception views;
2. Codex inspects the captured image and instruction, chooses one public
   obstacle phrase, and freezes it in a content-addressed label record before
   any AEGIS outcome is observed.

The label record must bind the case ID, instruction, exact image hash, allowed
public vocabulary, reviewer `codex`, and timestamp.  It must not contain or be
derived from the simulator obstacle name, object pose, collision geom, or
privileged geometry.  Changing a label after GroundingDINO or simulator
outcomes are observed is forbidden.

GroundingDINO, two-view point-cloud construction, public filtering/MVEE, the
literal full nine-variable CBF-QP, stale pre-settle first-step robot geometry,
and all paired simulator measurements remain unchanged from ADR-0064.  The
official GroundingDINO config and checkpoint must be content-bound before the
full canary runs.

## Naming and claim boundary

The evaluated arm is
`pi05_plus_aegis_codex_label`, not original end-to-end VLSA/AEGIS.  It answers:

> Given a Codex-selected semantic obstacle phrase, can the public AEGIS
> GroundingDINO/geometry/QP safety layer prevent these frozen collisions while
> preserving progress?

It cannot establish the accuracy or efficacy of the unavailable GLM selector,
and it cannot be reported as the original end-to-end public method.  The
collision-conditioned, 17-plus-3 stratum, no-probe, and no-general-benchmark
claim limits from ADR-0064 remain in force.

## Staged release

The first release may capture only manifest row 0,
`crfs-1069f29a8d76463a`.  After Codex freezes its label, one paired canary may
run.  The 20-case population is not automatic.  It requires a valid paired
canary, a separate 20-case image capture, a complete immutable 20-label
manifest frozen before AEGIS execution, and a separate population release.
