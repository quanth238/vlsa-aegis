# 0021 — Freeze the generated-source canary and direct transfer pilot

Status: accepted on 2026-07-14 before opening any generated-state or policy
outcome. The activated config SHA-256 is
`ccdccd9465706dd63607afcfa4fb6834168162fe9f0c1f0299a68b26cff46f4f`.

## Context

The released task-0 file cannot provide new independent groups: all 50 saved
states have already been used in development or evaluation, and the passed R00
population has no rows in the inclusive 0--10 mm band around the 5 mm safety
margin. Repeated policy seeds and replay-derived branches do not create new
state groups. A learned-probe result on those rows would therefore be
selection-biased or split-leaking.

The repository does not contain the benchmark authors' Level-II generator.
Plain BDDL reset instantiates all six obstacles, so a custom source must be
declared as a new estimand rather than mislabeled Level II. The direct
scientific question remains unchanged: can a learned direction improve paired
Safe-Progress Success, and is its benefit larger and faster than strong
non-learned and system baselines?

Independent review found and closed four threats before activation: the
high-level reset silently retried randomization failures; flattened MuJoCo
state omitted model-level randomized geometry; absolute asset paths leaked
into identity; and unexpected fields could encode outcome-conditioned
selection. The final generator uses one underlying reset attempt, complete
portable branch identity, exact keys, structured rejection, and two fresh-load
replays.

## Decision

1. Define the additive population as
   `task0_single_obstacle_generated_v1`, with `safety_level="generated"`.
   It is not official SafeLIBERO Level II.
2. Freeze ten outcome-blind generation requests with unique reset and
   placement seeds and a balanced five-obstacle schedule. A failed request is
   retained as a structured rejection and is never redrawn.
3. Derive the canonical `gsrc-*` identity from a portable branch digest that
   binds BDDL bytes, finalized model XML, semantic asset bytes/locators, final
   flattened state, observation, and simulator geometry. Retain the full state,
   branch, and bundle hashes separately.
4. Record raw reset, named-joint edits, pre-settle state, all 20 dummy-control
   states, final observation/geometry, allocation/code/RNG provenance, and two
   exact fresh-load replay proofs. No policy, checkpoint, training, safety
   outcome, planner, or probe is loaded during source generation.
5. Stage request 0 as an allocation canary on MIG/EGL. Submit an independent
   dependency-backed CPU validator against the exact source Slurm identifiers.
   Requests 1--9 may run only after both pass. This canary resolves real EGL,
   finalized-XML asset, and exact-render replay behavior; it is not another
   sampler or steering gate.
6. The ten groups are retired design-pilot inputs only. They may estimate
   valid-state, nominal-collision, boundary, unsafe, and direct-witness
   prevalence for production power planning, but may never enter probe
   training, calibration, validation, locked testing, or a research claim.
7. After accepted bundles are frozen into a v2 manifest before policy outcomes,
   run the existing R01 direct-witness ladder and R02 distributed-oracle,
   equal-L2 random, and registered analytic arms. Keep all scheduled groups in
   the intention-to-test denominator and retain invalid, safe, infeasible, and
   failed rows explicitly.
8. If the privileged distributed oracle fails to transfer on
   simulator-verified direct-feasible generated groups, stop R04 without
   training a probe. If it transfers, estimate production counts and only then
   collect support-matched perturbation labels on disjoint groups.

## Consequences

The source canary cannot demonstrate probe value, collision reduction, or
novelty. Its only success condition is a reproducible independent source
branch. The following retired policy pilot is descriptive and cannot mark
R01--R03 passing.

No sampler parity, R04A, or R04B job is repeated. The direct efficacy
comparison remains frozen for later untouched groups: frozen pi0.5, one locked
ECG arm, realized-final-norm random, a strong time-structured margin analytic
arm, AEGIS, the closest reproducible constrained-flow optimizer, privileged
direction, and direct planner, paired by state, observation, noise, latent, and
horizon with Safe-Progress Success and end-to-end latency as outcomes.
