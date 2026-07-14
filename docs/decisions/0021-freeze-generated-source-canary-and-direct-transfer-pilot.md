# 0021 — Freeze the generated-source canary and direct transfer pilot

Status: accepted on 2026-07-14 before opening any generated-state or policy
outcome. The activated config SHA-256 is
`332c90fdb9e4560ab522e6333fbe5f0846d1c7caca9190f1730436036856f22a`.

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
5. Stage request 0 as an allocation canary using the baseline CPU OSMesa
   renderer inside a MIG allocation. Submit an independent
   dependency-backed CPU validator against the exact source Slurm identifiers.
   Requests 1--9 may run only after both pass. This canary resolves real
   offscreen rendering, finalized-XML asset, and exact-render replay behavior; it is not another
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

The first source task, `27578_0`, retired the preregistered EGL transport before
any reset or scientific outcome because robosuite attempted to parse the MIG
UUID as an integer. Its rejection is retained. Freezing OSMesa aligns this
source-only path with the already used SafeLIBERO baseline renderer; no state
seed, placement, or selection rule changed.

The first OSMesa task, `27584_0`, performed its one seeded reset and raw-state
read, then exposed a portability bug before obstacle edits, settling, source
acceptance, or policy/safety outcomes: the runtime's outer `libero` directory
is intentionally a namespace package, so `libero.__file__` is `None`.
Portable assets are now located relative to the concrete `libero.libero`
package. This does not change the frozen config, state seeds, placements,
acceptance rule, or any policy outcome.

The asset-fixed task, `27592_0`, then completed its original edit/settle branch
but rejected on fresh-load settle step 1. The installed robosuite flattened
state contains only time, qpos, and qvel; it omits MuJoCo integration fields
that affect the exact next transition, including solver warmstart and user
inputs. The generated-source identity and replay contract therefore bind
MuJoCo's full `mjSTATE_INTEGRATION` state before settle, after every settle
step, and at the final branch. The failed request remains retained and is not
redrawn; this correction was made without opening a policy, safety, planner,
progress, or probe outcome.

A second independent pre-submission audit found a schema/semantic mismatch:
the semantic validator discarded unexpected settle-record keys before calling
the exact array validator, and it did not enforce the declared one-reset API
string. Those paths now fail closed for both flattened and integration-state
histories, boolean/non-integer step indices, the reset API, and nested geometry
array records. Positive accepted-fixture, recomputed-hash outcome-field,
integration/proof/MJB binding, JSON-Schema, and exact restore-call-order tests
cover the correction. This validation fix changes no source request, seed,
placement, acceptance criterion, or scientific config hash.

The same audit then exercised fully rehashed, cross-rebound records rather than
stopping at local hashes. The final contract requires the seven ordered source
joints to agree with the named edit log, requires the active edit to reuse the
matching parked joint, freezes a nonempty sorted typed active-geom record in
both semantic validation and JSON Schema, and binds compiled-MJB byte count as
well as SHA-256 into both fresh-replay proofs. MJB diagnostics remain excluded
from the portable scientific source identity. After the relational joint-name
mutation also failed closed, independent review returned GO for a fresh
request-0 canary; the frozen source config and all request seeds remain
unchanged.

Allocation `27604_0` then showed that the compiled-MJB premise was itself
invalid: two fresh compilations of the exact finalized XML/assets under the
same build produced equal-size MJB files with different raw hashes, before the
full-state replay was attempted. Raw MJB bytes are therefore removed from the
artifact, proofs, schema, validator, and runtime gate rather than retained as
an uninterpretable diagnostic. The scientific model input remains the exact
portable finalized XML plus hashed asset bytes. The functional compile check
is stronger and directly relevant: two fresh loads must reproduce all 20
flattened and `mjSTATE_INTEGRATION` transitions byte-for-byte, then match final
observation and geometry. The failed allocation is retained, no request is
redrawn, and independent review returned GO for a new immutable canary ID with
the same config and seeds.

No sampler parity, R04A, or R04B job is repeated. The direct efficacy
comparison remains frozen for later untouched groups: frozen pi0.5, one locked
ECG arm, realized-final-norm random, a strong time-structured margin analytic
arm, AEGIS, the closest reproducible constrained-flow optimizer, privileged
direction, and direct planner, paired by state, observation, noise, latent, and
horizon with Safe-Progress Success and end-to-end latency as outcomes.
