# Reproduction decisions

## ADR-0113: Discover grouped warning states only at real policy queries

- Status: accepted, coverage audit active; candidate collection and learning blocked
- Date: 2026-08-14

Replace arbitrary action-lead selection with an interface-aligned rule. In
each non-test episode, enumerate every real pi0.5 query boundary before the
first protected contact. Retain all boundaries whose current state is strictly
proxy/physically safe but whose exact next five released-AEGIS commands are
unsafe over internal OSC/MuJoCo substeps. Preserve the policy query hash and
seed, full simulator/controller state identity, episode group, seven row
minima, and the active link/row/time witness.

This is a coverage screen only. It neither searches candidates nor infers that
a retained state is recoverable. After independent aggregation, the unchanged
37-member structured candidate bank and complete fixed backup may run only on
retained states. Each state must then contain both safe and unsafe candidates;
timeouts remain unknown. Before any MLP training, publish per-row state,
episode, unsafe, safe, near-boundary, and active-witness counts. Missing L6/L7
or slab support must narrow the claim or trigger collection from additional
clean episode groups, never synthetic filling. Final generalization uses newly
reserved complete episodes because the existing test cases have influenced
method design.

Allocated-H100 producer array `39629` and independent validator `39650`
retained one real-query warning state in each of 15/15 episodes, so the state
rule is usable. The scientific coverage gate nonetheless fails: only L5 rows
1--2 and L6 row 3 are active/violated. L5 row 0, L6 row 4, and both L7 rows
have no boundary support. Therefore do not train a nominal seven-output MLP.
Proceed only to the preregistered structured candidate-plus-complete-backup
labels at these immutable states; if those counterfactuals still fail to
populate missing rows, restrict the first learned claim to supported rows or
collect new clean episode groups with the missing physical mechanisms.

## ADR-0112: E05 supports direct Monte Carlo risk learning, but only for L5

- Status: accepted, diagnostic gate passing; population learning still blocked
- Date: 2026-08-14

Allocated-H100 producer `39618` and independent validator `39621` establish
that the corrected five-action risk object has a nonvacuous boundary at E05:
2/37 structured candidates are exactly safe, while 35/37 are unsafe or
fail-closed timeouts. A front-loaded positive-normal radius-2 correction moves
the worst margin from nominal `-16.075686 mm` to `+2.193830 mm` and passes raw
contact, CAR, and a ten-action stable hold. Smaller normal radii improve the
margin but do not reach safety. Therefore the action interface has sufficient
local authority and the complete risk target is suitable for a grouped-data
learning test.

Do not train from E05 alone. All 37 active witnesses are the same L5 slab row;
there is no L6/L7 or cross-episode evidence. Preserve this as a diagnostic and
advance only to grouped episode collection with real query-boundary discovery,
the same frozen candidate/backup/terminal semantics, complete episode splits,
and explicit per-row witness coverage. If L6/L7 support is absent, narrow the
claim rather than synthesizing or hiding labels. TD, calibration, QP, and
closed loop remain blocked until the grouped dataset and direct seven-output
prediction gate pass.

## ADR-0111: Put the learned filter after released AEGIS and label complete backup risk

- Status: accepted, diagnostic apparatus implemented
- Date: 2026-08-14

For the first decisive pilot, define the candidate action coordinates as the
five Cartesian commands output by released AEGIS immediately before OSC. This
preserves the competent AEGIS/VLA baseline and tests an additive L5--L7 filter
without feeding a modified command back through the single-EE QP. Bind the
candidate center to archived query index 37 and executed steps 185--189; store
the original policy-query hash and seed as provenance. Later raw-VLA insertion
is a distinct arm and must not be conflated with this diagnostic.

Replace one-action candidate labels with complete five-action Monte Carlo
labels. A candidate executes all five commands, then a deterministic geometry-
only backup replans one action at a time until verified stable release,
physical contact/CAR, or a fixed timeout. Exclude immutable k0 from the
action-dependent prefix minimum while retaining it as strict state
eligibility. Timeout cannot be safe. Preserve separate prefix, backup, and
combined seven-row risks plus raw physical vetoes and exact replay hashes.

Freeze the candidate population before observing outcomes: paired
normal/tangent directions, constant/front-loaded temporal bases, and
0.5/1.0/2.0 L2 radii. The diagnostic asks only whether this correct target has
a nonvacuous learnable boundary at E05. It does not train, use a QP, run closed
loop, claim generalization, or establish a neural CBF.

## ADR-0110: Learn query-aligned five-action Monte Carlo backup risk

- Status: accepted, protocol redesign active
- Date: 2026-08-13

The current research target is the direct seven-output action-risk formulation
in the method/data memo. It conflicts with both the draft's earlier
joint-rollout-plus-Poisson formulation and the implemented one-action-plus-hold
pilot. For the decisive experiment, choose one object: learn exact Monte Carlo
future risk for a complete five-action candidate followed by a complete fixed
backup. Do not claim simultaneously that the MLP predicts joint motion.

Use real pi0.5 query boundaries. Center candidates on the exact five raw VLA
actions actually scheduled from that query, pass every candidate action
through the unchanged released AEGIS and OSC, and preserve nominal gripper
commands. Translation and rotation candidate families must be separate
registered arms because released AEGIS zeroes rotation; mixing them would
change the controller interface without attribution. Keep the complete 5x7
chunk in stored data even when the first pilot corrects translation only.

Select warning states adaptively across query boundaries, not by a fixed action
offset or arbitrary proxy crossing. A retained state must be currently proxy-
and physically-safe, have dangerous archived nominal continuation, and have at
least one exact-safe and one exact-unsafe member in the frozen structured
candidate bank. The bank uses paired signs, fixed radius ladders, smooth time
profiles, and geometry normal/tangents. State/radius search outcomes are part
of the population audit; unsupported episodes are reported rather than
dropped.

After all five candidate actions, execute the same deterministic,
ledger-independent backup repeatedly. Register its hysteresis, direction and
magnitude rule, action bounds, gripper hold, tie breaking, and fallback. Define
`SAFE_TERMINAL` by a release-margin state plus a verified stable hold tail;
define `UNSAFE_CONTACT_OR_CAR` from protected MuJoCo contact or paper CAR; use
`UNKNOWN_TIMEOUT` at the fixed horizon and never label it safe. Store separate
seven-row candidate-prefix, backup, and combined maxima with complete internal
traces and replay hashes.

Train first from these direct Monte Carlo labels; no target network or TD
bootstrapping. Ellipsoid rows are the optimization target, while raw contact
and CAR are separate physical authorities. A proxy-safe/contact-positive
sample fails the geometry/cohort gate. Episode groups are immutable, inspected
episodes are diagnostic, and final tests are newly reserved. Require mixed
support, safe support in every recoverable validation/test state, and active
coverage by link/row; otherwise restrict the scientific claim. No MLP, QP, or
closed-loop action is authorized until the replacement dataset validator
passes.

## ADR-0109: Gate learning on clean task-successful exact action risks

- Status: accepted, dataset collection active
- Date: 2026-08-13

The learned-filter question is restricted to episodes where AEGIS already
demonstrates native task competence but an L5--L7 contact causes CAR failure.
Perception failures, settled collisions, gripper/task-object collisions,
initially unsafe states, and the known E38 proxy mismatch are excluded by
registered evidence rather than post-hoc judgment.  E05/E10 are diagnostic and
cannot support the final generalization claim.

The fixed ledger-independent backup defines the continuation and is recomputed
after each candidate.  Exact labels are seven candidate-plus-backup
worst-future proxy risks; raw protected contact
and CAR remain separate physical acceptance authorities.  Complete episodes
are the split unit.  Dataset support must pass before training.  The first
learned gate is one direct seven-output action-risk MLP without calibration,
QP, or closed-loop execution.  QP remains forbidden until zero observed
false-safes and useful safe-action support are established on untouched cases.

The model implementation may be prepared before collection, but it cannot
execute unless the immutable dataset validator authorizes learning. Use one
fixed 167D compact input and seven-output MLP; validation episodes alone select
the checkpoint, E05/E10 remain diagnostic, and no prediction result authorizes
a QP or closed-loop controller without a later explicit decision.

Canary `39465` rejected two-stage successor cloning because its selected
backup suffix did not reproduce inside the complete proposal-plus-backup
rollout. Preserve the policy and candidate family, but evaluate every composed
branch uninterrupted from the original saved state. Use its successor suffix
for backup selection and the same branch's full trace for the seven labels.
Composition canary `39471` passed this binding, so the apparatus may progress
to a complete four-state diagnostic case. The canary remains non-scientific
and cannot satisfy any dataset or learning gate.

Full E05 job `39474` found 104/104 exact-safe proposal-plus-backup labels at
20/15/10/5 actions before contact. Reject scale-up from this population: it
cannot test false-safes. Shift only the registered warning offsets to 5/3/2/1,
continue to reject unsafe initial states, and require both safe and unsafe
support in every state before learning.
Job `39487` localized mixed support to the three-action lead (E05 step 184:
21 safe, 5 unsafe). Five actions was too early, two actions too late, and one
action initially outside the safety boundary. Use exactly one three-action-
lead state per episode. Since E05 remains diagnostic, this does not inspect or
select on the untouched E42/E44 test outcomes.
Array `39490` confirms the cohort cannot yet authorize training: most milk/L5
states fail initial eligibility, while seven milk/L6 tasks exposed a proposal-
successor binding failure. Do not reinterpret missing artifacts as negative
labels. Derive successor geometry from an uninterrupted composed branch and
rerun an L6 apparatus canary before any dataset retry.

Canary `39512_10` and full grouped H100 array `39516` close that apparatus
question. Every L6 proposal successor now matches across all 25 embedded
backup branches with zero error, and all 18 immutable case artifacts are
complete. Validator `39565` retains a strict dataset NO-GO: seven requested
states are initially unsafe, only one of three validation states is eligible,
and the two surviving test states contain only exact-safe candidates. Training
on this population would be undersupported and testing zero false-safes would
be vacuous. Keep `MLP_training_authorized=false`.

Reject fixed action offset as the population state selector. “Three actions
before contact” is not comparable across L5 and L6 trajectories. The next
permitted change is a no-candidate warning-state audit that defines state
selection from measured clearance and strict initial physical validity using
diagnostic/train/validation episodes only. Because the current E42/E44 cases
have now been inspected, they are diagnostic henceforth; reserve new complete
episodes after freezing the rule. Do not alter the risk target, backup policy,
MLP, or prediction gates, and do not train, add a QP, or run closed loop.

Freeze that audit rule before execution: select exactly two actions before the
first seven-row proxy minimum strictly below `+1 mm`, provided the selected
state itself is at or above `+1 mm`, has zero protected MuJoCo contact, and has
at most `1 mm` active-obstacle L1 displacement. Never skip an earlier crossing
or tune the lead per episode. Audit only diagnostic/train/validation cases and
perform no candidate rollout. This is a state-selection test, not evidence for
action-risk learnability, control, a QP, or a CBF. A selector pass authorizes
only a later exact candidate-support dataset using the unchanged backup.

H100 array `39576` cannot evaluate this decision because it omitted the state
immediately before the contact-causing action (`contact_step`). This is an
action-boundary indexing defect: the archive's contact step labels the action
transition, not the preceding state. Keep the selector fixed, include states
`0..contact_step` inclusive in version 2, and rerun. Do not interpret the five
L6 no-crossing records or train from them. Validator `39594` was canceled
pending the corrected producer.

## ADR-0104: Scale safety learning, not frozen-suffix task claims

Accepted after H100 producer `39384` and validator `39385`. Eight repeated
normal interventions kept every executed prefix raw-contact/CAR-safe with a
+1.037969 mm minimum proxy margin, but the frozen task-successful proposal
ledger no longer completed after state-changing corrections. This validates
the repulsive safe-set mechanism on one E05 counterfactual, not a complete
backup policy. Next data must come from live policy-conditioned continuations
created by the filtered states; do not learn a scalar force from the stale
suffix or increase force magnitude to repair task recovery.

## ADR-0103: An armed backup may execute a verified-safe deadband nominal

Hysteresis is memory, not a mandatory correction. Job `39382` showed a flat,
physically safe +4.496559 mm nominal and six equally safe repulsive candidates
inside the +1 to +5 mm deadband. Keep the backup armed, execute the verified
nominal, and reevaluate next window. Fail closed only below activation or on a
physically unsafe nominal when no safe improving normal candidate exists.

## ADR-0102: Separate current-state eligibility from future candidate value

Job `39380` deadlocked at action 262 because immutable `k=0` was the active
minimum for the nominal and all six candidates. Preserve that sample to reject
an initially colliding prevention state, but exclude it when comparing action
effects. A command cannot improve geometry that precedes its application.

## ADR-0101: Freeze a task-successful policy sample before judging backup efficacy

Job `39376` sampled a safe but task-failing pi0.5 realization, so its zero
repulsion windows cannot test collision avoidance. Register job `39354`'s
task-successful, later-colliding Cartesian proposal ledger as the immutable
counterfactual input. Require exact AEGIS reconstruction before intervention,
then evaluate only the fixed normal backup. Do not attribute stochastic policy
failure to the backup or call the frozen suffix live replanning.

## ADR-0100: Define the backup policy before learning its PNCBF value

Preregister one fixed policy: frozen pi0.5 and original AEGIS, with a
hysteretic five-action L5--L7 warning and only geometry-normal repulsive
residuals. The VLA owns task recovery. Raw protected contact and CAR determine
physical acceptance; ellipsoid clearance activates and ranks the backup but
does not overwrite physical evidence. Train no value model until this policy
first completes E05 safely or yields a clearly localized oracle failure.

## ADR-0099: Use a policy-conditioned recoverable set, not current-state safety

Accepted after H100 producer `39354` and replay `39358`. A single contact-free
repulsive prefix preserved native task completion but the live continuation
returned L5 to contact at action 223 in both runs. Therefore current contact-
free geometry is not a sufficient safe-set label. Approximate a bounded
policy-conditioned recoverable set offline: a warning state is positive only
when a bounded repulsive prefix followed by the realized frozen-policy
continuation is contact-free and reaches either another positive state or the
goal. Use the ellipsoid signals as early warning features and raw MuJoCo
contact/CAR as outcome authorities.

Do not impose five-action endpoint cancellation, enlarge one fixed correction,
or train from isolated local ranking labels. The next oracle may reapply a
short repulsive prefix when a hysteretic future-risk warning activates; the VLA
remains responsible for task progress between interventions.

## ADR-0098: Do not force task rejoining into the repulsive prefix

Execute one already verified raw-contact-free five-action repulsive prefix and
release control to fresh frozen pi0.5 observations. Endpoint preservation and
a task-return penalty are forbidden in the prefix. The VLA is responsible for
task recovery; the experiment contains no repeated distal intervention.
Define the practical learning target, if this passes, as a policy-conditioned
recoverable warning-state set rather than an assumed forward-invariant CBF
set. Raw MuJoCo protected contacts remain the physical authority and
conservative ellipsoid overlap remains separate.

## ADR-0097: Pass controllability; reject selector training without safe support

Accepted after H100 producer `39333` and independent validator `39338`. The
action-182 state is initially safe, a modified command influences L5--L7 at
internal sample 2, exact overlap begins only at sample 132, and registered
basis probes move link centers by up to 14.244 mm beforehand. Raw contact in
the nominal continuation confirms the failure is genuine rather than only a
Loewner-proxy artifact.

Reject the current five-action candidate family as selector supervision. The
soft/free continuous arm removed every protected contact and passed CAR, but
all ten finalists retained exact robot-ellipsoid/compiled-box overlap and
missed the 15 mm terminal bound. Endpoint preservation did the reverse: it
preserved task geometry while retaining contact and excessive obstacle
motion. The fixed 54-candidate library had no contact-free candidate.

Do not train a ranker, gate, safety potential, or QP. The next intervention
must alter the candidate horizon or use a fresh feedback continuation after a
free detour. Because the soft optimizer reached its registered generation
limit, retain only a bounded-search NO-GO; do not claim global infeasibility.

## ADR-0096: Audit controllability before learning a detour selector

Accepted before simulation. The statistically non-random frozen ranker does
not establish avoidance because all 128 tested local branches remained about
58 mm proxy-unsafe. Before collecting or training anything else, separate
initial safety, command timing, Cartesian link authority, proxy conservatism,
and genuine raw contact at the original action-182 state.

Only if that audit passes, search a generic task-relative detour manifold.
Compare a soft/free continuous family, strict endpoint-preserving control arm,
and frozen 54-member finite library under identical state and continuation.
Acceptance uses internal exact overlap, raw contact, CAR and task compatibility;
the conservative Loewner support gap is diagnostic. This distinguishes a bad
manifold from poor discretization before any learned selector is authorized.

## ADR-0095: Reject the frozen ranker as avoidance; retain weak ranking evidence

Accepted after H100 producer `39319` and independent validator `39324`. The
64-direction frozen audit proves the prior `7/8` observation was not solely
chance: branch accuracy is `49/64` with exact one-sided
`p=1.21823e-5`. It nevertheless fails the registered reliability requirement
and selects the physically worse sign `23.44%` of the time.

Do not equate statistically significant ordering with avoidance. Mean
selected near-active L5 gain is `-0.002563 mm`, and the best available branch
remains deeply unsafe. The exact-best top-1 selection at N=64 is retained as a
one-state Best-of-N diagnostic only; it cannot override the failed primary
gate or establish state generalization.

Do not build a QP, potential-gradient controller, or larger same-state dataset
from this checkpoint. Any later ranker must use grouped physical states and
candidate families with meaningful safety spread, then independently test
both pair ordering and exact selected clearance.

## ADR-0094: Freeze the scalar model and test 64 new paired directions

Accepted before simulation. Reconstruct the exact direction-conditioned
checkpoint from validated job `39312`; do not train or alter inputs. At the
same E05 action-182 state, test 64 new seed-`2026081410` directions and both
radius-`0.0125` branches using exact cloned-OSC outcomes.

Judge the paired-ranking hypothesis by branch accuracy, exact binomial
significance against 50%, and near-active L5 gain. Also audit whether ranking
the model-selected branches actually finds high-clearance candidates as N
grows. Do not interpret branch ordering as an integrable potential or control
gradient, and do not solve a QP or execute a selected correction.

## ADR-0093: Reject both scalar control fields; retain paired ordering as a hypothesis

Accepted after H100 producer `39312` and independent validator `39313`.
Neither matched model meets the fresh-direction response gate. The
direction-conditioned scalar has near-active cosine/sign only
`0.609441/0.833333`; the nonlinear absolute-value model is worse at
`0.063869/0.479167`. Do not use either as a gradient, QP row, or online
correction model.

Absolute local value regression is specifically rejected for this pilot. It
selects the safer branch in only `3/8` tests and has `1.224 mm` worst-margin
RMSE. The common margin signal overwhelms the small paired response needed for
steering.

Preserve one narrow positive observation: explicit direction conditioning
selects the safer +/- branch in `7/8` untouched directions. Its absolute
worst-margin RMSE (`0.073 mm`) does not beat a zero-response baseline
(`0.070 mm`), so this supports only a paired-preference hypothesis. Any next
gate should test preference/ranking across grouped physical states against
chance and analytical repulsion. Continuous-gradient, QP, and control claims
remain blocked.

## ADR-0092: Compare scalar secant prediction with nonlinear local action value

Accepted before training. Keep the compact physical input, E05 action-182
state, exact 32 direction vectors, radius `0.0125`, twenty-action horizon,
geometry, seed, optimizer, and model width fixed. Use directions 0--19 for
training, 20--23 for checkpoint selection, and retain the original 24--31 as
the untouched test.

Compare two matched scalar networks. One predicts a link/time witness's
central-secant response conditioned explicitly on the queried direction. The
other predicts its absolute margin conditioned on the signed local action
delta; its secant response is derived from paired predictions. Judge both on
fresh-direction response, nominally near-active witnesses, safer-branch
selection, and worst-margin error. This changes output representation only;
additional state inputs, QP, action correction, and closed loop are forbidden.

## ADR-0091: Reject radius reduction as the repair for the affine teacher

Accepted after H100 producer `39305` and independent validator `39307`.
Both smaller-radius experiments reused the exact original 32 directions and
changed only paired physical perturbation magnitude. Neither the compact MLP
nor independently fitted local ridge reached the registered held-out
cosine/sign gates at `0.025` or `0.0125`.

Do not blame model size or reintroduce the rejected 1,055D input. The compact
model still fits values within `0.652/0.518 mm`, and its six near-active rows
agree strongly with each within-radius ridge fit. The decisive failure is
that the ridge teacher itself reaches only `0.646382/0.664620` held-out cosine
and `0.790179/0.769643` sign accuracy. Its near-active rows also change across
radii (mean cosine `0.764505`).

Retain compact physical inputs, but replace the assumption that one affine
140-row response field transfers across fresh perturbation directions. The
next independent gate may test direction-conditioned scalar secant prediction
or a matched nonlinear local action-value model using immutable paired data.
QP, action correction, and closed-loop execution remain blocked.

## ADR-0090: Change only secant radius after compact-input improvement

Accepted before simulation or training. Keep E05 step 182, the compact 25D
input, exact 32 direction vectors, 20-action horizon, geometry, model, loss,
optimizer, seed, and split of 24 fit/eight held-out directions fixed. Reuse
directions generated under the original radius-0.05 support mask, but execute
paired perturbations at `0.025` and `0.0125`.

Evaluate the local ridge teacher before interpreting the MLP. A smaller radius
supports local affine-response learning only if both teacher and compact MLP
reach the registered held-out cosine/sign gates. Otherwise, do not add state
inputs, a QP, or control; replace the full-vector affine response target in a
later gate.

## ADR-0088: Isolate input representation with one-state memorization

Accepted before training. The frozen audit attributes primary failure to a
1,055D redundant representation that cannot fit its own five training states.
Change only the input: at original training step 182 retain the nominal
first-five XYZ chunk plus time/link witness identity. Keep the existing
128-unit architecture, joint value/paired-response objective, AdamW settings,
seed, and paired cloned-OSC labels unchanged.

Judge capacity against the per-state local ridge reference, not an impossible
perfect held-out gradient. Require high fit-equation and row-gradient cosine,
and require held-out performance to reproduce the ridge teacher within the
registered tolerance. Report near-active rows separately. Do not add `q`,
`qdot`, controller state, geometry, loss weighting, a QP, or closed-loop
control in this gate. A pass authorizes only the next one-group input ablation;
a failure selects loss/decoder or secant-locality diagnosis.

Attempt `39293` produced no scientific data because `worker-0` exposes a
zero-byte `/usr/bin/nvidia-smi`; the allocation identity check failed before
simulator construction and training. Exclude that exact node for producer and
validator retries. This is a live-infrastructure repair and does not change
the preregistered ablation.

## ADR-0089: Retain compact inputs and diagnose secant locality next

Accepted after H100 producer `39295` and independent validator `39298`. The
25D one-state model materially repairs the rejected representation: value
RMSE falls to `0.701 mm`, fit-response cosine rises to `0.880229`, and the six
near-active L5 rows reach `0.931639` cosine with their ridge rows. Do not
return to raw simulator snapshots, duplicated controller state, or absolute
obstacle constants.

Keep the result as a strict prediction NO-GO. Fresh-direction cosine is only
`0.408501`, while the full-rank local ridge teacher is itself only `0.449661`.
The current `+0.05/-0.05`, 20-action affine response is therefore not a
reliable local teacher at one fixed physical state. Do not add `q`, `qdot`,
geometry, weighting, QP, or closed-loop control yet. The next independent
gate should change only secant radius. If smaller radii do not improve ridge
held-out response, replace full-vector affine-gradient supervision with a
direction-conditioned or nonlinear local response target.

## ADR-0086: Audit the frozen E05 Moka model before changing it

Accepted before replay. Reconstruct the immutable `39230` model from its
recorded state payload and replay the identical paired cloned-OSC data. Do not
call training, alter parameters, enlarge corrections, solve a QP, or execute
any corrected action.

Diagnose in causal order. First determine whether the MLP fits its own paired
train and validation response labels. Then measure groupwise support for the
untouched states using training-only statistics. Finally determine whether
errors concentrate in near-active link/time/compiled-primitive witnesses or
at primitive witness switches. Preserve all train/validation/test state
groups and both physical authorities. Select any later model or data change
only from this audit rather than from aggregate test failure.

## ADR-0087: Attribute failure to training underfit before state coverage

Accepted after final H100 producer `39265` and validator `39272`. The frozen
MLP fails paired response prediction on the five physical states it trained
on, not only on unseen actions 185--186. Do not treat additional episodes as
the first repair and do not use calibration, QP, or correction magnitude to
hide an incorrect response field.

The representation is overcomplete and poorly supported: 1,055 inputs, five
train contexts, 875 constant dimensions, duplicated controller state, raw
simulator internals, absolute clock, and a shifting twenty-action chunk.
Critical L5/g4 rows form only 4.29% of the train loss and are optimistically
wrong by about 48 mm. Preserve that primitive switching is absent. The local
ridge teacher has useful sign but inadequate held-out cosine, so teacher
locality and magnitude require a separate test.

Authorize only a prediction-fit gate: compact nonduplicated physical inputs,
one-state memorization before generalization, scale-normalized near-active
response weighting, and a separate secant-locality arm. Control remains
blocked until learned training response passes.

## ADR-0069: Train only the E05 multi-primitive controller-response pilot

Accepted before launch. E38 demonstrated physical collision-free task success
but also a persistent false-unsafe single-MVEE proxy; training against that
sign would conflate obstacle representation error with learned steering.
Return to the validated primary Moka task and represent its obstacle as the
union proxy of 15 compiled collision boxes. Keep raw MuJoCo contact and exact
solid overlap separate from the quantitative Loewner-box support gaps used for
optimization labels.

Do not repeat scalar contact classification, joint-trajectory prediction, or
monotone weighting of an already available analytical direction. Learn one
15D controller-conditioned response for each explicit future action/robot-row
witness from paired long-horizon outcomes. Preserve state-group splits and
freeze the model before actions 185--186. Learning is justified only if the
frozen direction beats fixed repulsion and matched random steering while
tracking the exact local-secant ceiling on both untouched states.

This is a one-task nearby-state feasibility test, not authorization to execute
the learned correction. A failure blocks QP and closed-loop work and requires
separate audits of value prediction, response prediction, and state support.
A pass supports only subsequent corrected-execution testing under a new gate.

Attempt `39229` exposed only an allocation-runtime incompatibility: the pinned
evaluation PyTorch has no H100 sm_90 kernels. Preserve the requested H100
allocation and move this tiny network's tensor operations to its allocated
CPU cores. This is an apparatus repair, not a protocol change; no scientific
rollout or result existed before the repair.

## ADR-0085: Reject the current E05 Moka response MLP before control

Accepted after clean H100 producer `39230` and independent H100 validator
`39238`. The model failed both untouched state gates: learned/exact direction
cosines were `0.676880/0.313475` and paired-response sign accuracies were
`0.679464/0.517857` at actions `185/186`. Although every learned correction
increased the reported clearance, it remained colliding and did not robustly
beat fixed repulsion; only one of six radius/state rows passed all local
criteria.

Keep this as a strict one-task learning NO-GO. Do not add a QP, increase the
correction radius, or execute closed loop: those operations cannot repair the
wrong learned action response. Preserve the exact multi-witness and fixed
repulsion arms as comparators. The next allowed work is read-only diagnosis of
train/validation response fit, held-out state support, and active
link/time/primitive witnesses. Any new training protocol requires a separate
preregistered gate and must not reinterpret this immutable result.

## ADR-0083: Test live continuation locally before a complete receding episode

Accepted before release. Gate 0 established that the safe compound prefix
fails under the immutable raw suffix, so the next experiment must change the
continuation rather than enlarge the same prefix correction. Use exactly one
fresh frozen-`pi0.5` query at the verified post-compound action-187 state and
compare its released-AEGIS five-action chunk with the registered safe
continuation. Run the smooth counterfactual field only when the fresh chunk is
unsafe, using the identical frozen chunk and initial state.

Do not yet run a full episode, train an MLP, or introduce a QP. The decisive
local question is whether feedback supplies a safe continuation or whether the
existing smooth field can repair it. Preserve the registered continuation as
a binding positive control, not as evidence about the live policy. Verify all
internal MuJoCo steps with the same 1 mm L5--L7 clearance, zero-contact, and
paper-CAR authority. A local pass authorizes the longer receding experiment;
a local failure blocks it and identifies missing safe continuation support
without spending a full-episode rollout budget.

Attempt `38989` showed that bitwise equality of the complete final dynamic
vector is not a valid additional gate across the rendered primary environment
and image-disabled probe. Retain the observed maximum difference, but do not
use it to override the actual experiment authority: exact initial probe
synchronization, identical commands, 25 internal measurements per action,
hard ellipsoid clearance, raw protected contacts, and CAR. This is an
apparatus-only correction made before any live policy query or scientific
result.

## ADR-0084: Reject the current local live-continuation field

Accepted after clean H100 producer `38990` and independent validator `38993`.
The registered action-187 continuation passed at `+8.151976 mm`, proving that
the measured post-detour state contains a safe five-action continuation. The
fresh frozen-`pi0.5` plus released-AEGIS continuation instead reached
`-12.433323 mm` with 35 protected-contact samples. Live feedback therefore
does not automatically preserve the safe detour.

The registered smooth counterfactual field supplied accurate local directions
through all ten iterations (held-out cosine at least `0.975932` and sign
accuracy at least `0.875`) and improved exact clearance by `12.186052 mm`.
It eliminated raw contact and passed CAR, but stopped at `-0.247271 mm` after
using path budget `1.0`, missing the required +1 mm buffer by `1.247271 mm`.
Classify this as a strict local NO-GO: accurate local descent is not sufficient
to recover the known safe continuation from the fresh VLA mode.

Do not train an MLP on this single-vector field, relax the clearance threshold,
or spend a full closed-loop episode. The next no-learning diagnostic, if
continued, should test whether the known safe continuation requires a larger
or explicitly structured/multimodal proposal from this exact state, with the
smooth field retained only for local refinement. The result does not reject
counterfactual execution supervision; it rejects the present proposal family
as the complete continuation generator.

## ADR-0081: Prove raw fixed-suffix safe support before multi-start search

Accepted before release. The validated positive compound correction modified
both the first five actions and the continuation after action 186. It is not a
positive control for a search that changes only raw actions 182--186 while
freezing raw actions 187--201. Run the two controls separately under identical
pre-action-182 state binding and internal-substep authority.

Only a safe transplanted compound prefix authorizes the planned radii
`1.0/1.5/2.0`, derivative-free, structured temporal, and spatial-branch
comparisons. If the full compound passes but the transplant fails, stop rather
than interpreting search failure as bad optimization: the fixed raw suffix is
the missing factor. If the full compound itself fails, first audit replay
binding and transient internal collision. Treat every radius as an upper bound
and retain matched-budget and matched-norm comparisons as separate later
analyses.

Attempt `38977` revealed that bitwise action binding must use direct registered
value assignment rather than an algebraically equivalent displacement round
trip. Accept this as a scientifically neutral apparatus repair: displacement
is still reported as compound minus raw, while the applied prefix is copied
from the immutable compound values before explicit action-bound checking.

## ADR-0082: Reject raw fixed-suffix multi-start; require continuation adaptation

Accepted after clean H100 producer `38978` and independent validator `38979`.
The full compound positive control passed strict internal-substep authority,
but its identical first-five prefix followed by the raw suffix failed despite
correction L2 `1.849546`, zero clipping, and bitwise action binding. It delayed
first L5 contact by four actions and improved worst clearance by `9.285844 mm`,
but still reached `-6.789842 mm`, 85 protected-contact samples, and failed CAR.

Accordingly, do not run the preregistered radii or structured multi-start arms
on this fixed suffix. Such a search would no longer distinguish optimizer
failure from an invalid positive-control assumption. The evidence identifies
continuation adaptation after action 186 as necessary for this known safe
trajectory. The next no-learning oracle may optimize a longer horizon or use
short-prefix execution plus live frozen-VLA replanning, but it must preserve
internal-substep clearance/contact/CAR authority. This does not yet prove that
every possible five-action prefix with the raw suffix is unsafe; it proves that
the known safe compound is not evidence of one, so the planned comparison is
not decisive and remains blocked.

## ADR-0080: Reject direct smooth-field attribution; retain it only as a compound repair

Accepted after clean H100 producer `38974` and independent validator `38976`.
The stricter internal-substep audit reproduced raw AEGIS as unsafe at
`-16.075686 mm`. Applying the same `2 mm` smooth all-witness secant procedure
directly to raw actions improved clearance to `-7.009143 mm`, but retained 125
protected contact samples, missed paper CAR, and supplied no verified-safe
scale on the registered correction ray. The final held-out directional cosine
also fell to `0.694629`, identifying loss of local field fidelity where exact
line search stalled.

Do not attribute the prior `+1.673938 mm` result to standalone smooth-field
avoidance. That result was reproduced only by composing the earlier detour
with the smooth second-stage correction; its total correction L2 from raw
AEGIS is `1.849546`. Smooth aggregation remains useful for locally improving a
given detour, but the current local field cannot discover that detour mode from
the raw suffix. Consequently, do not run live closed-loop recovery and do not
train an MLP to reproduce this failed direct field.

The next proposal must target the missing nonlocal or multimodal correction
structure, and must again be tested as a no-learning oracle before scale-up.
Potential proposal generators may preserve multiple candidate detour branches
or use derivative-free/global search, while hard internal-substep ellipsoid
clearance, raw contact, and CAR remain the acceptance authority. This result
does not reject counterfactual OSC supervision or smooth risk evaluation; it
rejects the claim that one locally re-estimated smooth descent path is enough.

## ADR-0079: Attribute the smooth field before closed-loop recovery

Accepted before release. The validated smooth candidate was a compound
correction built on the earlier five-action detour continuation. Do not credit
the smooth field as a standalone avoidance method until the identical
counterfactual procedure is applied directly to immutable raw AEGIS actions
182--201 from the same pre-action-182 simulator state.

Keep the detour and compound candidates as read-only comparators. Fit the direct
field with action-boundary rollouts exactly as before, add independent paired
direction validation, and make final safety authority stricter by measuring all
internal MuJoCo integration steps. Accept only a candidate with at least `1 mm`
ellipsoid clearance, no raw protected contact, and paper CAR. Search scales on
the discovered correction ray and call the selected result only the smallest
registered safe ray scale, never a globally minimum-norm correction.

If direct attribution fails, the smooth result remains a useful second-stage
repair but cannot motivate standalone learned steering. If it passes, the next
independent gate may execute the verified five-action prefix, requery the
frozen VLA with matched policy noise, and repeat receding verification. No
learning or PNCBF claim is authorized by this attribution gate alone.

## ADR-0078: Reject hard top-M selection; retain smooth all-witness evidence

Accepted after clean H100 producer `38966` and independent validator `38967`.
Preserving eight near-active rows in a local epigraph was not sufficient: its
best exact margin remained `-6.678093 mm`, it retained three L5 contacts, and it
missed paper CAR. This is a strict failure of the preregistered multi-witness
gate while the fixed derivative-free safe-support control remains positive.
Do not train a link/time-conditioned hard-row MLP from this result.

The matched `2 mm` smooth maximum of risk is retained as positive, narrower
mechanism evidence. It combined all 140 future link/time rows continuously and
reached `+1.673938 mm` with zero protected contact and paper CAR after seven
updates. Hard top-M selection can change discontinuously and can exclude rows
before they become critical; the smooth arm supplies continuous pressure from
all witnesses. This causal explanation is an inference from the matched result,
not yet a population conclusion.

Do not collapse this into a claim that the primary problem is solved. The arm
used exact counterfactual cloned-OSC labels at one archived state, did not test
task recovery, and had `19.576451 mm` terminal EEF deviation. The next gate must
keep the smooth field frozen and separately test either multi-state avoidance or
live frozen-VLA task rejoining after its verified first five actions. Learning
remains downstream of that oracle evidence.

## ADR-0077: Preserve link/time witnesses in the next oracle

Accepted before release. The fixed-step field failure does not justify another
scalar loss or a larger MLP. Its direct evidence is active-witness switching:
the hard minimum has no unique gradient at the observed boundary. Preserve the
identity and fitted action sensitivity of each future link/time clearance row.

Compare a current-worst row, a smooth minimum, and an explicit multi-witness
epigraph under matched paired-rollout, trust-region, exact line-search, path,
and correction budgets. In clearance notation, the epigraph constraints are
equivalently `-h_j-g_j^T delta <= t`; this avoids reversing the feedback's
risk-space inequality. Select at most eight rows within 5 mm of the worst
clearance, retaining deterministic `(action offset, ellipsoid row)` identity.

Use exact cloned OSC only for counterfactual row labels and fresh acceptance.
Do not impose task or endpoint terms in this avoidance gate. If the coordinated
rows recover the known safe detour and beat the single row, the result supports
learning `G_theta(x,A,j)` rather than one 15D vector. If they fail while the
fixed derivative-free control is safe, the local convex representation is
still insufficient and the next method must represent alternative nonlocal
detours rather than add learning.

H100 attempt `38964` is retained as apparatus failure. A visited action was
near a normalized coordinate boundary, so the old rejection sampler could not
construct all symmetric `±0.05` pairs. Repair only the local sampling domain:
freeze coordinates whose bidirectional headroom is below `0.05` and sample the
same paired directions in the remaining axis subspace. This is the maximal
conservative symmetric neighborhood at the registered perturbation scale; it
does not waive action limits or change the scientific comparison.

## ADR-0075: Test the long-horizon field without task preservation

Accepted before release. Job `38955` already proved that collision-free
five-action corrections exist once endpoint preservation is removed, but its
learned field used exact backtracking and stopped after local witness
switching. This did not answer the narrower avoidance question requested here:
whether repeatedly recomputing the pure long-horizon field and taking small
fixed steps can reach an already-known safe detour.

Freeze an avoidance-only arm at E05 action 182. Re-estimate the field from 32
paired cloned-OSC outcomes at every center, take exactly one normalized `0.1`
step, and evaluate the complete twenty-action continuation containing the
action-197 contact. Do not impose endpoint equality or use task attraction,
task penalties, QP, MLP, or flow injection. Compare against repeatedly using
the initial field direction and retain the existing radius-1 analytical and
derivative-free oracles as no-learning safe-support controls.

Terminal EEF displacement is reported but is not part of this avoidance gate.
Passing requires positive exact L5--L7 clearance, zero protected contact,
paper CAR, radius at most one, and superiority to the fixed initial direction.
If the field fails while the controls remain safe, reject this single-gradient
correction representation before training. If it passes, the next independent
question is whether live frozen-VLA replanning can recover the task after the
temporary detour.

## ADR-0076: Reject the single hard-min action field before learning

Accepted after clean H100 producer `38961` and independent validator `38962`.
The registered field was not useless: ten recomputations improved exact hard
clearance by `7.138110 mm` from nominal and beat repeated use of the initial
direction by `5.736277 mm`. It nevertheless remained at `-7.125095 mm`, kept
two raw L5 contacts around actions 196--197, and missed paper CAR.

This is decisive because radius-1 analytical and derivative-free controls are
already exactly safe under the same avoidance authority. The failure is not
absence of an action-space detour. Every one of the 32 paired probes switched
the active future witness at every iteration, so a single gradient of the hard
minimum averages incompatible local modes and zigzags. Ten full normalized
steps resulted in only `0.749880` net correction norm.

Do not train an MLP to reproduce this vector, and do not proceed to live VLA
rejoining from an unsafe action. If the research direction continues, first
test the smallest multi-mode oracle that preserves separate active link/time
directions and selects a verified branch. Only a verified-safe field output
can authorize learning or closed-loop recovery.

## ADR-0073: Execute one action from every verified five-action window

Accepted as the decisive follow-up to `38881`. Recompute a fresh five-action
VLA window from the measured state after every executed action; do not execute
all five actions from a previously corrected window. A nominal window may pass
only with at least 1 mm exact L5--L7 clearance, no robot contact, and paper CAR.
Otherwise, run the validated task-rejoining five-action SQP, verify the complete
candidate again, execute only its first action, and shift the horizon. Preserve
the frozen VLA and OSC and train no MLP until this exact oracle achieves safe
task completion.

## ADR-0074: Do not learn from the infeasible receding five-action oracle

Accepted after clean H100 producer `38903` and independent validator `38907`.
The receding controller correctly identified a new unsafe window at action 185,
but its best exact five-action SQP candidate achieved only `+0.369923 mm`, below
the registered `+1 mm` buffer. It therefore stopped with zero contact and zero
CAR but without task completion. Learning a direct correction field from this
oracle is not justified because the teacher has no accepted action at an
encountered state. First test earlier intervention or a longer/richer
task-rejoining candidate family; do not relax the buffer after observing the
outcome.

## ADR-0069: Bind executable field recovery to archived action 185

Accepted. Do not interpret array `38848` scientifically. The executable
field-recovery comparison is bound to the exact archived pre-action-185 state;
the ordinary one-step candidate filter is therefore disabled for actions
0--184, which execute byte-for-byte from the immutable successful released
AEGIS artifact. This is a protocol correction, not a safety result.

## ADR-0070: Do not train a field-mixing MLP from the late E05 repair

Accepted after corrected H100 producer `38854` and independent validator
`38867`. Fixed analytical, normal-only, normal-plus-tangent, and unrestricted
two-action post-hoc repairs all prevented every active-obstacle robot contact
and paper CAR from archived action 185 onward, but no arm completed the task.
Normal-plus-tangent therefore offers no task-success value beyond the fixed
analytical repulsion baseline. Learning is not justified from this gate. The
next safety-and-task experiment must intervene earlier and/or optimize a longer
task-aware horizon; late-denoising timing cannot rescue a task mode already
destroyed by the required approximately `0.89--1.00` normalized correction.

## ADR-0071: Compose the early task-rejoining detour with receding verification

Accepted after H100 job `38874`. The zero-sum five-action detour restored task
competence and the frozen policy completed, but L5 collision recurred ten
actions after the detour. Keep the validated detour unchanged and filter each
subsequent released-AEGIS action with the existing exact cloned-OSC candidate
filter. This directly tests whether early task preservation plus receding
physical safety can jointly achieve safe task success; it adds no learning.

## ADR-0072: Reject one-step post-detour filtering and retain the five-action horizon

Accepted after paired producer jobs `38874`/`38876` and independent H100
validator `38881`. The endpoint-preserving five-action detour converted the
archived `-9.313104 mm` minimum to `+1.747199 mm` and allowed the frozen policy
to complete, but unfiltered execution contacted L5 at action 197. The existing
one-step exact filter found no verified candidate at its first post-detour
decision (action 187), despite 87 cloned candidates, and therefore could not
preserve task completion. Persistent safety must be optimized over a receding
task-rejoining horizon. Do not train a field-mixing MLP or enlarge one-step
repulsion from these outcomes.

## ADR-0073: Use persistent physical routes before learning policy value

Accepted after H100 producer `39029` and validator `39032`. At the identical
unsafe action-187 state, a constant five-action obstacle-normal retreat reached
the strict +1 mm internal-substep clearance gate at correction L2 `1.0`, while
left and up routes failed and the right route remained marginally unsafe.
Smooth local refinement and full 15-D derivative-free search also found exact
safe candidates. The earlier local-field failure was therefore not absence of
correction authority; it lacked a sufficiently useful route proposal.

Proceed to one full receding E05 oracle with persistent route memory, live
frozen-VLA feedback, exact five-action look-ahead, short-prefix execution, and
hard verification. Do not train a value model until that reproducible backup
policy completes E05 without L5--L7 contact, CAR, or deadlock.

## ADR-0001: Start from the untouched author release

Accepted. The reproduction branch starts at upstream commit `57b1aef`. The
previous same-state canary code is not copied into this worktree.

## ADR-0002: Reproduce the translational protocol first

Accepted. The user's Table 1 values correspond to `pi0.5,t` and `AEGIS_t`.
Full-action AEGIS is a separate supplementary protocol and cannot be mixed into
the primary result.

## ADR-0003: Repair apparatus defects without changing the safety algorithm

Accepted. Permitted repairs are limited to:

- pass the suite name into the existing point filter;
- recognize `safelibero_long` and its 550-action horizon;
- represent empty perception and infeasible QP as explicit terminal apparatus
  states rather than undefined-name crashes;
- add structured artifacts, deterministic pairing controls, streaming videos,
  and diagnostic measurements around the released loop.

No CBF, MVEE, GroundingDINO, threshold, action scale, solver, or policy setting
may be tuned to improve the result.

## ADR-0004: Make policy-noise pairing opt-in

Accepted. The reserved `__crfs__` request envelope carries a per-request JAX
seed and is removed before normal OpenPI transforms. Requests without the
envelope retain the upstream inference path.

## ADR-0005: Freeze Codex labels before outcomes

Accepted because GLM-4.5V is unavailable and the user explicitly requested
Codex. Capture runs execute only reset, the 20 settling steps, and lossless
image export. A label is valid only when bound to the exact settled image hash
before any policy or AEGIS outcome is produced.

## ADR-0006: Report paper metrics and physical diagnostics separately

Accepted. The primary table retains:

- CAR: no active-obstacle L1 displacement greater than 1 mm;
- TSR: environment task success within the suite horizon;
- legacy ETS: the released loop counter.

Each result also records executed action count, MuJoCo contact evidence when
available, selector/perception/MVEE/QP state, intervention magnitude, and video
identity. These diagnostics must never be substituted into the paper table.

## ADR-0007: Never aggregate a partial population

Accepted. The aggregator requires all 1,600 unique cases for every requested
method. Policy failures remain scientific failures; apparatus failures block
publication and must be rerun rather than dropped.

## ADR-0008: Align the first canary with the paper's qualitative task

Accepted. The capture-only canary uses manifest ordinal 100
(`safelibero_spatial`, Level I, logical task 2, episode 0), matching the task
in the supplied failure/success figure. The episode index is not claimed to be
the unpublished episode used to make the authors' figure.

## ADR-0009: Separate capture readiness from outcome readiness

Accepted. A fresh capture-only allocation may collect pre-outcome evidence
while `A01-reproduction-apparatus` remains active. No pi0.5 or AEGIS outcome
may run until the full checkpoint identity and paired-result validator are
implemented and independently reviewed.

## ADR-0010: Use the allocation-compatible OSMesa renderer

Accepted after exact job `28460_0` failed before reset because EGL attempted
to open inaccessible host render devices. VinUni's legacy Robosuite/MuJoCo
workloads use `MUJOCO_GL=osmesa` and `PYOPENGL_PLATFORM=osmesa`. This changes
only the headless rendering backend; the H100 remains allocated for pi0.5
inference in outcome jobs.

## ADR-0011: Restore the released pre-construction NumPy seed

Accepted after exact job `28461_0` reached OSMesa but failed during the
environment constructor's temporary random placement. The released evaluator
sets NumPy seed 7 before constructing OffScreenRenderEnv. Capture now restores
that same ordering before applying the immutable full simulator state.

## ADR-0012: Retain released-method failures and preserve the frozen action space

Accepted. Once all required dependencies and inputs are valid, an OSQP
exception, infeasible/invalid QP solution, or empty perception output is a
failure of the released method on that case, not permission to exclude the
case as an apparatus failure. Hard QP failures terminate and remain in the
population; empty perception uses an explicitly reported fail-open path.

The upstream empty-perception branch forwards the raw nominal action, including
rotation. That behavior conflicts with the registered Table-1
translational-only arm. The reproduction therefore applies the already
corrected translational nominal action (XYZ and gripper preserved, rotation
zeroed), reports `method_failure_passthrough`, and records both the corrected
execution and upstream behavior in every affected artifact. This is a
protocol-preserving wrapper correction, not a claim that AEGIS intervened.

## ADR-0013: Treat deterministic geometry failure as no-execution method failure

Accepted. A valid GroundingDINO result can still fail in the released
ConvexHull/MVEE construction. Such a case is retained as an exact pre-control
`aegis_geometry` method failure with zero policy queries, zero executed
actions, one evidence frame, and `safety_by_no_execution=true`. Only this
exact state may omit the first returned-action hash; all other failures remain
fully action-bound.

## ADR-0014: Require real QP execution before the paired canary can pass

Accepted. Successful schema validation is insufficient to establish that the
AEGIS integration ran. The paired-canary receipt additionally requires the
frozen selector to match the active obstacle, perception status `ready`, and
at least one executed `aegis_qp` action with valid finite OSQP diagnostics.
Fail-open, no-action, selector-mismatch, and empty-geometry canaries are
integration-inconclusive and cannot authorize the population.

## ADR-0015: Use the released GroundingDINO CUDA path first

Accepted. The first paired canary uses `GROUNDINGDINO_DEVICE=cuda`, matching
the release default. A CPU perception run would be a separately preregistered
compatibility variant, not a silent replacement for the released baseline.

## ADR-0016: Exclude mount-local `st_dev` from cross-worker identity

Accepted after paired-canary task `28468_0` stopped before execution. The
checkpoint hash was computed on worker-2, but Linux reports a different
`st_dev` for the same shared file through another cluster mount namespace.
`st_dev` identifies a mounted filesystem view, not file content, and therefore
cannot be compared across Slurm workers. The stable identity continues to bind
the resolved checkpoint path and every file's relative path, byte size, inode,
mtime, and ctime, in addition to the allocation-backed full-content tree hash.

## ADR-0017: Keep evaluator and publisher compatible with Python 3.8

Accepted after retry-B task `28470_0` reached the evaluator but failed before
the first policy query because `str.removesuffix` is unavailable in the
released SafeLIBERO Python-3.8 environment. Equivalent prefix/suffix slicing
is used in the evaluator, manifest builder, and aggregator, and a structural
test prohibits both Python-3.9-only string helpers in these runtime modules.

## ADR-0018: Prove failure diagnostics are action-invariant

Accepted. The revised ordinal-100 integration gate runs four serial rollouts
inside one allocation and one pi0.5 server:

1. pi0.5 with diagnostics disabled;
2. pi0.5 with diagnostics enabled;
3. pi0.5+AEGIS with diagnostics disabled;
4. pi0.5+AEGIS with diagnostics enabled.

For each method, the enabled and disabled runs must preserve exact returned
action bytes, executed action bytes, query-indexed noise schedule, scientific
outcome fields, simulator state semantics, native goal progress, and decoded
terminal video content. Only observer timing, diagnostic records, diagnostic
file encodings, and their hashes may differ. This is an apparatus gate, not a
population efficacy result.

## ADR-0019: Freeze the revised gate to GroundingDINO on CPU

Accepted and supersedes ADR-0015 for the revised canary and dependent
population. The validated historical canary
`vlsa-table1-paired-canary-20260717d` used GroundingDINO on CPU and produced
the exact AEGIS action reference used by the action-invariance gate. Allowing a
CUDA detector in the revised gate could change detector numerics and AEGIS
actions, making the reference comparison ill-posed. Reservation, allocation,
publisher, and receipt validation therefore all require the CPU detector.

## ADR-0020: Bind outcomes to the complete outcome-blind label publication

Accepted. Evaluation requires the exact 1,600-row Codex label manifest and
its independently validated publication receipt. Every canary result must
embed the exact selected ordinal-100 label record, and receipt regeneration
rechecks the same binding. A copied action reference, a changed label row, a
different mode/arm mapping, or a label publication with any outcome execution
is rejected.

## ADR-0021: Keep the full population blocked until diagnostics scale safely

Accepted. Enabling diagnostics in the population runner is insufficient by
itself. Before release, the CPU publisher must deeply validate every case's
GroundingDINO boxes, point clouds, filtering and MVEE outputs, QP attempts,
step-resolved contacts, native goal-progress trace, terminal lossless frame,
and decoded video. Validation and aggregation must stream or compact each pair
so they do not retain all 3,200 full diagnostic result dictionaries in memory.

## ADR-0022: Accept the action-invariant canary, not a population claim

Accepted from terminal allocation-backed evidence. Exact Slurm task `28590_0`
completed on `worker-2` from clean release commit
`5105894faebd40d2e27e2011d33688b25c2578dc`. Receipt SHA-256
`6a80afce3ce019bf43090da65c85717344b6dafb4a983d5a9efcf6bbecd46d95`
validated all four frozen ordinal-100 rollouts. Diagnostics off/on preserved
exact action bytes, policy-query schedules, outcome semantics, and video bytes
for both arms. The baseline collision and task failure and the AEGIS
collision-free task success reproduce the historical canary. This passes
`A03-one-case-paired-canary` and permits work on `A04-population`; it does not
establish AEGIS population efficacy or authorize a launch before the
memory-safe deep population validator passes.

## ADR-0023: Stream, recompute, and byte-bind the population publication

Accepted for implementation; allocation-backed validation remains pending.
The publisher may hold at most one full result pair in memory. It deeply
validates that pair, writes a compact receipt row, and releases the full
objects before reading the next case. After all 1,600 pairs are present, an
independent finalizer reloads the immutable manifest and configuration,
recomputes the aggregate table, exhaustive failure report, and indexed video
gallery from the actual result tree, and requires their exact hashes and bytes
to match the staged publication. It retains every scientific and apparatus
failure. Static-support contacts remain reported but are not counted as
robot--obstacle collision mechanisms.

## ADR-0024: Bind failure explanations to live simulator authority

Accepted for a fresh allocation-backed canary; population authorization
remains pending. The active obstacle is bound to the settled-input contract,
not inferred from a mutable result field. Each episode freezes the complete
MuJoCo body-parent and geom-to-body tables, exact named/unnamed ID tokens,
body-to-joint ranges, joint ownership/types/names, robot body IDs, the
active-obstacle root body, and the native task-object mapping. Every step
stores the ordered raw MuJoCo contact ledger; canonical events, normals, body
lineages, dynamics, mobility, roles, and goal membership are re-derived
against that authority. Goal arguments are also checked against the immutable
BDDL file. Direct goal objects, parented goal sites, and parentless fixed
arena sites are distinct bindings. Any ambiguous or unnamed non-world role is
`unknown` and fails publication. This contract is explicitly versioned as
contact schema `v3`, model-authority schema `v2`, and source-bound paired
canary receipt schema `v2`; the population reservation rejects stale receipt
schemas before allocation.

Independent validation reconstructs the exact robot-body ID set from frozen
MuJoCo body names rather than trusting the producer's list. It also requires
every task-object and goal-site-parent root ID to resolve to that object's
exact frozen body name or registered instance prefix. Coordinated,
fully-rehashed attempts to omit a contacted robot body or exchange two task
roots are rejected. A malicious rewrite of the complete frozen authority,
raw ledger, and all hashes remains the explicit clean-source producer trust
boundary; the fresh source-bound canary must validate that boundary before
population work.

These observer changes were made after validated task `28590_0`. That task
remains valid historical integration evidence for `A03`, but it cannot
authorize a population from the new source. The final clean release therefore
requires a new source-bound checkpoint receipt and the same four-rollout
ordinal-100 action-invariance canary before any population submission.

## ADR-0025: Start multi-link geometry as an opt-in read-only AEGIS observer

Accepted. Branch `codex/multilink-ellipsoid-qp` starts from clean current
AEGIS reproduction commit `1592aa59361f431ba96c6ddcbebcb596f6c20853`.
Completed Table-1 artifacts are immutable inputs. The first gate reruns only
primary case `vlsa-t1-goal-ii-t0-e05` and requires exact action-ledger equality
with archived AEGIS while evaluating the new geometry and QP before each
unchanged `env.step` call. This separates the questions "does the geometry see
the arm?" and "how expensive is the coupled QP?" from active-control efficacy.

Every contact-participating collision geom on Panda link 1 through link 7 has
its own conservative enclosing ellipsoid. This is a task-independent whole-arm
set; the historical collision link is not an input. The obstacle remains the
released frozen AEGIS perception MVEE. The correction variable is a physical
seven-joint velocity, derived from the executed translational AEGIS action only
as a read-only resolved-rate nominal. No neural model is trained.

## ADR-0026: Preregister optimizer clearance, simulator evidence, and timing

Accepted. The first shadow uses `alpha=10 s^-1`, `D_opt=0.01 m`, physical
joint-velocity bounds of `+/-0.5 rad/s`, resolved-rate damping `0.05`, and an
end-effector-preservation metric `I + 10 J_ee^T J_ee`. OSQP tolerances and all
parameters are frozen in
`configs/vlsa_multilink_ellipsoid_shadow_e05.v1.json` before H100 execution.

`D_opt` is the support-gap buffer used by the optimizer. `D_sim` is raw MuJoCo
nonpositive contact distance plus the separately reported active-obstacle
displacement. They remain distinct in every step record and validation
receipt. QP setup, solve, solver-reported, and total observer times are all
retained. Infeasibility is a reportable result, never silently dropped. The
only claim available from this gate is oracle geometry/QP implementation and
timing evidence; active safety requires the dependent `E02` gate.

## ADR-0027: Accept certified mesh bounds and retain hard-QP infeasibility

Accepted after validated Slurm job `36757`. The live Panda collision model has
one mesh geom on each of links 1 through 7. For this gate, each mesh remains
inside MuJoCo's compiled broad-phase `geom_rbound` sphere. The independent
validator recomputes every recorded sphere semiaxis from that authority and
requires all seven rows to enter the same QP at every action. We do not infer a
tighter mesh ellipsoid from `geom_size` without a separate mesh-vertex-bound
certificate.

The conservative geometry produced 182 solved and 55 primal-infeasible hard
QPs under the preregistered `+/-0.5 rad/s` limits. This is a valid, retained
oracle outcome, not an apparatus failure and not evidence for active safety.
Any `E02` active-control gate must first preregister either tighter certified
mesh ellipsoids, an explicitly penalized slack hierarchy, or both. A KKT/VI
network cannot be trained against silently discarded infeasible targets.

## ADR-0028: Narrow the next geometry gate to surface-fitted links 5, 6, and 7

Accepted by direct user instruction. `E02` excludes links 1 through 4 and
uses exactly three independent ellipsoids: one each for `robot0_link5`,
`robot0_link6`, and `robot0_link7`. These are not grouped multi-body
ellipsoids. Each bound is fitted from that rigid link's compiled MuJoCo
collision-mesh vertices using a Khachiyan minimum-volume fit, inflated just
enough to contain every vertex, and
retains the ordinary analytical rigid-link Jacobian used by the QP.

The obstacle remains the frozen released-AEGIS MVEE, and the QP receives
exactly three simultaneous constraints. This gate remains read-only until its
H100 visualization, containment certificate, action-invariance check, and
runtime receipt pass. Links 1 through 4 are deliberately outside the claim.

The validation rollout replays the immutable archived AEGIS `env.step` inputs
instead of asking the GPU policy to resample them. This is required after
attempt `36774` reproduced the settled simulator and policy-noise seed contract
but returned a different first policy chunk before the shadow observer ran.
Exact replay preserves the executed-action horizon while independently
recomputing MuJoCo contacts, task progress, three-link constraints, and timing.

## ADR-0029: Execute the first multi-CBF counterfactual in AEGIS XYZ space

Accepted. SafeLIBERO exposes a translational OSC action rather than direct
joint-velocity control. The first active experiment therefore maps executable
XYZ through the live damped resolved-rate kinematics, maps each L5/L6/L7
barrier row back to that XYZ variable, and solves all three inequalities in
one QP. This avoids claiming that an arbitrary joint-velocity solution was
executed when the simulator accepts only Cartesian actions.

The nominal sequence is the immutable released-AEGIS `env.step` ledger. This
is a controlled counterfactual for the safety layer, not closed-loop policy
efficacy. The QP may change only XYZ; rotation remains zero and the gripper
command is exact. Raw protected-link contact, paper CAR, and native goal state
jointly determine whether the primary report problem is solved.

## ADR-0030: Retain the negative Cartesian multi-CBF outcome

Accepted after clean H100 job `36782`. The exact-action controller solves all
three L5/L6/L7 inequalities at every action, but direct L5 contact and paper
CAR occur at action 189 and the native task does not complete. This is a
scientific controller failure, not an apparatus failure: source, initial
state, archived nominal ledger, geometry, solver status, and raw simulator
contacts are all complete.

We do not tune `alpha`, `D_opt`, ellipsoid size, or a fallback after observing
this result. The outcome shows that satisfying the continuous resolved-rate
linearization in the executable XYZ action is insufficient for discrete
SafeLIBERO OSC contact avoidance in this case. A later experiment must
preregister direct joint-space execution or nonlinear discrete-time
rollout/step reduction; KKT/VI learning is not a remedy for this model-to-
execution mismatch.

H100 video replay `36788` independently re-executed the accepted safe-action
ledger and matched the recorded end-effector, obstacle, and raw-contact traces
with zero error. A finite-difference audit identifies the concrete certificate
failure: action 186 predicted positive L5 barrier recovery (`+0.044412 m/s`)
while the next simulator state moved inward (`-0.143104 m/s`). This reinforces
the decision: retain the negative outcome and change the execution model in a
new preregistration, not the QP solver or KKT implementation.

## ADR-0031: Retain the direct joint-velocity pilot as baseline-incompetent

Accepted after paired H100 producer job `36802` and independent validator job
`36806`. The `pi05_droid` checkpoint directly drove all seven SafeLIBERO
joint velocities, and the active arm executed one simultaneous QP containing
the three independent L5/L6/L7 ellipsoid constraints. Pairing is exact until
the first material safety intervention at action 71, and every QP in both
225-action arms is valid.

Neither arm contacted the protected links or crossed the paper CAR threshold,
but neither arm ever satisfied the native task goal. Most decisively, the
unfiltered baseline never issued a close-gripper command. This makes the
baseline incompetent for the requested task, so the collision-free result is
not evidence that the three-constraint filter solves the original
task-completing AEGIS failure. We retain the videos, timings, raw evidence,
and negative interpretation without tuning the checkpoint or safety
parameters after outcome inspection. Any learned joint-space replacement
requires a new preregistration and a competent paired baseline before safety
efficacy or KKT/VI training can be considered.

## ADR-0032: Retain the AEGIS Cartesian-to-joint bridge as controller-incompetent

Accepted after paired H100 producer job `36824` and independent validator job
`36826`. The bridge starts from the exact archived AEGIS OSC-settled simulator
state, converts every immutable AEGIS XYZ action to live seven-joint velocity,
preserves every archived gripper command, and in the active arm executes one
simultaneous QP containing the separate L5, L6, and L7 ellipsoid constraints.
The two arms have identical pre-step simulator states and nominal joint
velocities through their first material difference at action 200; every one
of the 474 QPs is valid.

This replacement does not preserve the task or the original collision
mechanism. Neither arm completes the SafeLIBERO goal. Both push the obstacle
with the hand/gripper beginning at action 159 and displace it by about 0.245 m,
while neither records L5/L6/L7 contact. Consequently the single late L5--L7
intervention cannot answer whether the filter prevents the released AEGIS
link-5/link-6 failure. The retained interpretation is
`bridge_incompetent_no_safety_efficacy_claim`, not a safety success.

We do not tune the frozen Cartesian scale, damping, QP parameters, or
ellipsoids after inspecting this outcome. An archived Cartesian ledger is
feedback-dependent: once a different low-level controller changes the state,
later open-loop commands no longer represent what the policy would choose.
The next controller experiment therefore requires a new preregistration with
a task-competent Cartesian policy queried on live observations and competence
established in the unfiltered joint-controller arm before applying L5--L7
multi-CBF. KKT/VI learning remains gated on that oracle comparison.

## ADR-0033: Test a cloned-step discrete L5--L7 multi-CBF before learning

Accepted by direct user instruction. The accepted continuous XYZ multi-CBF
remains frozen as the first comparison. A new opt-in arm replaces only its
resolved-rate derivative with seven cloned SafeLIBERO transitions per action:
one nominal transition and clipped central perturbations in the three XYZ
dimensions. Their three next-step L5/L6/L7 buffered clearances define one
simultaneous three-row QP. The candidate is then executed in the clone and may
be reduced by the frozen scales `0.5, 0.25, 0.125, 0` toward stop until an exact
next-step check passes.

Every probe begins from a synchronized copy of the complete MuJoCo and
stateful OSC/gripper transition state. The primary `env.step` must reproduce
the accepted clone state and clearances within the preregistered tolerance.
No unverified action is executed. `D_opt=0.01 m`, the accepted L5/L6/L7 MVEEs,
the frozen obstacle MVEE, archived rotation/gripper commands, and the 237-step
nominal ledger remain unchanged. The experiment measures oracle ability and
runtime on the primary case only; it is not claimed to be deployable in real
time and does not authorize KKT/VI or universal-formula training.

## ADR-0034: Retain the cloned-step multi-CBF as safe refusal, not task success

Accepted after authoritative H100 job `36875`. The discrete controller used
exactly three L5/L6/L7 next-clearance constraints, all 186 QPs were valid, and
each of the 185 executed primary transitions reproduced its accepted cloned
transition with zero state error. It produced no robot contact, no paper CAR,
and negligible obstacle displacement, but it stopped before executing action
185 and never completed the native task. Therefore
`primary_problem_solved=false`; collision avoidance obtained by terminating
the task is not counted as a safety-method success.

The terminal evidence distinguishes the remaining failure from the original
continuous resolved-rate mismatch. The finite-difference QP predicted its
large correction would put L5 exactly on the buffered boundary, while exact
simulation put it `3.697 mm` inside. More importantly, even the predefined
zero-XYZ stop action ended `6.852 mm` inside the buffered boundary on the next
step. Thus exact one-step verification correctly vetoed every action, but the
state had already left the one-step viable set for the frozen `D_opt=10 mm`
requirement. Three ellipsoid rows are present and jointly solved; adding a
neural approximation of this same QP would not create a feasible action.

We do not reduce `D_opt`, weaken the exact verification tolerance, or add an
unregistered retreat after observing this outcome. A follow-up must be newly
preregistered and address anticipation, for example a multi-step viability
margin or direct simulator-in-the-loop nonlinear search begun before the
one-step stopping boundary. The observed mean complete-filter time of
`131.895 ms` also exceeds the 20 Hz control period, although the QP itself
remains fast at mean `1.182 ms`. KKT/VI or universal-formula learning remains
gated because it can only approximate a controller after the oracle target is
both feasible and task competent.

## ADR-0035: Test predictive full-body flow guidance with cloned OSC dynamics

Accepted by direct user instruction before implementation or outcome review.
The next `E02` experiment moves the safety correction from a post-sampling,
one-action QP into every Euler step of the ten-step pi0.5 flow sampler. It uses
40 discrete trajectory-CBF inequalities: ten future steps for each of the
separate L5, L6, L7, and released AEGIS end-effector ellipsoids. The L5--L7
MVEEs, frozen obstacle MVEE, `D_opt=0.01 m`, paper decay `gamma=0.9`, and
translational-only action protocol remain fixed.

The trajectory model is not the rejected resolved-rate approximation. It is
locally identified by centrally perturbed ten-step rollouts from a complete
clone of the current MuJoCo, stateful OSC_POSE, and gripper state. The affine
constraints are transformed into pi0.5's normalized action coordinates with
scale only for displacement, then projected after each neural Euler update.
The returned chunk must pass an exact cloned ten-step OSC rollout before any
of its first five actions may execute. One frozen outer relinearization is
allowed; failure after that is retained as method failure, never passed
through or hidden by stopping-task success.

This is a single-case oracle capability test, not a real-time or population
claim. The default OpenPI sampler and ordinary AEGIS evaluator remain
unchanged without the reserved opt-in control envelope. No KKT/VI or neural
approximation training is authorized by this decision.

## ADR-0036: Retain predictive flow guidance as safe but task-incompetent

Accepted after clean H100 job `37024`. The live initial observation and first
pi0.5 chunk exactly match the immutable Table 1 task-success episode. All 18
guided proposals used cloned stateful OSC_POSE horizon models, every executed
guided chunk passed exact ten-step verification, and the 300 executed actions
produced no robot contact, no paper CAR, and negligible obstacle motion.

This is not counted as solving the primary problem because the robot never
moved the bowl or satisfied any native goal atom. All 16 modified policy
queries occurred in steps 0--79. The minimum L5/L6/L7 buffered clearances
remained `108.667/129.095/95.291 mm`, while the end-effector proxy reached
`0.333 mm`; therefore the predictive end-effector constraint, not a distal
link constraint, caused every intervention and diverted the closed-loop
policy before the archived L5/L6 collision regime.

The retained interpretation is
`predictive_full_body_safe_but_task_incompetent_no_safety_efficacy_claim`.
The result verifies that an OSC-consistent cloned transition and exact rollout
gate can enforce the specified horizon inequalities, but it also demonstrates
that those hard inequalities can remove the task route. We do not weaken the
margin, shorten the horizon, remove the explicitly requested end-effector
constraint, or tune activation after observing the outcome. Any follow-up is
a new preregistered experiment and must recover a task-competent paired
baseline before KKT/VI or neural approximation training is reconsidered.

## ADR-0037: Require decoded-pixel verification for simulation videos

Accepted after auditing the job-`37024` MP4. A readable container, plausible
frame count, and valid first frame are insufficient evidence: that file became
visually corrupted after frame zero. It is retained only for provenance and
must not be presented as simulation evidence.

Replacement videos are reconstructed on H100 from the immutable accepted
action ledger. Acceptance requires exact replay of recorded primary states,
end-effector coordinates, obstacle motion, contact, CAR, and task outcome;
visible L5/L6/L7/end-effector and obstacle ellipsoid overlays; successful
decoding of every expected frame; and distributed decoded-pixel comparison
against the in-memory source frames. H100 job `37031` satisfied this contract
and completed normally. A renderer receipt written before a nonzero Slurm exit
is not sufficient: prior attempt `37029` is an apparatus failure even though
its bytes happened to pass the content checks.

## ADR-0038: Test EmbodiSteer's metric and schedule as an OSC surrogate

Accepted by direct user instruction before the outcome is known. The primary
reference is Wang et al., *EmbodiSteer* (arXiv:2606.12965). We adopt two
specific mechanisms: the task-preservation metric from Eqs. (6), (14), and
(15), and the reverse-step guidance schedule from Eq. (16). We extend the
paper's one safety inequality to one simultaneous projection containing every
registered L5/L6/L7/end-effector horizon row.

We do not label the experiment full EmbodiSteer. The released, task-competent
SafeLIBERO path executes Cartesian `OSC_POSE` actions; converting a corrected
joint null-space trajectory back to Cartesian commands would erase the paper's
central redundancy. The preregistered surrogate instead identifies the
end-effector-trajectory metric and barrier derivatives through exact cloned
OSC transitions, applies metric multi-row projection after each pi0.5 flow
Euler update, and verifies the proposed final chunk by exact cloned rollout.
It isolates whether the paper's task-preserving metric plus late guidance fixes
the earlier Euclidean hard-projection task diversion.

The four paper baseline groups are used as an explanatory evidence matrix, not
as a falsely matched reproduction. Existing immutable evidence supplies the
Cartesian baseline, direct-joint diagnostic, and post-hoc multi-CBF outcomes;
only the scheduled task-metric flow arm is newly executed. The direct-joint
diagnostic remains baseline-incompetent and uses a different checkpoint, so it
cannot support a cross-group efficacy claim. Success for the new arm still
requires native task completion, no robot contact, no paper CAR, and exact
verification of every executed guided chunk.

## ADR-0039: Retain task-metric guidance as safer but still task-incompetent

Accepted after clean H100 job `37042`. The run exactly matched the immutable
initial pi0.5 chunk, executed 300 actions with no contact or CAR, and accepted
only chunks passing exact cloned-OSC verification. It did not move the bowl or
complete the task. Therefore it is a negative capability result, not evidence
that the requested primary problem is solved.

EmbodiSteer's metric and late schedule materially reduced over-steering versus
the earlier Euclidean flow projection: guidance ended 40 actions earlier and
the end effector approached the bowl by another `72.707 mm`. The remaining
failure is structural. At the initial paired state the successful nominal
route violates only the released end-effector-proxy horizon constraint by
`12.015 mm`, while every distal-link constraint is far inside its safe set.
Any hard projection must change that route; minimizing end-effector deviation
cannot make a declared-infeasible route feasible.

This result does not reject full EmbodiSteer. The tested surrogate lacks the
paper's defining joint-space sampling and null-space execution authority.
Conversely, implementing true joint-space denoising requires a task-competent
joint controller baseline, which the prior direct-joint and bridge pilots did
not establish. We do not post-hoc remove the end-effector row, change its
geometry, weaken the margin, or tune the schedule. Such changes require a new
preregistered comparison.

## ADR-0040: Establish matched EE and Joint competence before geometry

Accepted by direct user instruction before outcome review. The previous
`pi05_droid` direct-joint pilot used a different checkpoint, and the Cartesian
bridge applied joint conversion only after Cartesian sampling. Neither is the
`Joint` baseline defined in EmbodiSteer Appendix B.4. The new paired pilot
therefore uses one frozen `pi05_libero` checkpoint for both arms and performs
FK and damped-Jacobian lifting after every reverse denoising step in the joint
arm. The Cartesian arm remains an ordinary unmodified policy request.

All collision interventions are disabled. No L5/L6/L7 ellipsoid is built, no
clearance gradient is queried, no barrier inequality is formed, and no QP is
solved. The obstacle remains physically present so raw MuJoCo contact, CAR,
and native goal evidence are still observed. Acceptance requires one H100
allocation, exact settled-state pairing, the same query-index noise schedule,
valid decoded videos for both arms, and truthful reporting of failure or
infeasible direct-joint tracking.

This pilot adapts the paper because its policy and ours use different action
representations and generative schedulers. EmbodiSteer uses 10D chunk-start
relative Cartesian poses and a 16-step DDPM; SafeLIBERO pi0.5 uses 7D
incremental OSC commands and ten flow-Euler steps. Incremental commands are
composed into chunk-start poses for Eqs. (3), (4), and (8)--(10), then converted
back before each frozen pi0.5 query. The result may support or reject this
adaptation's competence on one primary case, but cannot be reported as an
exact reproduction of the paper's population result.

The user's L5/L6 geometry concern is accepted as a separate blocking issue for
future guidance. EmbodiSteer represents the robot with multiple link-attached
cuRobo collision spheres and aggregates the top four signed distances. One
mesh-enclosing MVEE per link can be a valid enclosure yet still be too loose,
hide local contact structure, and supply a qualitatively different gradient.
The existing MVEEs remain disabled and untrusted for paper-fidelity claims
until audited or replaced by a sphere/capsule decomposition.

The sampler regression is accepted in physical rather than bitwise units.
Interleaving simulator FK/Jacobian work between reverse steps requires ten
separately compiled Euler calls, whereas ordinary pi0.5 fuses them in one JAX
while-loop. Allocation-only diagnostics before any arm rollout measured at
most `45.767 micrometers` translation error, `0.000124 rad` rotation error,
and identical first-five gripper signs. The frozen gate is `0.1 mm`,
`0.001 rad`, `0.005` raw action units, and exact gripper-sign equality. This
calibrates numerical equivalence only; it does not change any task, geometry,
controller, or safety parameter.

## ADR-0041: Reject v1 Joint execution and correct the absolute-target adapter

Accepted after auditing clean H100 job `37055`. EmbodiSteer returns the joint
configuration chunk `Q_0`; those samples must be supplied as position targets.
SafeLIBERO's `JOINT_POSITION` controller instead interprets each normalized
input as a delta from the current joint state. The v1 adapter used a
`0.05 rad` output range and clipped `(q_target-q_current)/0.05`, causing
saturation in 220 of 300 actions and up to `0.963 rad` post-step target error.
That is not a faithful direct-joint baseline, regardless of its no-contact
outcome.

Protocol v2 keeps the same `Q_0`, checkpoint, controller gains, control
frequency, action horizon, obstacle scene, and no-guidance condition. It sets
the delta encoding range to `6 rad`, larger than the maximum Panda joint range,
so normalization represents every bounded absolute target without clipping.
Zero encoding saturation is an acceptance condition; physical PD tracking
error remains measured rather than assumed.

The v1 joint video is also rejected. All 301 frames decoded, but source pixels
became visibly corrupted, proving that shape/count validation is insufficient.
V2 executes each arm in a fresh process to isolate MuJoCo/OSMesa contexts and
checks adjacent-pixel variation on every source and decoded frame. The
threshold is fixed before v2 execution from valid Cartesian frames near
`2/255` and corrupted joint frames above `17/255`; acceptance requires at most
`8/255`. Neither correction authorizes L5/L6 geometry or safety guidance.

## ADR-0042: Align the barrier-free stress pilot with the paper's control rate

Superseded by ADR-0047.  The paper's `103 ms` / `9.61 Hz` number is inference
throughput for one guided call containing all reverse denoising steps; it does
not establish a `10 Hz` simulator control protocol.  The v3 and AEGIS-EE v1
runs remain immutable historical evidence, but their rate must not be called
the paper's control rate.

Accepted after H100 v2 job `37058` failed before a valid Joint result. Running
the two arms in fresh processes did not prevent the fixed agent-view stream
from flipping at action 10 and failing the pixel-integrity gate at action 11.
The failure is apparatus/control evidence only; it is not a task, collision,
or EmbodiSteer outcome.

The paper executes its whole 16-action joint chunk at `10 Hz`. The frozen
pi0.5-LIBERO checkpoint has a ten-action horizon, so v3 executes all ten
available actions at `10 Hz` and explicitly records the remaining horizon and
action-representation differences. It retains the exact absolute-target
adapter, checkpoint, initial state, query-seed schedule, controller gains,
primary obstacle scene, and zero-guidance condition. The primary obstacle
case is a stress test and must not be equated with the paper's obstacle-free
`Joint` baseline population.

The visual gate is extended before v3 execution to catch both high-frequency
corruption and 180-degree orientation changes. On failure, raw and processed
frames, action/joint-state history, and fixed-camera poses are atomically
retained. No failed visual run is promoted to scientific evidence, even if
its MP4 decodes.

## ADR-0043: Eliminate the second OSMesa environment from joint denoising

Accepted from the fail-closed trace of H100 job `37061`. The first direct
joint target was well behaved: zero encoding saturation and at most
`0.299 rad/s` observed velocity. The fixed MuJoCo camera transform remained
bitwise identical, but the source image became an almost exact 180-degree
rotation immediately after action zero. This rules out the sampled target or
physical camera motion as the cause of that frame.

The remaining context violation was the separate rendered environment used
only for kinematic queries inside the Joint worker. The repair reuses the
live MuJoCo model for FK and Jacobians, then restores the full flattened state
and requires bitwise equality before action execution. This preserves exact
kinematics while avoiding a second OSMesa context. It is an apparatus repair,
not a change to EmbodiSteer equations or the baseline protocol.

## ADR-0044: Scale the FK-pose serialization gate by float32 precision

Accepted after H100 job `37062` passed the renderer repair but stopped on a
fixed `1e-7` comparison between a float64 FK action and the same action after
server-side normalization, float32 model storage, and affine decoding. The
comparison is an apparatus-integrity check; it does not constrain safety or
task behavior.

The replacement bound is `32 * eps_float32 * max(1, |input|, |decoded|)`.
It scales only with IEEE float32 precision and the serialized value magnitude,
and is frozen before observing its live error. Each denoising step records the
error and bound. The policy input, denoising update, joint trajectory, control
rate, execution horizon, and all acceptance outcomes remain unchanged.

## ADR-0045: Reject the pi0.5 direct-Joint adaptation as a matched baseline

Accepted after validated H100 job `37067`. The apparatus gates passed: one
frozen checkpoint and settled state, no collision guidance, zero joint-target
encoding saturation, bitwise state restoration after live-model kinematics,
precision-valid sampler round trips, and two valid 301-frame videos. The
negative outcome is therefore retained rather than repaired by more tuning.

Neither arm completed the primary obstacle task. Cartesian contacted link 6
at step 193; Joint contacted link 5 at step 79. Joint target tracking remained
poor (`0.255 rad` mean and `1.225 rad` maximum error) and aggressive (maximum
`3.503 rad/s` observed velocity), despite exact target encoding. The result
does not reproduce the paper's competence-preserving `Joint` baseline.

This does not falsify EmbodiSteer. Its released description assumes a 10D
chunk-start relative pose diffusion policy, a 16-action chunk, a native joint
execution interface, and reports `Joint` competence without obstacles. Our
frozen pi0.5-LIBERO checkpoint produces ten incremental 7D OSC flow actions
and the primary SafeLIBERO stress scene retains its physical obstacle. Future
EmbodiSteer safety work is blocked until a competent Joint baseline is
established with a natively compatible policy/controller. If geometry is
reintroduced afterward, use audited link-attached spheres/capsules and
top-critical-distance aggregation; do not present one MVEE per L5/L6 link as
the paper's geometry.

## ADR-0046: Isolate the released AEGIS end-effector constraint

Accepted as the next bounded diagnostic.  L5/L6/L7 ellipsoids and their CBF
rows are disabled, not refit.  Both controller arms receive exactly the
released Table-1 end-effector proxy and archived E05 obstacle MVEE.  Reusing
the archived MVEE is valid here because the evaluator must reproduce its
settled simulator-state and camera hashes before control; the completed Table
1 result is never modified or reinterpreted.

For Cartesian EE, the released six-variable translational CBF-QP is executed
unchanged, including its virtual direction state and `0.05 s` internal update.
For Direct Joint, the same QP is evaluated on the joint trajectory's
equivalent Cartesian translation and only its translation correction is
lifted through the live damped EE Jacobian into the joint target.  This tests
whether changing the downstream controller representation helps the original
AEGIS constraint.  It is not a claim of an exact EmbodiSteer implementation,
native joint-space barrier, full-body protection, or matched paper baseline.
Raw simulator contacts and task state remain authoritative over QP clearance.

H100 attempt `37077` showed that exact physical-state equality does not imply
byte-identical RGB after that state is transplanted into a fresh 10 Hz render
context.  Requiring the new rendering to equal the old perception input was
therefore the wrong provenance check: the new run does not refit perception;
it consumes the archived numeric MVEE.  The revised fail-closed binding keeps
the exact 170-value simulator-state hash and checks the active obstacle world
position against the archived Table-1 value within `1e-9 m`.  Old and current
camera hashes are both retained, with mismatch expected rather than hidden.

The validated outcome from H100 job `37080` rejects both possible optimistic
interpretations.  The Cartesian EE arm completed the task, but L5/L6 contact
and CAR occurred while every single-row QP satisfied its optimizer
constraint.  At first protected contact, released barrier `h=-0.003259` and
the QP applied a `0.245590` action-unit translation correction; optimizer-row
feasibility was not physical-link safety.  Direct Joint contacted L7 at step
27 with `h=-0.054550`, later contacted L5/L6/L7, and never progressed the
native goal despite 245 nontrivial lifted corrections.

The accepted conclusion is limited but decisive for this diagnostic: changing
from OSC to the current Direct-Joint adapter cannot make an end-effector-only
barrier protect unmodeled links.  It also adds controller-tracking and
linearized-Jacobian error; its maximum lifted correction was `0.453 rad` and
mean target tracking error was `0.289 rad`.  No L5--L7 geometry was enabled,
so this run does not evaluate or rehabilitate the earlier distal ellipsoids.

## ADR-0047: Restore released AEGIS timing and classify Direct Joint correctly

Accepted after re-auditing the paper and validated H100 jobs `37083` and
`37084`.  The released AEGIS reproduction runs at `20 Hz`, executes five of
the ten predicted actions per query, and uses an internal QP step of `0.05 s`.
The corrected AEGIS-EE v2 protocol restores those values.  AEGIS-EE v1's
`10 Hz` simulator step was inconsistent with its unchanged `0.05 s` safety
model and is not the primary result.

EmbodiSteer's `Joint` baseline means joint-space denoising with FK calls to the
frozen Cartesian denoiser and damped-Jacobian residual updates, with collision
guidance disabled.  Our no-guidance arm implements that structural pattern and
the published heuristics `alpha=0.1`, pseudoinverse damping `0.001`, and joint
update clip `0.5 rad`.  It is still only an adaptation: pi0.5-LIBERO supplies
ten incremental 7D OSC flow actions rather than the paper's chunk-start pose
DDPM representation and the targets are executed through SafeLIBERO's
`JOINT_POSITION` controller.  The arm named `joint_denoising_with_aegis_ee`
adds a post-hoc EE correction and therefore is not the paper's `Joint`
baseline.

At restored timing, no-guidance job `37083` validated both 300-action videos;
neither arm completed the task.  Direct Joint contacted L5 at step 97 and had
mean/max target tracking error `0.218/0.978 rad`.  In job `37084`, Cartesian
AEGIS completed at step 231 but contacted L5 and failed CAR at step 196.
Direct Joint + AEGIS avoided L5--L7 contact in that rollout but contacted the
hand/finger at step 15, failed CAR at step 16, displaced the obstacle by
`0.034719 m`, and never completed the task; tracking error remained
`0.183/0.701 rad`.  Thus the timing repair does not establish a competent
Direct-Joint baseline or successful safety method.  The remaining mismatch is
architectural, not a missing scalar heuristic from the paper.

## ADR-0048: Retain the released EE proxy and use exact hull slabs for L5--L7

Accepted after hand review of the H100 geometry renders. The AEGIS
end-effector proxy remains the released site-attached ellipsoid with semiaxes
`[0.06, 0.12, 0.11] m`; changing it would confound the requested comparison.
The additional L5--L7 bounds are opt-in geometry and do not reinterpret or
modify the completed Table 1 results.

The common-apex facet partition from job `37086` is mathematically enclosing
but rejected as the final geometry because one L5 part remained nearly as long
as the original one-link MVEE. The replacement partitions the convex hull into
contiguous dominant-axis slabs and encloses each exact clipped polytope. This
gives a formal no-gap union certificate while allowing each ellipsoid to
remain local to one short section of the physical link.

H100 job `37087` passed the allocation numeric gate and produced the accepted
seven-part render: three L5, two L6, two L7, plus the unchanged cyan AEGIS EE
proxy. No active controller or QP is changed by this decision. Geometry must
be promoted under a separate preregistered control protocol before any safety
claim is evaluated.

## ADR-0049: Test learned weights on controller-pulled physical repulsion

Accepted as the simplest decisive follow-up to the failed unconstrained MLP
gradients. The learned object is a monotone scalar safety potential over the
seven accepted L5--L7 rollout margins. Exact paired cloned-OSC perturbations
provide the local Cartesian pullback. Consequently its action gradient is a
nonnegative combination of physical clearance-increase directions; learning
chooses strengths but cannot reverse all geometry normals.

The design explicitly separates evidence for a physical repulsive mechanism
from deployability. It uses privileged online clone rollouts and only one
temporally adjacent E05 test state. It therefore cannot establish episode or
task generalization, real-time performance, or SafeLIBERO completion. Those
claims remain blocked even if the local direction passes.

The MLP is frozen from steps 182--184 before test step 185. Learned, fixed
soft-min and paired-random directions are evaluated at equal correction norm.
The learned force must improve exact clearance, outperform the fixed physical
field, and beat the matched random population. If it does not, the correct
conclusion is that learning adds no evidence beyond analytical repulsion. A
two-action receding continuation is conditional on the direction gate and is
not a substitute for full closed-loop task completion.

H100 jobs `38789` and `38791` now reject learned monotone weighting under this
contract. The learned direction beat every matched random direction, so the
physical controller-pulled repulsive basis is meaningful. But it was almost
the same as the fixed analytical field and slightly worse at both registered
radii. The local two-action region also contained no exact-safe candidate;
its best learned hard margin remained `-6.801919 mm`.

The accepted interpretation is not “repulsion failed.” It is that the MLP
learned an already available analytical direction and cannot compensate for
missing safe support. Do not enlarge the MLP, add a QP, or run closed loop
from this result. First establish exact safe support using fixed physical
repulsion under an independently preregistered earlier-intervention or
larger/iterative trust-region experiment. A learned task-conditioned force is
only motivated after that oracle support gate passes.

## ADR-0050: Test fixed repulsion inside denoising before the E05 boundary

Accepted as the smallest follow-up to ADR-0049. The experiment reuses the
analytical controller-pulled L5--L7 direction that already beat matched
random steering. It intervenes from the pi0.5 query at action 180 and targets
chunk slots 2--4, which execute as actions 182--184. Guidance is injected
inside only the final five Euler updates and is bounded in physical normalized
action units.

An equal-budget post-hoc arm is mandatory. Without it, any improvement could
be attributed merely to adding the same vector to the completed action rather
than to denoising interaction. Ordinary pi0.5 remains unchanged unless the
reserved opt-in envelope is present, and physical displacements are converted
to model coordinates by scale only. The arm executes only after exact cloned
OSC verification. Passing would establish local early fixed-force support;
it would not establish learning, generalization, invariance, or task success.

The initial-query `0.005` action tolerance from job `37054` is not a valid
late-query gate. After attempt `38806` exposed that unsupported extrapolation,
late live-versus-archived action differences are diagnostic only. This does
not weaken the paired comparison: ordinary and guided denoising are evaluated
from the same exact replayed state and live observation with the same explicit
RNG seed, while post-hoc repulsion is constructed from the ordinary live
chunk. No outcome or candidate clearance was observed before this correction.

Job `38808` rejects the specific hypothesis that a fixed force becomes more
effective merely by adding it during final pi0.5 Euler updates. It did improve
the ordinary exact proxy clearance, but less than adding the same registered
budget after denoising. The denoiser partially canceled the injected vector.
This is not evidence against analytical repulsion, and all three prefixes
were already positive-margin through action 184. Do not claim that the later
E05 collision was prevented: the comparative gate intentionally blocked
execution and no suffix was run.

Independent job `38809` validates this interpretation. Keep the result as a
NO-GO for fixed-vector injection during denoising and as apparatus evidence
that a reserved flow hook works. It is not evidence that early intervention
prevents the archived E05 collision, because the current live pi0.5 query did
not reproduce that archived query's dangerous action mode.

## ADR-0051: Establish the structured physical-field oracle ceiling first

Accepted after the denoising-insertion NO-GO. The dangerous archived actions,
not a new live policy sample, are authoritative for this gate. Post-hoc exact
cloned-OSC search separates three questions: whether any bounded two-action
repair exists, whether positive mixtures of physical clearance normals can
express one, and whether adding task-tangent motion preserves useful intent.

The tangent is not an arbitrary avoidance vector. It is the nominal end-
effector progress gradient projected into the nullspace of active safety
rows. Nonnegative normal weights retain the physical push-away interpretation.
Exact cloned rollouts, raw contact, episode-relative CAR, and end-effector
progress decide outcomes; affine derivatives only generate directions and
are recomputed after every iteration. Do not train an MLP unless a structured
arm finds a verified-safe, task-progressing repair. Do not call this online
control or task completion even if it passes.

## ADR-0052: Accept structured field expressivity, not online efficacy

Accepted after H100 producer `38834` and independent validator `38838`.
Both registered structured arms repaired the exact archived `-9.313104 mm`
two-action transition without raw L5--L7 contact or paper CAR and retained
more than 82% of nominal end-effector progress. The structured field gate
therefore passes: a positive mixture of controller-pulled physical clearance
normals is expressive enough, and adding the projected task tangent remains
compatible with the verified-safe repair.

This is not yet an online-control result. The smallest retained iterative
structured repairs have correction L2 about `0.89` and only micrometre-scale
positive margins. At matched one-shot norms, no arm is safe below radius
`1.0`. Accordingly, do not promote fixed inside-denoising injection, train an
unconstrained action predictor, or run an unverified closed loop from this
gate. The justified next learned object is only the structured mixture
coefficient/step selector, trained across boundary states and always compared
with fixed analytical mixtures and matched random directions. A positive
buffer plus fresh exact cloned-OSC verification remains mandatory before any
candidate executes.

## ADR-0053: Compare denoising timing by surviving correction norm

Accepted as a mechanism-only follow-up to the earlier uniform-final-five
inside-flow NO-GO. Equal injected forces are not equal interventions because
later policy vector-field updates can cancel an earlier perturbation. The
scientifically paired comparison therefore measures the final output XYZ
correction over the same three chunk slots and constructs a separate post-hoc
arm at that exact surviving L2 norm.

Three schedules isolate timing while holding the injected physical budget
fixed: uniform updates 5--9, a late ramp over updates 8--9, and update 9 only.
The opt-in OpenPI path accepts an explicit nonnegative ten-step schedule and
leaves the ordinary sampler unchanged. This test may identify reduced
cancellation; it cannot establish safety efficacy because the paired live
step-180 action is already safe and is not the archived dangerous mode. No
candidate executes regardless of outcome.

## ADR-0054: Accept late timing for cancellation, retain post-hoc for efficacy

Accepted after producer `38839` and independent validator `38840`. Delaying
the same physical injected budget from uniform updates 5--9 to updates 8--9
and then update 9 raised final correction survival from `41.01%` to `78.85%`
and `97.11%`. This directly validates denoiser cancellation as the mechanism
behind the earlier weak inside-flow correction.

Do not interpret this as a safety solution. Every compared live prefix was
already more than 54 mm proxy-safe and no action executed. At equal surviving
correction norm, the final-step arm was `0.228 mm` worse than post-hoc, which
is consistent with the two becoming equivalent as timing approaches the
completed sample. Retain late scheduling only if future work specifically
requires modifying intermediate flow states; otherwise, post-hoc exact-verified
correction is the simpler and strongest baseline for the archived collision.

## ADR-0055: Learn a local physical action field from paired long-horizon outcomes first

Accepted as the smallest mechanism test for the proposed rollout-supervised
repulsive field. The geometry backend remains the accepted seven L5--L7
ellipsoid slabs; Poisson is deliberately deferred. The learned object is not a
joint trajectory or collision classifier. It is the local action-space
direction that increases a quantitative long-horizon execution utility.

The task-completing five-action detour run supplies a fixed open-loop
continuation through its later action-197 collision. Pairing both signs of a
five-action endpoint-preserving correction against that identical continuation
allows the learned field to see later physical consequences while avoiding
policy-query noise. A separate short-horizon analytical ellipsoid field and
matched random directions determine whether counterfactual learning adds
information beyond ordinary repulsion or abundant safe directions. This is an
offline, single-state expressivity test; exact cloned OSC is label generation,
not the proposed deployment evaluator. Do not train an MLP or modify pi0.5
unless this strict mechanism gate passes.

## ADR-0056: Accept learned long-horizon direction, reject the small local safety mechanism

Accepted after producer `38929` and independent validator `38932`. Paired
counterfactual long-horizon outcomes do teach information that the immediate
analytical ellipsoid field misses: held-out directional Pearson was `0.9973`,
the learned correction improved exact long-horizon clearance while the fixed
field worsened it, and it beat all 32 matched random directions at the
registered norm. This supports the narrow hypothesis that rollout supervision
can learn how a five-action change affects a later physical collision.

It does not support collision avoidance. The action-197 contact remained,
hard clearance improved by only `0.248 mm` from an initial `-14.263 mm`, and
no sampled endpoint-preserving correction at L2 `0.1` was safe. Classify the
experiment as a strict safety NO-GO and do not train an MLP merely because the
directional metrics passed. First find a verified-safe member of a richer but
still task-preserving correction family. Only then is learning that field from
multiple states scientifically justified.

## ADR-0057: Diagnose pure risk, endpoint constraint, and safe support separately

Accepted before the decisive follow-up. Remove the task penalty only from the
field target, not from final task-preservation evaluation. Re-estimate the
field after every accepted correction because the worst future link/time
witness is nonsmooth and may switch. Include radius `1.0`, since earlier
structured-field evidence required correction L2 near `0.89`.

A no-learning derivative-free search is essential. Without it, another unsafe
learned path cannot distinguish a bad gradient from an empty candidate family.
Run exact endpoint preservation first; relax it only after the exact-support
search fails, so the result can identify whether the zero-sum condition caused
the failure. This remains an offline local oracle diagnostic inspired by policy
value learning, not a PNCBF or formal safety result.

## ADR-0058: Reject fixed-tail five-action task preservation, retain the learned direction mechanism

Accepted after producer `38955` and validator `38960`. Pure-risk refitting did
not materially change the learned direction, and the fixed analytical field
was correctly signed over its local five-action horizon but harmful over the
twenty-action outcome. The earlier result was not caused by the task penalty
or a repulsion-sign bug.

Exact endpoint-preserving search found no safe support even at correction L2
`1.0`. Removing endpoint equality allowed multiple zero-contact, CAR-free,
positive-clearance trajectories, but every such retained candidate displaced
the terminal end effector by more than 15 mm. Thus, avoid claiming that the
five-action action space lacks geometric avoidance. The missing component is
task-compatible rejoining beyond that five-action prefix. The next oracle
should provide a longer rejoin horizon or live receding policy feedback; do not
scale direct-field learning against this fixed unsafe tail.

## ADR-0059: Reject greedy immediate-clearance route selection

Accepted after execute-five producer `39048`, validator `39051`, route-value
array `39053`, and validation array `39065`. Selecting the smallest currently
safe route at action 182 led to a state at action 187 for which the registered
oracle found no buffered continuation. Larger right/retreat routes remained
collision-free through the time limit but never made native task progress.

This is direct evidence that route selection cannot be based only on current
clearance or correction norm. A route-conditioned continuation value is
scientifically motivated because it must distinguish future feasibility and
task compatibility. Do not respond by increasing the repulsive gain or by
training on a policy whose tie-breaking is not fixed.

## ADR-0060: Accept collision-free task completion as an oracle positive control, not a robust pass

Accepted after complete-compound producer `39073` and validator `39079`. The
registered compound trajectory completed E05 at action 205, produced no
protected contact or paper CAR, and retained positive L5--L7 ellipsoid
clearance. This proves the action space and frozen OSC contain a physical
collision-free task-completing trajectory for the primary case.

Retain the strict NO-GO because its minimum was only `+0.167720 mm`, below the
registered `+1 mm` buffer. Correction-scale sweeps `39082/39096`, independently
validated by `39090/39101`, show that even `+1%` scaling changes the downstream
mode and causes later collision/task failure, while slightly lower scales lose
positive clearance. The compound is a narrow open-loop positive control, not
a robust policy and not training authorization.

## ADR-0061: Intervene before the terminal corridor; terminal action authority is insufficient

Accepted after the two-action terminal authority search `39115` and validator
`39116`. Starting from the exact compound state before action 204, a 26-direction
grid over radii through `1.0` found many goal-satisfying, zero-contact actions,
but the best task-preserving internal margin was only `+0.667327 mm`. Starting
at action 205 was even more constrained because the initial cloned state was
already below the buffer.

Do not weaken or retroactively redefine the `+1 mm` gate. A robust oracle must
act earlier and evaluate route-conditioned continuation/task value before the
state enters this narrow terminal corridor. Learning remains blocked until
that reproducible oracle supplies buffered safe task completion.

## ADR-0062: Activate robust correction at buffered warning, not contact prediction

Accepted after paired H100 producer `39141` and validator `39143`. The
scale-`1.01/1.02` compound states were still physically contact-free at action
205 and missed task success only by the native 30 mm horizontal `On` threshold.
Fresh frozen-pi0.5 feedback did not recover the task. Exact five-action OSC
lookahead warned about the `+1 mm` buffer immediately at action 206 and later
failed closed before executing predicted protected contact/CAR at actions 231
and 211, respectively.

Thus the robust architecture should retain receding predictive evaluation,
but a detector alone is insufficient. Trigger route generation at the first
buffer warning, not at raw-contact prediction. The backup policy must preserve
route memory and choose among task-compatible alternatives using a fixed
continuation value; simple live VLA continuation and larger repulsion have
both failed. Continue to block policy-value MLP training until this oracle
policy completes buffered-safe E05.

The same audit also records that identical simulator state and registered
policy seed can yield materially different pi0.5 action chunks across server
instances. Treat policy randomness as more than an integer seed: freeze action
tensors for matched mechanism comparisons, or evaluate a registered sample
set/quantile for future policy-value claims.

## ADR-0063: Retain repulsion as a transferable local mechanism, reject fixed-suffix generalization

Accepted after producer array `39149` and independent validator `39152`.
Across three previously unused Table-1 collision cases, the smooth
counterfactual field increased internal L5--L7 ellipsoid clearance in every
case, but satisfied the zero-contact/CAR/+1 mm gate only in goal-II E00.
Therefore the evidence supports a transferable local clearance direction,
not generalized collision prevention under one five-action edit followed by
an immutable unsafe suffix.

Do not combine the two failed cases into a generic model error. Goal-II E24
had a proxy-valid warning state. Smooth correction improved but did not clear
the first five actions; analytical repulsion cleared those actions, but the
later unchanged continuation remained unsafe. Test both stronger/multimodal
prefix proposals and receding intervention there.
Spatial-I E12 already violated the frozen obstacle-MVEE proxy by `264 mm`
without raw contact at the intervention state; first audit proxy/contact
consistency and select a proxy-valid warning state. Do not train a repulsive
MLP, weaken the +1 mm gate, or claim population generalization from this
outcome-conditioned three-case pilot.

## ADR-0064: Require task-valid prevention states before scoring safety efficacy

Accepted after auditing the raw Table-1 task ledgers. Goal-II E24 cannot count
against repulsion efficacy: both pi0.5 and AEGIS failed the native task, the
goal was never satisfied, and the bowl was already about `0.301 m` from the
AEGIS end effector by the registered intervention. Spatial-I E12 also remains
ineligible because its obstacle-MVEE proxy was already deeply negative without
raw contact. Preserve both as diagnostic mechanism evidence, but remove them
from any task-preserving success-rate denominator.

The next focused case is goal-II E38, where both policy arms completed the
task, the cream cheese remained coupled to the gripper before intervention,
and AEGIS subsequently made raw L5 contact with the wine-bottle obstacle.
Keep the repulsion algorithm and correction budgets frozen. Eligibility must
be checked before optimization; failure of that check is an invalid case, not
a failed controller.

## ADR-0065: Separate physical contact removal from conservative proxy certification

Accepted after clean H100 producer `39162` and independent validator `39166`.
Goal-II E38 passed every preregistered task-validity and prevention-state gate.
Smooth counterfactual repulsion improved the twenty-action ellipsoid minimum
from `-52.880546` to `-36.087801 mm` and removed all 115 protected MuJoCo
contact samples; analytical and bounded derivative-free arms also removed raw
contact. Nevertheless, every corrected arm remained negative under the frozen
ellipsoid/MVEE proxy, so the registered zero- and +1 mm gates fail.

Record this as transferable contact-removal mechanism evidence, not certified
collision prevention and not task-preserving success. A stronger force is not
the immediate conclusion: physical contact is already absent in the evaluated
window, while the conservative proxy remains negative. Also do not infer that
E24 was repaired or caused by the filter; it is excluded because the original
VLA had already lost the bowl and both baseline arms failed the task.

The next experiment should retain E38, extend evaluation through native task
completion with receding intervention, and keep two authorities explicit:
compiled MuJoCo protected contact for physical collision evaluation, and the
ellipsoid margin for conservative optimization/certification. Do not collapse
them into one success label or report an efficacy rate from the two eligible,
outcome-conditioned cases.

## ADR-0066: Recede over the immutable E38 task policy and separate authorities

Accepted before execution. Reuse the exact validated smooth-field estimator,
action bounds, correction radius, and twenty-action lookahead. Recompute after
each executed prefix and trigger below the registered `+1 mm` ellipsoid warning.
Use the immutable archived AEGIS action ledger so changes in task outcome can
be attributed to repulsion rather than a new stochastic pi0.5 sample.

Physical success requires zero protected MuJoCo contact and CAR at most 1 mm
through every internal substep. Ellipsoid zero/+1 mm results are reported as
separate conservative certification gates. Native BDDL completion is a third
authority. No result may be described as full success unless physical safety
and task completion both pass; proxy failure must still be reported even if
physical contact is absent.

## ADR-0067: Accept E38 physical safe task success; reject proxy certification

Accepted after producer `39181` and independent validator `39185`. Repeated
warning-state smooth repulsion completed the native E38 task at action 120
with zero protected contact at every internal MuJoCo substep and CAR pass.
This validates the immediate hypothesis that a receding physical repulsive
field can remove this task-valid L5 collision without inherently destroying
task behavior under the frozen archived continuation.

Do not call the method ellipsoid-safe or certified. Its executed minimum proxy
clearance remained `-31.577525 mm`; both registered certification gates fail.
Do not increase repulsion merely to satisfy that proxy: every corrected window
already had zero physical contact, and correction norms were near the maximum.
First audit why the robot/obstacle proxy remains deeply overlapping after raw
collision disappears. Also retain the limitation that this is archived-policy
receding execution, not live stochastic VLA feedback or population evidence.

## ADR-0068: Reject one obstacle MVEE as the E38 certification authority

Accepted after matched H100 producer `39224` and independent validator
`39225`. Holding the successful action trajectory and seven L5--L7 robot
ellipsoids fixed, the released perceived MVEE reported `-31.577525 mm`, while
a privileged MVEE fitted directly to the 21 compiled collision boxes still
reported `-20.285566 mm`. Exact solid intersection with the compiled box union
and raw MuJoCo contact both reported zero violations across all 326 matched
samples.

Interpret this as a proxy-representation failure, not solely a perception
failure. Perception materially worsens the proxy: its MVEE volume is 8.13969
times the compiled-vertex MVEE and its center differs by 128.831 mm. But one
ground-truth enclosing MVEE remains falsely unsafe because it fills empty
space, and the support-gap calculation adds further conservatism. Therefore do
not solve E38 by increasing repulsive magnitude until the single MVEE becomes
positive, and do not train against that sign as physical collision truth.

For the simplest next learning pilot, keep the accepted robot ellipsoids but
replace the obstacle with multiple tight primitives/fields and retain raw
MuJoCo contact as the physical evaluation authority. The exact box-union audit
is privileged simulation evidence; it validates the diagnosis but is not a
deployable perception solution or a population safety claim.

## ADR-0105: Evaluate the fixed backup before learning its policy value

The validated E05 job 39384 establishes a useful safety-only backup outcome:
eight warning-triggered normal-repulsion windows kept every executed prefix
above the registered 1 mm L5--L7 proxy buffer with zero protected contact and
CAR pass. It did not complete the task because its frozen proposal ledger
ended. This is sufficient to test PNCBF-style policy evaluation, but not to
claim task-compatible control or a neural CBF.

Freeze the exact augmented policy: simulator/controller state, proposal-ledger
index and tensor, warning latch, fixed candidate magnitudes, deterministic
tie-breaking, and fail-closed selection. At ledger exhaustion, verify a
zero-motion hold for 25 actions; if it is unsafe, use the preregistered normal
retreat. Record all seven per-row internal-substep clearances and construct
exact finite-horizon suffix maxima with positive values unsafe. Require the
discrete Bellman recursion, all values nonpositive, physical contact/CAR pass,
and a verified terminal tail. Do not train, add a QP, or claim infinite-horizon
invariance from this gate.

## ADR-0106: Learn policy value from executed-policy traces, not candidate clones

Producer `39410` exposed a 0.255712 mm successor-boundary mismatch and
validator `39411` localized it to cloned candidate traces versus the actually
executed primary policy. A PNCBF-style recursion must evaluate one trajectory
of one fixed policy. Therefore record all seven clearances at every primary
MuJoCo substep during execution and build suffix maxima only from those
traces. Keep cloned OSC rollouts solely for rejecting unsafe candidates before
execution. Do not mix the two measurement streams or relax the `1e-12`
successor-consistency gate.

## ADR-0107: Authorize policy-value approximation, not learned control

Accepted after producer `39422` and independent validator `39424`. Exact
finite-horizon suffix values for the fixed augmented backup are well-defined
and safe at all 22 recorded decision states. Executed-policy internal traces,
not cloned candidate traces, are the authoritative value labels. This clears
only the next prediction experiment: approximate the seven policy-conditioned
values from grouped backup-policy rollouts and test false-safes, recall, and
action gradients on untouched states. It does not clear a QP, online learned
filter, neural-CBF/invariance claim, or task-completion experiment.
## ADR-0108: Audit a pure backup before learning direct action risk

Accepted for H100 evaluation after reviewing the uploaded mathematical system
specification. Preserve jobs 39422/39424 as valid exact finite-horizon values
for the registered augmented policy. Do not reinterpret them as values for a
VLA-ledger-independent backup.

Activate one representative-state audit at E05 steps 187, 222 and 288. The
validated action ledger may reconstruct the physical/controller state, but it
is forbidden from the backup action and score. Use a fixed finite family of
hold, world-axis and geometry-local candidates, one-action execution, and a
25-action hold tail. Require deterministic/order-invariant exact evaluation,
no source-state mutation, seven-row +1 mm support, zero protected contact, and
paper CAR. Do not train an MLP, add a QP, or claim a neural CBF/invariant set.

Accepted after H100 producer `39443` and independent validator `39444`. The
pure backup has exact verified support at all three representative states, and
fresh replay error is zero. The next permitted scientific gate is broader
policy-consistent rollout collection using this same fixed backup definition,
with `SAFE_TERMINAL`, `UNSAFE`, and `UNKNOWN` kept distinct. Do not train a
direct action-risk model until the complete pure-backup oracle demonstrates
adequate recoverable-state support over grouped episodes. The augmented-policy
jobs 39422/39424 remain valid evidence but are not pooled with pure-backup
labels.

## ADR-0110: Expand boundary coverage before direct action-risk learning

Accepted after the E05 Monte-Carlo target positive control. Use real pi0.5
five-action query boundaries, identical simulator/controller snapshots, the
unchanged 37 structured normal/tangent candidates, and the complete fixed
ledger-independent backup. Retain safe, unsafe, and timeout outcomes
separately. Require initially safe states, mixed exact safe/unsafe candidate
support, no ellipsoid-safe physical collision, and report safe, unsafe,
near-boundary, active-witness, state, and episode counts for every claimed row.
Unsupported rows are reported rather than populated with easy far-clearance
samples. Reserve complete test episodes and forbid learning, QP, calibration,
and closed loop until this coverage gate passes.

H100 canary `39652` passed and reproduced E05. Array `39657` exposed a
20-minute allocation limit that censored long complete-backup rollouts before
atomic output. Extend only the Slurm wall limit to 60 minutes; do not modify
the candidate family, backup, target, terminal semantics, or acceptance gate.
Uniform array `39669` and dependent independent validator `39674` are the
authoritative population run. Any proxy-safe MuJoCo/CAR violation remains a
state-level NO-GO and cannot become a safe training label.

## ADR-0112: Freeze the grouped coverage snapshot; reject seven-output training

Accepted after uniform H100 array `39669` and independent H100 validator
`39701`. Preserve all 15 validated state artifacts and the immutable
complete-episode split, but mark the snapshot
`coverage_no_go_not_for_training`. Unknown backup timeouts are censored and
excluded from the candidate manifest; proxy-invalid and no-safe states remain
in the state manifest but cannot contribute learning examples.

Only rows 0 (L5 slab 0) and 3 (L6 slab 0) satisfy every preregistered boundary
coverage threshold. Row 1 lacks a third train active-witness episode, row 2
has no active witnesses, and rows 4--6 have no meaningful unsafe/active
support. E10 is retained as a diagnostic proxy-invalid case with 20
proxy-safe physical vetoes. Three states contain no exact-safe candidate.
These are distinct population outcomes and must not be repaired by relabeling,
dropping timeouts, or filling rows with far-safe examples.

Do not train the seven-output action-risk MLP, add calibration, enable a QP,
or run closed loop from this freeze. Retain the exact real-query candidate and
complete ledger-independent backup protocol. The only authorized next gate is
targeted grouped collection for L5 rows 1--2, L6 row 1, and both L7 rows,
followed by the same independent coverage validator. Reserved test episodes
remain unlabeled and untouched until the training-coverage gate passes.

## ADR-0113: Replace equal active-witness coverage with row identifiability

Accepted after H100 producer `39725` and independent H100 validator `39726`.
Preserve the grouped v1 freeze as an immutable learning NO-GO, but do not
require each safety row to become the global active witness. A row is useful
when the audited executable action set contains a robust safe/unsafe crossing
and the safe side is globally feasible across all seven rows and physical
verification. Independent-violation witnesses diagnose empirical domination;
their absence is not proof of mathematical redundancy.

On the frozen train/validation actions, rows 0--3 have useful globally safe
two-sided boundaries. Row 2 has no independent-violation witness and is only
empirically dominated on this finite audited set. Rows 4 and 6 vary but do not
cross the robust boundary; row 5 crosses only in a state with no globally safe
candidate. Do not train or resume broad passive collection. The only
authorized next gate is a bounded active search for rows 4--6 using executable
five-action candidates, the unchanged complete backup, and the same physical
verification. If useful boundaries remain absent, resolve the safety scope or
expand the deployment distribution rather than manufacture balanced rows.
Reserved test episodes remain untouched.

## ADR-0114: Restrict the first learned claim to L5, retain seven-row vetoes

- Status: accepted; prediction gate validated NO-GO
- Date: 2026-08-14

For the first feasibility test, learn only L5 rows 0--2. Rows 3--6 are outside
the learned claim and loss, but every accepted action must still pass all seven
exact ellipsoid rows, protected MuJoCo contact, and paper CAR. This tests the
actual E05 L5 mechanism without manufacturing L6/L7 learning support or
claiming learned whole-arm safety.

The immutable grouped dataset passes the restricted coverage gate (H100 job
`39759`), so the three-output MLP experiment is scientifically interpretable.
The frozen model nevertheless fails: producer `39778` reports 29 held-out L5
false-safes despite 75% exact-safe recall and support in all 3/3 recoverable
validation states. Independent H100 validator `39781` reproduces all
predictions and metrics with zero error. Therefore the current compact
state/action representation plus plain regression MLP is rejected for safety
filtering even at the narrowed L5 scope.

Do not soften the zero-false-safe gate, add a QP, open reserved test episodes,
or run closed loop. Rows 3--6 remain physical vetoes, not ignored hazards. The
next scientific intervention must address grouped-state prediction
generalization/representation while preserving the frozen labels and splits;
additional epochs or a larger correction cannot repair this validated
false-safe failure.

## ADR-0115: Make L6/L7 transfer diagnostic in the L5-only feasibility claim

- Status: accepted for the next control gate; prediction remains blocked
- Date: 2026-08-14
- Supersedes: ADR-0114 only for the future action-acceptance rule

Narrow the first control claim to **learned L5 collision avoidance under the
released AEGIS end-effector constraint**. The learned decision has exactly
three risk outputs (L5 rows 0--2). L6/L7 rows are not learned constraints and
do not reject an action in this scoped feasibility arm; instead, their exact
ellipsoid traces and MuJoCo contacts are mandatory diagnostic outcomes. Report
accepted-action L6/L7 contact as cross-link collision transfer, not as a false
safe of the stated L5 predictor and not as whole-arm safety.

The existing grouped artifacts cannot yet establish AEGIS EE compatibility of
a corrected candidate. They store actions in released-AEGIS output coordinates,
but the structured L5 residual is applied afterward and no released EE-QP
input/output record is stored for that modified action. Do not infer that a
modified candidate remains AEGIS-compatible merely because its nominal parent
was filtered. The future gate must record the original EE-QP check on the final
candidate and bind the learned L5 prediction to the exact final executed action.
If the EE filter materially changes the candidate, recompute L5 risk on that
output or reject it; never evaluate one action and execute another.

Report separately: (1) L5 success (final action predicted L5-safe, AEGIS EE
compatible, and no evaluated L5 contact); (2) L6/L7 cross-link transfer among
accepted actions; and (3) CAR/task outcomes as diagnostics. Keep the terms
`whole-arm safety filter`, `L5--L7 safety`, and `collision-safe VLA` forbidden.
This protocol change cannot rescue the frozen model from jobs 39778/39781,
because its 29 held-out L5 false-safes fail before the AEGIS or cross-link
decision. QP and closed-loop execution remain unauthorized.

## ADR-0116: Diagnose action interpolation versus state generalization

- Status: accepted; independently validated strict NO-GO
- Date: 2026-08-14

Before collecting or retraining a deployment model, use the immutable grouped
dataset for one matched no-new-simulation diagnostic. Fit the unchanged 86D,
three-output architecture, symmetric loss, optimizer, seed, and 1,200-epoch
schedule only on nominal/constant-profile actions from the existing training
episodes. Evaluate the same frozen model on front-loaded actions in two sets:

1. Test A: held-out actions at the same training states.
2. Test B: the identical action family at disjoint validation episodes/states.

Use the same zero L5 false-safe, at least 50% L5-safe recall, and support in
every recoverable state gates for both. If Test A passes and Test B fails,
classify state generalization/coverage as the primary observed failure. If
Test A also fails, action interpolation, representation, or model capacity
remains implicated. This diagnostic opens no reserved test episode, runs no
simulation, and cannot establish AEGIS compatibility or authorize control.

Do not add asymmetric loss, inference margins, calibration, QP, or new state
features in this audit. Those changes follow only after isolating the present
failure. Any future collection must sample around the exact post-AEGIS and
final-EE-checked executable action and bind its L5 label to that same action.

H100 producer `39809` and independent validator `39810` completed the matched
audit with exactly zero prediction-replay discrepancy. Test A was close but
failed the strict gate: one L5 false-safe among 67 held-out actions, 2.883 mm
RMSE, 100% L5-safe recall, and support in 7/7 recoverable known states. Test B
failed materially: 14 false-safes among 27 matched actions, 35.955 mm RMSE,
75% recall, and support in 3/3 recoverable new states. All 14 Test-B
false-safes occurred at one unseen state; row-wise optimistic errors affected
L5 rows 1 and 2.

The formal registered interpretation remains
`known_state_action_interpolation_also_fails_representation_or_action_coverage_not_excluded`
because Test A did not reach zero false-safes. The effect size nevertheless
shows state transfer is the dominant observed degradation, not the only one.
Proceed by increasing distinct deployment-distributed, post-AEGIS executable
states and representing relative L5--obstacle/controller state. Do not merely
increase candidate density at existing states or train longer.
## ADR-0117: Compose the L5 residual before one original AEGIS pass

- Status: accepted after independent H100 validation
- Date: 2026-08-14

The restricted learned system keeps the released AEGIS EE constraint enabled,
but it must not apply AEGIS twice. The released filter is stateful: besides its
Cartesian output, each QP advances an auxiliary virtual direction. H100 canary
`39825` showed that reapplying AEGIS to its own archived outputs changes later
actions by up to `2.829361e-3` action units even with no L5 residual. Therefore
the rejected composition is `VLA -> AEGIS -> L5 residual -> AEGIS`.

The accepted opt-in composition is:

```
raw VLA action + bounded L5 residual -> one unchanged released AEGIS pass
-> exact OSC rollout label
```

At zero L5 residual this is exactly the recomputed released-AEGIS baseline at
the restored state. Candidate and backup actions both pass through the same
released EE filter, and every stored risk label binds to its final AEGIS output.
The immutable Table 1 action ledger remains read-only. Its archived action may
differ slightly from a counterfactual replay because the replay state is not
reinterpreted as the historical physical state; that difference is reported as
a diagnostic and never silently used as a baseline equality gate.

H100-host canary `39831` passed exact repeated state, clearance, contact, CAR,
and executed-action replay, and its zero-residual arm exactly reproduced the
recomputed original-AEGIS actions. Full producer `39832` and independent
validator `39846` then passed the same contract for all 37 E05 candidates.
The population contains two exact-safe candidates, 33 contact/CAR failures,
and two censored timeouts; all 37 active witnesses are L5 row 1. This freezes
E05 as a nonvacuous diagnostic positive control and authorizes only
episode-grouped collection with the identical action contract. It does not
authorize training, calibration, QP, or closed-loop execution.

## ADR-0118: Expand AEGIS-consistent collection only after grouped support audit

- Status: accepted after independently validated grouped canary
- Date: 2026-08-14

The corrected composition from ADR-0117 must be tested across disjoint episode
states before broad collection or learning. The two-state canary therefore
uses one train episode (E19) and one validation episode (E22), executes all 37
candidates through one unchanged released AEGIS pass, and retains exact-safe,
known-unsafe, and `UNKNOWN_TIMEOUT` outcomes as separate categories.

Producer `39902`, validators `39939`/`39940`, and summary `39941` passed this
gate. E19 has 15 safe, 7 known-unsafe, and 15 unknown candidates; E22 has 4
safe, 9 known-unsafe, and 24 unknown candidates. Both are usable mixed-support
states. The support is nevertheless specific to L5 row 0: rows 1 and 2 never
cross or approach their boundary, and rows 3--6 remain diagnostic-only.

This result authorizes broader grouped collection under the identical action
contract. It does not authorize training the three-output L5 MLP, because two
claimed outputs remain unsupported. Timeouts must remain censored and cannot
be counted as unsafe examples. Failed/no-support states must remain in the
population ledger rather than being silently removed.

## ADR-0119: Replace exhaustive per-state sampling with adaptive boundary labels

- Status: accepted after independently validated H100 canary
- Date: 2026-08-14
- Supersedes: ADR-0118 only for the broader collection procedure

The complete fixed backup remains the authoritative Monte-Carlo target, but it
is no longer executed for all 37 proposals at every state. Screen nine fixed
normal/tangent proposals with the exact five-action prefix, stop after a
physical veto, identify the closest observed safe/unsafe L5 pair, and perform
five action-space bisection probes. Run the complete backup only for the
nominal and retained boundary samples. Unknown timeouts remain censored;
proxy-invalid, no-safe, and no-bracket states remain visible.

This change tests state generalization rather than spending most of the budget
on redundant actions at one state. It preserves the single original-AEGIS
composition, exact final-action labels, immutable episode splits, seven-row
diagnostics, MuJoCo contact, and CAR. The three sealed test episodes remain
unopened. Because the clean cohort has 18 groups rather than the requested
20--30, this is a 15-development-state feasibility collection; known-invalid
episodes cannot enlarge it.

After independent validation, adequate L5 row coverage authorizes only a
frozen no-training feature audit comparing the existing 86D input, compact
exact-action input, and compact action plus complete physical state. Training,
calibration, QP, closed loop, and any CBF or whole-arm claim remain forbidden.

H100 producer `39982` and validator `39983` passed the mechanism gate. Nine
prefix screens plus five bisection probes retained six unique complete-backup
labels with one exact-safe and five known-unsafe candidates, no timeout, and no
proxy-safe physical collision. Runtime was 222.088 seconds, 41.8% below the
381.535-second exhaustive E05 control. Proceed to the 15 unsealed development
episodes under the identical protocol; retain every no-bracket, no-safe,
proxy-invalid, and timeout outcome.

## ADR-0120: Identify one explicit controllable L5 row per state

- Status: accepted; v2 support NO-GO retained, v3 recovery-arm canary pending
- Date: 2026-08-14

The v1 hard maximum is adequate for proving adaptive runtime and mixed support,
but not for attributing boundary coverage to a claimed output. The grouped v2
protocol therefore audits all three L5 rows separately and selects the
highest-nominal-risk row that has both a globally L5-safe coarse endpoint and
a row-positive endpoint separated by 0.5 mm. It then bisects that registered
row rather than whichever witness happens to win a maximum after each probe.

The 12-candidate coarse bank remains bounded and task-relative: paired normal,
up-tangent, and side-tangent axes; constant and front-loaded profiles; radii
1.0, 1.5, and 2.0; and three mixtures generated by fixed seed 20260814. The
complete backup still supplies every authoritative label, early physical
vetoes remain known unsafe rather than timeouts, and every unsupported state is
retained. This is a mechanism pilot on 15 development groups, not a population
generalization claim. No training may start from v2 collection alone; it must
first pass coverage and the matched no-training feature audit.

H100 producer `39986` and validator `39987` validated the v2 implementation but
found no registered two-sided row: its best row-1 negative endpoint was only
`-0.009707 mm`, short of the fixed `-0.5 mm` gate. No state or failure was
dropped and no bisection was fabricated.

## ADR-0121: Restore the validated recovery arm without relaxing support

- Status: accepted after independently validated H100 canary
- Date: 2026-08-14

Version 2 accidentally replaced the v1 canary's front-loaded radius-2 normal
proposal with a constant-profile radius-2 proposal. This is a candidate-family
omission, not evidence against per-row controllability: the immutable v1 result
already measured `-1.193830 mm` for the former, while v2 measured only
`-0.009707 mm` for the latter at the identical E05 state.

Version 3 restores the front-loaded radius-2 positive and negative normal arms
inside the same 12-candidate budget. It changes no epsilon, action limit,
per-row definition, safe-endpoint rule, bisection count, authoritative rollout,
or downstream gate. The v2 NO-GO remains immutable. The grouped 15-state pilot
may launch only if an independent H100 v3 canary reconstructs the row-1 bracket
and validates both known-safe and known-unsafe retained labels.

H100 producer `39989` and validator `39990` passed. Row 1 spanned
`-1.193830 mm` to `+24.347685 mm`, five bisections retained six authoritative
labels with one exact-safe and five known-unsafe actions, and every original-
AEGIS/action/backup/physical check passed. Proceed to all 15 unsealed
development groups with v3 and classify every support failure, timeout, and
proxy-invalid state. No learning is authorized by the canary.

## ADR-0122: Reject action-count sufficiency and target active state diversity

- Status: accepted after complete grouped H100 validation
- Date: 2026-08-14

Jobs `39993`, `39996`, and `39997` completed the preregistered 15-group v3
mechanism pilot. Although all three L5 rows meet the aggregate known-safe,
known-unsafe, and near-boundary candidate-count thresholds, none meets the
train state-diversity gate; rows 1 and 2 also lack any validation boundary
state and any active witness in the eligible fit population. The population
contains only five usable mixed-support states, while six are no-safe, three
are unknown-only, and one is proxy-invalid.

This distinguishes action coverage from state identifiability. Additional
actions at the same states are not authorized as the remedy. Future collection
must add complete episode groups whose post-AEGIS warning states make the
missing row active, admit a globally safe endpoint, and cross the same fixed
0.5 mm per-row boundary. If the existing deployment distribution cannot expose
such states, either expand the declared task/obstacle distribution or narrow
the learned claim; do not synthesize balanced rows. The current artifacts are
frozen as a validated mechanism-pilot snapshot but not as a learning-ready
dataset. Deployment-gated feature fitting and MLP training, calibration, QP,
closed loop, and sealed-test access remain blocked.

## ADR-0123: Permit diagnostic training without weakening deployment gates

- Status: accepted; H100 diagnostic validated, deployment gate remains NO-GO
- Date: 2026-08-14

The grouped coverage NO-GO does not prevent a smaller capacity experiment. A
fixed 32--32 three-output MLP may fit the known train labels and be tested on
actions held out within the same physical states. This asks whether the
implementation can interpolate the learned L5 action-risk surface; it does not
ask whether the filter generalizes to new row-1/row-2 boundary states.

The split is deterministic and label-independent: for every train state with
at least two known candidates, hold out the candidate with the largest
registered order. Fit the remaining 26 and evaluate six action-held-out
candidates, including target rows 0/1/2. Keep all 16 validation labels grouped
by episode, but make a generalization statement only for row 0 on the two
validation states that actually contain its two-sided boundary. Rows 1--2 are
local interpolation diagnostics only. E05 is excluded from fitting and
reported as a positive control; unknown timeouts, the proxy-invalid state, and
sealed tests are not admitted. The model and its gradients cannot enable a QP
or closed-loop execution regardless of its diagnostic metrics.

Attempt `40048` ended before training when the loader tried to cast the null
legacy correction-norm field on adaptive midpoint candidates. The exact final
post-AEGIS norm already exists in each candidate's validated residual binding.
The representation-only repair uses that field and preserves the complete
preregistered diagnostic.

Retry `40054` then stopped before its first optimizer update because the
evaluation environment's PyTorch lacks H100 `sm_90` kernels. Use the registered
OpenPI Python for both producer and replay validator. This is an allocation-
runtime correction only; the model and protocol remain frozen.

Attempt `40057` exposed only an OpenPI namespace collision in module-style unit
test loading. Use `unittest discover -s tests` so the allocation executes the
repository's test file. No training code ran and no scientific setting changes.

Producer `40060` and independent replay validator `40061` completed the frozen
diagnostic on one H100 with exactly zero prediction-replay discrepancy. The
small model demonstrates local action interpolation: six same-state holdouts
have `0.231658 mm` overall RMSE. This is not a safety pass because row 1 has one
per-row false-safe even in that local test.

Row 0 supplies the only grouped mechanism evidence justified by the coverage
audit. On ten boundary actions from two disjoint supported validation states,
it has zero false-safes, `90%` boundary-side accuracy, `2.750499 mm`
near-boundary RMSE, `75%` exact-safe recall, and safe support in both states.
The broader grouped result must not be collapsed into that claim: unsupported
rows 1 and 2 each have six false-safes, and the excluded E05 positive control
retains no predicted-safe candidate although an exact-safe candidate exists.

Therefore the experiment validates only implementation capacity plus limited
row-0 state transfer. It rejects generalizable three-output L5 filtering from
the current population. Do not train longer, calibrate, open sealed tests,
construct a QP, or run closed loop. Collect new episode groups whose exact
post-AEGIS action family crosses controllable row-1 and row-2 boundaries, then
repeat the grouped gate. Result and validation payload SHA-256 values are
`3ed16bcbdb445f5fda5f8a854ca5ef3fc7b2b1091ebe61b94f10ea1bbb620342`
and `6c9691b79d1a9c43a1adccf53389c761f85f01b9e7b68d3d5d95428939028f6a`.

## ADR-0124: Audit coverage before changing the diagnostic model

- Status: accepted after independent H100 recomputation
- Date: 2026-08-14

Do not infer that all unseen-state failure comes from data merely because the
coverage gate failed. Freeze the validated model and labels, then compare its
error hierarchy across fitted actions, held-out actions at fitted states, and
disjoint physical states. A model-capacity failure should remain visible on
train or same-state holdout. A state-coverage failure should instead show good
same-state interpolation and a large grouped-state error increase coincident
with missing row-specific boundary states and active witnesses.

The audit measures support in the actual 86D feature groups and explicitly
records that q, dq, EEF pose, OSC goals/controller memory, obstacle pose, and
per-row ellipsoid transforms exist in the authoritative artifacts but are not
inputs to this model. It cannot prove those omissions causal without a matched
complete-input ablation. Likewise, any train false-safe is reported as evidence
that symmetric Huber regression is not itself a conservative safety gate. The
only authorized conclusion is a ranked root-cause diagnosis; no model change,
test access, calibration, QP, or closed-loop execution follows automatically.

Jobs `40063` and `40064` validate the primary diagnosis as insufficient grouped
state and active-boundary coverage. Same-state action interpolation remains
sub-millimetre, while grouped validation error rises `135.569x`; E05 row-1
near-boundary error rises `71.677x`. Row-1/row-2 validation boundary-state
counts and fit active witnesses are both zero. E05 is far outside the fitted
86D support, with 33 dimensions beyond the train range and state-context RMS-z
distance `3.489`.

Do not simplify this to “the MLP is fine” or “more random data.” The justified
intervention is more distinct, controllable row-1/row-2 boundary states, plus
the registered missing row-0 train state. After those exist, run a matched 86D
versus complete-physical-state input ablation. The current audit cannot
identify whether the omitted controller variables are additionally causal.
The retained fitted row-1 false-safe also means symmetric regression alone is
not a safety certificate. Audit/validation payload SHA-256 values are
`d25bcca06d7d463887daf0e34e39bf0ac00690b120ed372492296b93302a18c1`
and `40692ccb9932d252d86a06a96f2276573ca14ab026fa3bda227f55f1eadb920a`.

## ADR-0125: Stop task-2 expansion after the targeted row-coverage test

- Status: accepted after independent H100 validation; row 0 extended, rows 1--2 NO-GO
- Date: 2026-08-14

Preregister five unused, clean, task-successful task-2 L5-contact groups before
inspecting their row outcomes. Preserve the existing adaptive-v3 candidate
family, controllability threshold, complete backup label, AEGIS EE gate,
episode grouping, timeout censoring, and sealed tests. The purpose is to seek
distinct row-1/row-2 boundary states plus one additional row-0 training state,
not to add more candidates at known states.

Jobs `40067`/`40068` independently validated one real, initially safe warning
query in every selected episode. Jobs `40077`/`40078` and summary job `40079`
then retained two mixed-support states, two no-safe states, one unknown-only
state, and all 12 censored timeouts. E27 supplies the requested extra row-0
training state and E39 a row-0 validation state. No row-1 or row-2 targetable
boundary was found; cumulative useful state counts are `[3,2,1]` train and
`[3,0,0]` validation.

Interpret this as population-specific evidence, not a universal redundancy
claim. The remaining unused task-2 cohort repeats row-0 mechanics, so further
sampling there is not authorized as a row-1/row-2 remedy. Do not substitute
CAR-only task-3 episodes or open sealed tests without a new preregistered
population definition. The legitimate next choices are: active-boundary
discovery in a separately declared task-3 cohort, new clean task-3 rollouts,
or narrowing the learned feasibility claim to row 0. Until one is chosen and
validated, the feature ablation, model training, calibration, QP, and closed
loop remain blocked.

## ADR-0126: Audit physical contact alignment before changing the safety field

- Status: accepted after independent H100 validation; geometry unchanged
- Date: 2026-08-14

Do not infer an ellipsoid defect from absent row-1/row-2 validation states.
Reuse the existing independently validated internal-substep traces from all 20
unsealed development/targeted states. Deduplicate exact prefix actions, retain
authoritative executed backup actions and terminal holds, exclude unexecuted
backup branches, and keep observed timeout prefixes without treating them as
complete safety labels.

For a physical L5, L6, or L7 contact, compare only the simultaneous rows bound
to that physical link. A physical contact with positive corresponding-link
row minimum is a proxy false-safe and authorizes a geometry audit/refit. A
negative proxy value without raw contact is conservative overlap, not a false-
safe. The closest corresponding row at a contact substep is a physical active
witness; its absence from grouped validation identifies population coverage,
not automatically geometry failure or row redundancy.

Proxy-invalid states are reported separately and cannot support the geometry
gate. The audit opens no sealed tests and performs no new simulation. Poisson,
model training, calibration, QP, and closed loop remain forbidden. If the
proxy-valid false-safe count is zero, keep the current ellipsoids and correct
the data population/claim according to which rows have physical active contact
witnesses.

Attempt `40106` exposed one source-format boundary convention before producing
an audit: the next action may record `substep=-1` for the same physical state
already retained as the previous action's final substep. Map that event to the
retained previous sample. This is an identity-only apparatus repair; do not
duplicate the clearance state or change any geometry threshold. Validator
`40107` was cancelled without running.

Final producer `40111` and independent validator `40112` pass the preregistered
gate. Across 57,275 proxy-valid future samples, every raw L5/L6 contact has a
nonpositive corresponding-link ellipsoid row and every contact is also inside
the 1 mm certification buffer. Keep the current ellipsoids; do not introduce
Poisson as a response to this coverage result.

Revise the data interpretation by physical mechanism. L5 row 0 is the active
contact witness in six task-2 milk states (258 samples). L5 row 1 is the active
witness only in the E05 task-0 moka state (118 samples). L5 row 2 has no active
contact witness. Thus, the task-2 extension was a valid row-0 extension but
could not possibly validate the E05 row-1 mechanism. Future row-1 collection
must use distinct, proxy-valid E05-like/task-0 physical states and preserve
episode grouping. Do not force row-2 learned balance from unrelated data; keep
it as an exact monitored constraint until a declared deployment population
provides an independent physical witness.

The one proxy-invalid state alone introduces 68 apparent false-safe samples
when included. Continue excluding it from geometry/learning claims and report
it as a perception/proxy failure. A row-0-only learned model is permitted only
as a mechanism diagnostic and cannot be presented as solving E05. No learned
filter, QP, or closed-loop gate is opened by this audit.

## ADR-0127: Generate new task-0 moka trajectories without weakening eligibility

- Status: preregistered; H100 discovery pending
- Date: 2026-08-14

The immutable Table-1 population cannot provide another clean E05-like row-1
group: E05 is the only task-successful proxy-valid task-0 L5/moka contact. Do
not reinterpret task-failed E15/E20 as clean deployment episodes and do not
change the accepted ellipsoids merely to manufacture row balance.

Run original AEGIS on all ten frozen task-0 level-II moka initial states with
one new outcome-blind pi0.5 noise schedule per state. Preserve the complete
state, task, action horizon, original EE QP, perception, diagnostics, and
frozen labels. The new run root is separate from Table 1 and all outcomes are
retained. Eligibility requires native task success, proxy validity, no settled
or dynamic task/other contact, and a post-control L5 contact. L6/L7 remain
diagnostic rather than eligible.

At least two new eligible distinct initial states authorize only the unchanged
AEGIS-consistent boundary collector. Fewer than two is a deployment-coverage
failure and blocks training; it is not a failure of the learned method because
no model is trained. No sealed test, Poisson/SDF backend, calibration, learned
QP, or closed-loop experiment is authorized by this discovery population.

Producer array `40129` completed all ten simulations. Summary job `40130`
failed before artifact access because `PYTHONPATH` was not exported; dependent
validator `40131` is cancelled exactly. Add only the repository import path and
rerun summary/validation against the unchanged producer root.

## ADR-0128: Advance validated moka trajectories to grouped boundary audit

- Status: accepted after independent H100 discovery validation; boundary audit pending
- Date: 2026-08-14

Summary `40141` and validator `40142` validate four eligible task-successful,
proxy-valid L5/moka trajectories from producer `40129`: E00, E05, E10, and E45.
This passes ADR-0127's discovery gate. It does not establish boundary support or
authorize learning.

Preserve E05 and E10 as diagnostic initial states because they have repeatedly
influenced method selection. Use E00 as development train and E45 as development
validation. The split unit is the initial-state/policy-noise trajectory, and the
diagnostic variants may not count toward unseen generalization.

At every real five-action query before first L5 contact, audit the exact released
AEGIS action prefix. Retain only states that are initially proxy/contact/CAR safe
but whose nominal prefix is unsafe. Bind every source trajectory, summary, and
validation hash. If retained states exist, run the unchanged adaptive-v3
normal/tangent coarse screen, per-row two-sided 0.5 mm boundary rule, five exact
prefix bisections, and candidate-plus-fixed-backup authoritative labels. If no
retained state exists, record coverage failure rather than changing geometry or
eligibility.

This experiment may supply distinct row-1 evidence or disprove support in this
population. Row 2 remains diagnostic unless physically activated. It neither
changes the accepted ellipsoids nor opens training, calibration, QP, closed loop,
Poisson, or sealed tests.

Attempt `40143` stopped during immutable discovery binding because the summary
uses `payload_sha256` as its digest field. Correct only that schema key and rerun
against the identical summary digest; no environment or scientific evaluation
ran. Cancel dependency-blocked validator `40144` exactly.

Attempt `40148` was also pre-simulation apparatus history: the submitted full
commit string did not match the checked-out source. Cancel validator `40149` and
do not reuse that run root. Producer `40153` and validator `40154` use the exact
verified commit and pass.

All four validated trajectories expose one timely row-1 warning state. Advance
those four states to unchanged adaptive-v3 collection. Preserve E05/E10 as
diagnostic and E00/E45 as train/validation. Bind the source trajectory's
outcome-blind policy-noise seed when replaying candidates; this is required to
reproduce the archived action ledger and is not a new candidate variable.
Continue to log all seven rows, contacts, CAR, failures, and timeouts, while the
learned claim remains L5-only and row 2 unsupported.

Adaptive producer `40158` and validator `40159` pass. E00 and E45 provide the
required new row-1 train/validation boundary states; E05/E10 remain no-safe
diagnostics. Preserve every candidate and failure.

Do not apply the generic seven-link physical-veto classification to the scoped
L5 learning target. E00's nominal candidate stops on an L6-only raw contact
while all L5 risks are safe. Because the stopped rollout has no complete backup,
censor it for L5 rather than label it safe or unsafe. Continue reporting that
L6 contact as cross-link failure evidence. L5 contact and CAR remain hard L5
physical vetoes. This realizes the declared rule that L6/L7 are outside the L5
decision but inside evaluation; it does not waive them for a future whole-arm
claim.

Scoped summary `40172` passes the complete row-0 and row-1 coverage gates but
row 2 remains unsupported. Authorize only a preregistered matched row-0/row-1
feature audit, with row 2 calculated and logged exactly. Do not train the
three-output L5 model, add calibration/QP/closed loop, or open sealed episodes.
If the intended claim still requires learned row 2, collect a physical
population that activates its boundary first.

## ADR-0129: Restrict the first learned correction test to L5 rows 0--1

- Status: preregistered; H100 prediction/selection pending
- Date: 2026-08-14

The user narrows the learned claim together with its acceptance gate. The
validated population supplies three useful train states for each of L5 rows 0
and 1 and three/one disjoint validation states for rows 0/1. Row 2 remains
unsupported and is not a learned output. This authorizes one fixed two-output
capacity-and-selection experiment; it does not authorize a three-output L5 or
whole-arm claim.

Fit the existing compact 86D action-conditioned architecture with two outputs
and unchanged deterministic symmetric boundary-weighted Huber training. Use
all 43 known train labels and 29 known grouped-validation labels from the
immutable base, targeted, and scoped moka sources. Exclude timeouts,
proxy-invalid states, and the E00 candidate stopped only by an out-of-scope L6
contact. Preserve complete episode grouping and do not open reserved tests.

At each validation state, accept candidates only when both predicted L5 risks
are nonpositive, then choose the minimum registered correction norm. Compare
with the closest exact-safe oracle, raw AEGIS record, and 1,024 fixed-seed
random bank draws. Passing requires zero row-0/row-1 false-safes, predicted-safe
support in all four recoverable validation states, every selected action to
pass all seven exact ellipsoid rows, protected MuJoCo contacts, CAR, and the
unchanged original AEGIS EE projection, and a selected safe rate above random.
Any selection in an unrecoverable state that transfers collision to row 2,
L6, or L7 is a failure rather than a silently discarded sample.

Only a complete prediction gate authorizes fresh H100 replay of each selected
candidate. Candidate selection is tested before gradients: calibration, QP,
closed-loop execution, neural-CBF language, and learned row-2/L6/L7 claims
remain forbidden.

Producer attempt `40173` stopped before model construction on a nullable receipt
alias for correction norm; dependent `40174` is cancelled. Fall back to the
already validated `residual_binding.applied_residual_l2_action` only when the
top-level alias is null. This is apparatus repair, not a protocol change.

H100 producer `40175` and independent prediction replay `40176` complete the
fixed experiment at commit `07f855f0488d138e1c453223e9e47f238e251d58`.
The result is a strict prediction-and-selection NO-GO. The model fits the 43
training labels to `0.164472 mm` RMSE, but grouped validation RMSE is
`14.716616 mm` (`17.939185/10.551929 mm` by row) and contains 16 row-0/row-1
false-safes among 29 labels. The validator reproduces every prediction exactly.

All four recoverable validation states contain an exact-safe candidate and at
least one candidate predicted safe, but minimum-change learned selection is
exact-safe in only one: E22. It selects truly unsafe candidates in E37
(`+1.630478 mm` row-0 violation), E39 (`+3.838207 mm` row-0 violation), and E45
(`+9.017434 mm` row-1 violation; nominal selected). It also predicts all six
candidates safe in unrecoverable E34 and chooses an unsafe nominal action.
Original AEGIS EE projection compatibility passes for every selection, so the
failure is learned grouped risk prediction rather than bypassing AEGIS.

The selected exact-safe rate is `25%`, statistically and practically
indistinguishable here from the preregistered seeded-random bank rate
`24.5768%`. Do not interpret the Boolean greater-than check as evidence of an
advantage. Because the prerequisite zero-false-safe prediction gate fails,
fresh selected-action simulation, QP, calibration, and closed-loop execution
remain blocked. This result shows that row-count coverage alone did not fix
generalization. The next justified experiment is the already motivated matched
input audit/model comparison using complete physical/OSC state; do not collect
more candidates at these same states or train longer first.

## ADR-0130: Match compact and complete physical/OSC inputs before changing loss

- Status: completed; strict prediction/selection NO-GO
- Date: 2026-08-14

Run exactly two arms on the immutable 43 train and 29 grouped-validation labels:
the validated compact 86D input and a 354D enumerated physical/controller input.
Keep both outputs, candidate actions, episode splits, hidden widths, SiLU
activation, symmetric boundary-weighted Huber loss, optimizer, seed, 2,000
epochs, and fixed-final checkpoint identical. Normalization is fit separately
per arm using training episodes only. The compact arm must reproduce frozen job
40175 to `1e-9 m` before the comparison is accepted.

The 354D contract contains q/dq, current EE pose, OSC goal and every numeric
field in the recorded controller snapshot, complete nominal and exact final
candidate chunks, obstacle transform/semiaxes, the three L5 ellipsoid
transforms, their relative centers/normals/clearances, and the complete numeric
five-step original-AEGIS projection/intervention record. Nullable `ori_ref` is
encoded by a presence bit plus a zero-filled 3x3 value. Do not include the raw
648D simulator dynamic-state vector: it is not in the requested deployable
feature list and would introduce a high-dimensional simulator-only identifier.

Passing requires zero row-0/row-1 false-safes, predicted-safe support and exact
all-seven-safe minimum-change selection in all four recoverable validation
states, safe selection above fixed-seed random, validation RMSE below the
compact arm, and near-boundary RMSE at most 1 mm. Fresh selected-candidate replay
is authorized only after every prediction gate passes. Do not change the loss,
candidate bank, labels, calibration, QP, geometry backend, or control policy.

H100 producer `40181` and independent validator `40182` complete the fixed
comparison at commit `cc996f5a0ecd93baed35a4b926ebe22adf884624`.
The compact arm reproduces job `40175` exactly and independent prediction replay
error is zero. Complete physical/OSC input improves validation RMSE from
`14.716616 mm` to `11.427894 mm`, near-boundary RMSE from `10.135355 mm` to
`7.220164 mm`, and false-safes from 16 to 13. It nevertheless supports only
three of four recoverable states, selects an exact-safe candidate in only one,
and remains effectively tied with random selection (`25%` versus `24.5768%`).

The decisive split is training versus grouped validation: compact/complete
training RMSE is only `0.164472/0.112910 mm`, while grouped-state error is two
orders of magnitude larger. Thus the omitted fields were useful but not the
root solution. Do not add a one-sided buffer merely to erase false-safes: the
complete model already loses E37 safe support. The next intervention must first
increase distinct state/episode support or test a representation designed to
transfer relative physical structure across states. No fresh replay, QP,
calibration, closed loop, or sealed-test access is authorized by this result.

## ADR-0131: Separate rigid-frame representation from state-support failure

- Status: completed; representation helps but strict safety remains NO-GO
- Date: 2026-08-14

The current evidence proves grouped-state failure but does not separately prove
the two proposed causes: too few independent states and a non-invariant flat
input. The existing 86D arm is already partially relative (clearances plus one
normal/tangent frame), while its action vectors remain in the global VLA frame.
The 354D arm adds complete state but mixes absolute world transforms with only
43 labels. Calling representation the root cause without one further matched
audit would therefore overstate the evidence.

On the unchanged 43 train and 29 grouped-validation labels, compare the frozen
354D result against one 134D physics-aligned two-output MLP, five-neighbor
regression, and fixed-ridge regression. The 134D representation keeps q/dq,
expresses OSC error and both five-action chunks in the obstacle frame, and
expresses each learned L5 row's obstacle pose and outward normal in that
ellipsoid's frame. A synthetic rigid-transform test must establish exact
feature invariance. Keep the MLP widths, loss, optimizer, seed, and training
schedule identical to ADR-0130.

Audit nearest distinct training-state and candidate distances using training
normalization only. Also record whether each row's worst rollout value comes
from the candidate prefix or fixed backup and whether that witness phase changes
at the nearest training candidate. The representation hypothesis passes only
with zero false-safes, safe support and exact-safe selection in all four
recoverable states, near-boundary RMSE at most 1 mm, and lower RMSE than the
frozen 354D arm. Failure with close training fit but distant grouped states
supports new independent boundary collection; nearby inconsistent targets plus
witness switching instead motivates structured prefix/backup or time-resolved
outputs. No simulation, new label, calibration, QP, closed loop, Poisson, or
sealed-test access is authorized.

H100 producer `40186` completed the frozen-data audit at preregistered commit
`8457ccd8d501e826c7e020022ac4fbde0fc0237e`. Validator `40187` failed only
because it reused a legacy loader with an 86D normalization assertion; it did
not write validation output or alter the producer. Validator-only commit
`7ae89a630fd5aec0128d7f93c5efd4d6ebd4a2c7` removes that hard-coded shape, and
independent validator `40188` reproduces every model and baseline prediction
with exactly `0.0 m` maximum error.

Rigid-frame features improve the safety-relevant part of the map but do not
solve it. Relative-MLP near-boundary RMSE is `4.476086 mm` versus `7.220164 mm`
for complete 354D, false-safes fall from 13 to 6, and exact-safe selection rises
from 1/4 to 2/4. Overall validation RMSE worsens from `11.427894 mm` to
`14.417849 mm`; E45 has no predicted-safe candidate and E39 retains an unsafe
selection. KNN and ridge are worse (`16.255683/54.821087 mm` RMSE and 0/4 safe
selections), so the frozen samples do not contain a simple transferable local
map that a different optimizer was missing.

The state audit separates two failures. E45 is far from training at `1.284`
RMS z over the 64 state features, has 14 features outside the train range, and
its corrected candidates are prefix-dominated while their nearest training
examples are backup-dominated; target gaps are roughly 40--49 mm. This is a
clear missing E05-like state/phase-support failure. E39 is close (`0.266` RMS z,
one feature outside range) and retains the same backup witness phase, yet
nearest candidate target gaps still reach several millimetres and learned
selection violates row 0 by `3.838207 mm`. This is state-conditioned nonlinear
action-response variation, not merely absolute-frame aliasing. Global nearest-
candidate witness-phase disagreement is only `20.69%`, so hard prefix/backup
switching is important for E45 but is not the sole population-wide cause.

Retain rigid-frame features as the better boundary representation, but do not
claim they yield a correct gradient. Next collect additional independent row-1
E05-like prefix-warning states and row-0 E39-like boundary states, explicitly
covering prefix and backup risk sources. The next predictor should expose
prefix risk and backup continuation risk as separate heads and combine them by
a hard maximum for acceptance. First validate Best-of-N candidate selection;
only after it passes may action gradients be tested against exact paired
directions. Calibration, QP, and closed loop remain blocked.

## ADR-0132: Audit prefix and backup components before new collection

- Status: completed; both component predictors fail grouped safety
- Date: 2026-08-14

Before collecting another rollout, decompose the immutable row-0/row-1 target
into exact five-action prefix risk and observed fixed-backup risk. A missing
backup value means the prefix terminated before backup execution and remains a
censored value; it is never imputed. Fit two matched 134D component MLPs using
the ADR-0131 representation, architecture, symmetric Huber loss, seed, and
schedule. Evaluate prefix and backup errors separately, then form the structured
prediction by their elementwise hard maximum.

Attribute each direct and structured false-safe to the exact active row and
prefix/backup phase. For E39, select the nearest training state among states
with at least four distinct registered correction magnitudes, and compare
least-squares slopes of prefix, backup, and combined risk versus correction
norm for both rows. This audit determines whether the component maps fail,
whether their maximum is uniquely responsible, or whether E39 has a genuinely
different local action response. It changes no state, action, label, candidate,
controller, or split and opens no sealed episode. QP, calibration, gradient,
closed-loop, and neural-CBF claims remain forbidden.

H100 producer `40190` and independent validator `40191` complete the audit at
commit `32f194670c302695d0070dbd297011dc85fd5354`, with exactly zero prediction
replay error. Prefix labels cover all 43/29 train/validation actions; observed
backup labels cover only 28/25 because missing backups remain censored.

Both components fail independently. Prefix validation RMSE/near-boundary RMSE
is `14.357290/9.369379 mm` with 7 false-safes and `53.85%` safe recall. Backup
is better on average (`8.752153/5.309479 mm`) but has 9 false-safes. Their hard
maximum has `14.073917 mm` RMSE, `8.505896 mm` near-boundary RMSE, 9 false-
safes, and only 2/4 exact-safe selections. Therefore target switching is not
the root by itself, and separate heads are not yet justified as a safety fix.
All six false-safes of the better direct relative model are backup-dominated;
the structured heads create eight backup and one prefix false-safe.

E39 supplies the important positive mechanism evidence. Its nearest training
state with a real response curve is E27 at `0.342` RMS z. All six identifiable
prefix/backup/combined slope signs for rows 0/1 agree. E39 row-0 combined slope
is `-0.019415` m risk per action-norm versus `-0.015002` for E27; prefix slopes
are `-0.019647/-0.021133`. Thus the physical correction direction transfers,
while offset, magnitude, and the safe crossing point remain state-dependent.

The next data/model hypothesis should factor each component as a state baseline
plus action-induced change,

`Q_j(z,A) = B_j(z) + Delta_j(z,A-A_nominal)`,

using paired/bisection candidates to supervise `Delta_j` and absolute rollout
labels to supervise `B_j`. Collection must prioritize distinct E39-like states
with complete backup safe/unsafe crossings and E45-like prefix-dominated states,
not additional random actions at current snapshots. This may support a useful
ranking/steering direction, but absolute acceptance remains blocked until the
baseline and crossing point produce zero false-safes and 4/4 support.

## ADR-0133: Collect grouped anchors before fitting policy-conditioned B plus Delta

- Status: completed; passive grouped collection is a coverage NO-GO
- Date: 2026-08-14

The factorized model is specified as

`Q_j(z,A) = B_j(z,A_nominal) + Delta_j(z,A_nominal,A-A_nominal)`,

with `Delta_j(z,A_nominal,0)=0` enforced architecturally. This prevents policy
noise from giving one state-only baseline inconsistent nominal targets. The
current frozen population is not sufficient for a fair fit: only five training
states contain a known nominal candidate plus complete authoritative target.

Before training, collect twelve outcome-blind trajectories from unused initial
states in task-0 moka and task-2 milk. Eight complete task/episode groups are
fixed as train and four as validation; episodes 31 and 36 in each task remain
unopened for a later final gate. Every eligible task-successful, proxy-valid L5
trajectory uses a real five-action query boundary, the unchanged adaptive-v3
candidate bank, five bisections, original AEGIS EE projection, and candidate-
plus-fixed-backup labels. Missing backup after prefix contact/CAR is censored.

Training is authorized only if the new population supplies at least two train
and one validation states with a known nominal plus four response actions, and
both prefix- and backup-dominated two-sided boundaries occur in train and
validation. The first learned gate will evaluate `Delta` as a direction/ranker
against fixed repulsion and matched random selection. Only afterward may
`B+Delta` be tested as an absolute safety gate. QP, calibration, closed loop,
Poisson, whole-arm, and neural-CBF claims remain forbidden.

H100 producer array `40201` completed all twelve registered trajectories at
commit `334bbffb76ff1221e0de320a268afe76d39860fb`; independent validator array
`40202` reproduced every classification. Eleven trajectories completed their
task without L5 contact and one task-0 training trajectory failed the task.
Consequently no case entered warning-state/action-risk collection: both splits
have zero known nominal anchors, zero response curves, and zero prefix- or
backup-dominated boundary states. Replacement summary `40229` ran on H100 at
commit `ff86928aea179db8d42399d7c8da3757e3e958ce` after CPU-only attempt `40203`
failed in provenance collection before writing an artifact.

This is a deployment-coverage result, not a rejection of the factorization.
Changing policy noise across arbitrary unused episodes mostly produces safe
task trajectories, so passive trajectory sampling is too sparse to identify
the rare L5 transition. Training on this extension would add no boundary data
and is forbidden. The next collection decision must be explicit: either expand
the passive episode population substantially, or run a mechanism-only active
state/obstacle perturbation pilot around existing E39/E45 warning states. The
latter is more efficient but cannot be presented as independent deployment
generalization; final validation would still require untouched natural
episodes. No model, QP, calibration, or closed-loop job follows this NO-GO.

## ADR-0134: Test action response with an exact nominal-risk anchor

- Status: completed; exact-anchor scalar response is a strict NO-GO
- Date: 2026-08-14

Do not launch active state or obstacle perturbation after ADR-0133. The frozen
rollouts permit a cheaper diagnostic that removes nominal-risk prediction from
the experiment. For every state with one authoritative nominal candidate and
at least four response actions, define

`B*_j = Q_j(z, A_nominal)` and
`Delta*_j = Q_j(z, A) - B*_j`.

The two-output model receives rigid-frame relative context and the exact
candidate-minus-nominal action. Network subtraction enforces
`Delta_theta(z, 0) = 0`. The exact `B*` is supplied at validation, so failure
cannot be attributed to estimating the nominal-risk offset. Episode splits,
actions, labels, model width, seed, optimizer, and fixed final epoch remain
unchanged.

Only three train and three validation states satisfy the anchor/response rule,
with 14 and 15 non-nominal responses. Their actions are bisection amplitudes
along one registered physical correction path per state. This gate therefore
measures response magnitude, improvement sign, safe/unsafe ordering, and
amplitude selection; it cannot establish a general 15-dimensional steering
direction. The largest-amplitude registered candidate and fixed-seed random
amplitude selection are reported as baselines. No unavailable direction label
is invented.

Passing requires lower response RMSE than the zero-change baseline, at least
80% improvement-sign accuracy, at least 85% safe/unsafe pair ordering, safe
support and exact-safe selection in every recoverable validation state, a safe
selection rate above random, and no larger correction than the strongest path
candidate when both are safe. A pass authorizes only a small capped active
direction pilot. A failure stops the exact-anchor `Delta` formulation before
new collection. Nominal-risk learning, calibration, QP, closed loop, sealed
tests, and neural-CBF claims remain blocked.

H100 producer `40237` and independent replay validator `40238` complete the
gate with exactly zero prediction-replay error and exactly zero architectural
response at zero correction. The result is a strict NO-GO for the scalar
exact-anchor `Delta` predictor. Validation response RMSE is `18.134943 mm`,
only slightly below the `19.091172 mm` zero-change baseline. Although all 15
improvement signs, all 16 safe/unsafe pairs, and all 45 non-tied candidate
pairs are ordered correctly, the response magnitude is not transferable: the
model creates three false-safes, supports only one of two recoverable states,
and its minimum-correction acceptance is exact-safe in zero of two.

The ordering result is not sufficient to authorize active perturbation. Each
state varies only the amplitude along one monotonic registered path. The
top-ranked candidate is the strongest registered repulsion in both recoverable
validation states, so the learned ranking adds no detour or task-preservation
value beyond the fixed strongest-amplitude baseline. It does outperform random
amplitude ranking, but that is the expected consequence of learning monotonic
amplitude on this bank, not evidence of a general physical action direction.

Therefore do not launch the active-boundary pilot under this formulation. The
useful surviving observation is narrower: existing data reliably identify that
more motion along the already-chosen path improves risk. They do not identify
how much is safe or which independent direction should be chosen. Any next
learning proposal must add independently varied physical directions and be
formulated as a candidate ranker/multimodal proposal only if it demonstrates
value beyond the registered analytical repulsion. Absolute safety acceptance,
QP, calibration, closed loop, and sealed tests remain blocked.

## ADR-0135: Audit whether registered physical modes vary by state

- Status: completed; analytical normal dominates the observed prefix population
- Date: 2026-08-14

Pause learning and new data collection after ADR-0134. Reuse the immutable
coarse-screen traces from the 16 proxy-valid grouped row-0/row-1 states to ask
whether a learned mode selector has a physical role at all. Each state already
contains nominal, positive/negative obstacle normal, upward tangent, side
tangent, and fixed-mixture proposals.

The primary comparison is the strongest analytical outward normal
`normal_pos_front_loaded_r2.0`. A matched requested-radius-one comparison keeps
positive/negative normal, upward tangent, and side tangent separate. Applied
norm and clipping are reported because action bounds can break exact norm
matching. Risk is the maximum exact five-action prefix violation over L5 rows
0--2, and physical contact/CAR remains a veto.

The audit asks two distinct questions. First, does the strongest outward normal
pass in every observed state where any registered coarse mode passes? Second,
at matched requested radius one, does a tangent rescue a state where outward
normal fails, or do multiple mode families beat normal by at least 1 mm across
states? The first supports analytical repulsion; the second is preliminary
evidence for state-dependent proposal selection.

These artifacts do not contain complete candidate-plus-backup labels for all
coarse modes. Therefore even a positive prefix-diversity finding cannot
authorize a learned selector or policy-value claim. It can only justify a later
small matched-mode complete-rollout experiment after advisor review. No new
simulation, model training, active perturbation, QP, calibration, closed loop,
or sealed-test access is authorized here.

H100 artifact audit `40244` and independent replay `40245` reproduce all 192
stored coarse prefix probes across 16 states. The strongest outward-normal
radius-2 proposal is the lowest-risk registered mode in all 16 states. Fourteen
states contain at least one zero-margin prefix-safe mode, and the same outward
normal is safe in all fourteen. In the remaining two states no registered mode
is safe. No tangent or mixture rescues a failure of the strongest normal.

At requested radius one, normal wins 13 states, upward tangent two, and side
tangent one. Only one tangential winner beats positive normal by at least 1 mm,
and none changes an unsafe normal result into a safe one. This is not meaningful
state-dependent detour evidence. The current artifact population therefore
does not justify learning a candidate-mode selector: the selector would learn
to choose the same strongest analytical normal already available explicitly.

This does not establish that radius-2 normal is a satisfactory controller.
Sixty-six non-nominal coarse proposals experienced action clipping, radius 2 is
a large correction, and the audit covers only the five-action prefix. It does
not measure complete fixed-backup safety, live VLA recovery, intervention norm,
or task preservation for every mode. The research should consequently pivot
to the simpler analytical story: verified predictive normal repulsion followed
by VLA replanning, with task compatibility and complete continuation as the
remaining questions. Learned selection should be reconsidered only if future
matched-norm, complete-rollout evidence exposes genuine mode-dependent rescue.

## ADR-0136: Test frozen analytical repulsion on sealed complete episodes

- Status: completed; collision prevention transfers, strict safe-task gate fails
- Date: 2026-08-14

Open the three episode groups sealed by ADR-0109 only after freezing the
algorithm. The cases are goal-task-2 E42 (historical L5 contact) and goal-task-3
E42/E44 (historical L6 contact). Their immutable released-AEGIS results are the
paired task-successful collision controls. No development or test outcome may
change the method after this registration.

At every real five-action query boundary, an exact cloned-OSC rollout of the
nominal post-AEGIS prefix supplies a privileged warning only. If any of seven
L5--L7 rows falls below +1 mm, a protected MuJoCo contact occurs, or paper CAR
would fail, apply exactly the previously registered
`normal_pos_front_loaded_r2.0` correction. The normal points from the obstacle
proxy center to the currently closest L5--L7 slab center; the unit-L2 temporal
profile is 5:4:3:2:1; XYZ is clipped at [-1,1]; and the unchanged released
AEGIS EE QP is applied sequentially once more. Magnitude, threshold, temporal
profile, and direction cannot adapt. Execute five actions, query frozen pi0.5
from the changed observation, and repeat through task completion or horizon.

The strict gate requires all three episodes to complete the native task with
zero raw L5--L7 contact, paper CAR pass, no collision transfer, and no timeout.
Every intervention norm and clipping event is retained. Failure in any case is
a NO-GO for this fixed controller. A pass supports only transfer of the
analytical correction mechanism under privileged prediction: it is not an
unbiased population result, deployable warning detector, CBF, or formal safety
claim. Learning, QP changes, and outcome-dependent retuning remain forbidden.

Pre-outcome apparatus amendment (2026-08-15): producer arrays `40263` and
`40271` stopped before writing any scientific result because byte-exact replay
equality and then the preregistered `1e-12` absolute tolerance rejected
clone/execution roundoff of `2.397738e-12` and `2.416745e-12`. The tolerance is
changed once to `1e-9`, still six orders below the 1 mm safety buffer. The
measured maximum is now recorded and independently validated. Cases, actions,
warning threshold, safety acceptance, repulsion direction/magnitude/profile,
AEGIS QP, policy seeds, and strict three-of-three gate are unchanged.

The next apparatus attempt exposed a distinct contact-rich fidelity issue:
before any repulsion, identical five-action execution at step 105 diverged by
`0.0133021` in the MuJoCo simulator state despite an exactly synchronized
initial snapshot. Full dynamic-state equality is therefore not a valid hard
gate for grasp/contact transitions. It becomes a reported diagnostic, together
with final seven-row clearance error and forecast/observed contact and CAR.
The privileged cloned rollout still triggers the frozen controller, but actual
all-substep MuJoCo contact, CAR, and native task state remain the only outcome
authorities. The claim is correspondingly narrowed from an exact oracle to a
privileged cloned-OSC forecast with measured fidelity; no outcome-dependent
controller change is allowed.

Final H100 producer array `40285` and independent validator `40286` complete
all three frozen episodes. The analytical correction eliminates every raw
L5--L7 contact and passes CAR in 3/3 cases. This is positive evidence that the
outward-normal mechanism transfers from the development cases to one sealed L5
and two sealed L6 collision episodes. It is not a safe-task controller: native
task completion is only 1/3. Goal-task-3 E42 succeeds after one intervention;
goal-task-2 E42 times out after ten radius-2 interventions, and goal-task-3 E44
times out after one. Neither timeout is classified as a static deadlock.

The main failure is task compatibility, not insufficient collision authority.
Case E42/task-2 receives repeated early and late interventions from step 25,
including two clipped corrections, and never satisfies its task predicate.
Case E44/task-3 shows that even one large correction can move the frozen VLA
onto a continuation that stays physically active but does not finish. Exact
compiled-obstacle overlap remains zero in every case, while conservative proxy
margins reach `-11.073`, `-12.050`, and `-18.283 mm`; thus proxy warning
conservatism likely contributes to over-intervention and must be separated from
physical collision prevention.

This verdict is superseded by the camera-integrity audit in ADR-0137. The
contact measurements remain observed facts, but task completion and live-VLA
continuation from job `40285` are not valid evidence because the policy received
corrupted camera frames after the probe renderer was constructed. No controller
conclusion may be drawn until the identical frozen three-case gate is rerun with
validated images.

## ADR-0137: Require portable playback validation before video handoff

- Status: completed
- Date: 2026-08-15

The first user-facing handoff linked the raw experiment MP4 files. Although
their remote/local hashes matched, the files were 96/34/28 MB and failed in-app
playback. Container existence and checksums are therefore insufficient video
acceptance criteria.

The codec was not the root cause. Full transcoding and decoding preserved the
same horizontally interleaved image corruption, and the independently written
terminal JPEG is also corrupted. The valid immutable Table 1 video for the same
case decodes normally. The generalization evaluator constructed a 1024px main
`OffScreenRenderEnv` and then a 32px probe `OffScreenRenderEnv` in one OSMesa
process. The second offscreen context replaced the process-global buffer used
by subsequent main observations. Those damaged observations were written to
video and sent to live pi0.5, invalidating the task-preservation result.

The apparatus repair constructs and image-disables the 32px probe first, then
constructs the 1024px main renderer last. A normalized adjacent-pixel integrity
metric is checked on the initial frame and every executed frame; dense buffer
interleaving fails closed before a scientific result is written. The frozen
repulsion rule, cases, seeds, warning logic, action horizon, and safety/task gate
remain unchanged for the required rerun.

All future simulation-video handoffs must first run
`slurm/finalize_simulation_videos.sbatch` on H100. It generates bounded-width
H.264 Main/level-3.1, yuv420p, fast-start, silent MP4 files, fully decodes every
output with ffmpeg, and writes an immutable size/hash manifest. Raw experiment
videos remain scientific artifacts but must not be used as the delivered copy.

H100 job `40296` validates the portable encoding mechanics at 512-pixel maximum
width and CRF 30. Sizes are 4.10/0.829/1.86 MB and SHA-256 values are
`db2a15afabdc384558949dd1ee6f6398488b5483dcf7debea6822dd5d496893a`,
`51fc327420a364fd3845be4387aba4d532b013dd116c0dd3df310dd54689971f`,
and `ef2efe0568024fa5171a5b7fc14c3aefce2579ad88b6d12927061c2137649f3a`.
They are not valid visual evidence because source-frame corruption precedes
encoding. Future handoff requires both the runtime frame-integrity gate and the
portable full-decode gate.

## ADR-0138: Accept corrected renderer rerun; reject safe-task generalization

- Status: completed
- Date: 2026-08-15

H100 producer `40303` and independent validator `40304` reran the unchanged
ADR-0136 controller after the ADR-0137 renderer repair. Every raw frame passes
the runtime integrity gate; per-case maxima are `0.007512`, `0.008730`, and
`0.008682` against threshold `0.15`. Visual inspection of terminal JPEGs and
macOS QuickLook-decoded video frames is clean. H100 finalizer `40319` fully
decodes all three portable H.264 outputs. The video apparatus gate therefore
passes, and run `40285` remains invalidated history rather than evidence.

The corrected run eliminates raw L5--L7 contact and passes CAR in 3/3 sealed
episodes, but native task completion is 1/3. Goal-task-2 E42 completes after
five interventions; goal-task-3 E42 and E44 time out after 39 and 31
interventions. The strict safe-task gate is therefore NO-GO. This supports only
the collision-prevention authority of fixed outward-normal repulsion under a
privileged cloned-OSC warning. It rejects the current fixed radius-2 warning
and intervention schedule as a task-preserving general controller.

The root issue after repairing observation integrity is excessive repeated
intervention, not failure to push the arm away. The next scientific change, if
authorized, must isolate warning calibration/hysteresis and bounded repulsion
magnitude on new untouched complete episodes. It must not be described as
learned steering, a deployable detector, a CBF, or whole-population safety.
Result and validation payload SHA-256 values are
`421adf1150c74fa7f194f14c54d6a0fa3c8b037efa834ca63a1ba139be871ea5`
and `7b246dbdbb8f67a3017471289a2cd3cd1421740e292caf004e80775101aa5900`.

## ADR-0139: Test exact post-AEGIS normal-magnitude risk curves before learning

- Status: completed; safe-support gate passes, monotone bisection remains unsupported
- Date: 2026-08-15

The fixed radius-2 controller establishes outward-normal collision authority
but not task-compatible magnitude selection. Risk is measured in metres while
the correction radius is in normalized Cartesian action units; no conversion
between them is valid. The next no-learning gate therefore measures

\[
q_j(z,\bar A,\alpha)=
Q_j^\star\!\left(z,P_{\rm AEGIS}(\bar A+\alpha D_n(z))\right)
\]

at requested radii `0, 0.25, ..., 2.0`. Every candidate transfers its residual
to the raw VLA chunk, passes once through the unchanged released AEGIS EE
filter, executes five OSC actions, and then follows the complete registered
fixed backup. Prefix and backup risks remain separate; the combined label is
their per-row maximum violation with the existing 1 mm buffer. Requested,
clipped pre-AEGIS, and effective post-AEGIS correction norms are all recorded.
Timeouts are censored as unknown and never participate in monotonicity or safe
support.

This first mechanism pilot is limited to the first validated warning state of
the three now-open diagnostic cases: goal-task-2 E42 at step 25, goal-task-3
E42 at step 100, and goal-task-3 E44 at step 110. It tests whether each curve
contains a known exact-safe magnitude and whether worst future risk is
sufficiently monotone to justify one-dimensional minimum-magnitude search. It
does not rerun complete adaptive episodes, train a model, collect denoising
states, change a QP, reopen a generalization claim, or establish a CBF. A
complete adaptive-magnitude episode is authorized only if these exact curves
contain safe support. Newly untouched natural episodes remain mandatory for
any later generalization claim.

H100 producers `40339`--`40341` and independent validator `40348` completed
all 27 registered points. Every diagnostic state contains a known exact-safe
magnitude: the smallest requested/effective post-AEGIS L2 magnitudes are
`1.25/1.065836` for goal-task-2 E42, `0.75/0.711527` for goal-task-3 E42,
and `1.75/1.585015` for goal-task-3 E44. None of the 27 proposals clipped.
The nine complete-backup timeouts remain censored unknown rather than being
relabeled unsafe. Goal-task-2 E42 is monotone over all known points; the two
task-3 curves each contain one measured monotonicity reversal, including a
3.368 mm risk increase from requested radius `1.0` to `1.25` in E42.
Consequently, safe support authorizes a complete adaptive-magnitude oracle
comparison using conservative registered grid search, but not monotone
bisection, direct risk-to-force conversion, an MLP, or flow guidance.

Summary/validation file SHA-256 values are
`c02b37766c4d8b6b8268b5f29869fd8002466b4ed2614065e8668f7c218c6c37`
and `83d4e42f06d19e0bd0e904267e289cf9cb4e2fad1866913fc689ddb1a52affc2`;
payload SHA-256 values are
`a046bc854e32f8cf297cc00efd57d74d3f67467daedbb5fabe5100df40a1f4bc`
and `9829b9fc84205077980766fdf44d026cde5f2a63ff427ca2e9e9145143bbbd46`.

## ADR-0140: Test first-warning-calibrated magnitude through complete episodes

- Status: completed; fixed per-episode calibration is a strict NO-GO
- Date: 2026-08-15

ADR-0139 establishes safe support but not an online magnitude predictor. Before
paying for nine complete backup evaluations at every warning, isolate whether
the fixed radius `2.0` caused task failure through excessive intervention. For
each now-open diagnostic episode, freeze the smallest exactly safe requested
radius found at its first warning: `1.25` for task-2 E42, `0.75` for task-3
E42, and `1.75` for task-3 E44. Reuse that one case-specific radius at every
later privileged five-action warning while preserving the normal, temporal
profile, original released AEGIS EE filter, live frozen-pi0.5 replanning,
internal-substep measurements, contact/CAR authority, and video-integrity gate.

This is an outcome-tuned diagnostic on opened cases, not online adaptation or
generalization. It is paired against the validated radius-2 result
`40303/40304`, whose intervention counts are `5/39/31` and total requested
correction norms are `10/78/62`. Passing requires all three episodes to remain
contact-free, pass CAR, complete the native task without timeout, and use less
total requested correction than the paired fixed controller. Failure will
distinguish two outcomes: collision/CAR failure means the first-warning scale
does not transfer even within an episode; safe task failure means magnitude
reduction alone does not fix task compatibility and justifies neither an MLP
nor denoising guidance. MLP training, online curve search, QP changes, CBF
claims, and new test episodes remain forbidden.

Initial H100 array `40352` retained an apparatus failure in task-3 E42 after
the episode finished. A genuine distal contact geom can have more than one of
L5--L7 in its body ancestry. The result writer incorrectly required exactly
one protected ancestor and raised before atomic output. The repair classifies
the first protected ancestor while walking from the contacted geom toward the
root. It does not remove or relabel the contact. The incomplete array and
dependent validator `40355` remain apparatus history.

Clean H100 array `40357` and independent validator `40360` complete the gate.
Task-2 E42 remains contact-free and passes CAR but times out after 12 radius-1.25
interventions; total requested correction is `15`, exceeding the validated
radius-2 total `10`. Task-3 E42 completes, but radius `0.75` is already
forecast unsafe at the second warning (step 105), and raw L6 contact begins at
step 108 with 31 samples and CAR failure. Task-3 E44 remains contact-free and
passes CAR but times out after 33 radius-1.75 interventions. Its total requested
correction is `57.75` versus `62` for radius 2. Aggregate safe task success is
`0/3`; the strict gate fails.

The root result is not that magnitude is irrelevant. It is that the minimum
safe magnitude is state-dependent: a radius certified at the first warning
cannot be reused after VLA replanning changes the physical state. The E42
collision is not a missed warning—the controller knowingly executed a later
forecast-unsafe radius because this diagnostic froze the first-warning scale.
E44 also contains several forecast-unsafe corrected prefixes without raw
contact, exposing continuing proxy conservatism. The next decisive oracle is
therefore an exact risk curve at later warning states (beginning with task-3
E42 step 105), with abstention/backup whenever no registered magnitude is safe.
It is not MLP training or direct risk-to-force conversion.

One repeatability limitation is retained. In incomplete attempt `40352`,
task-2 E42 completed with seven interventions, whereas clean rerun `40357`
timed out with twelve. The paired initial state and executed actions through
the first intervention are identical, but the first live pi0.5 response hash
differs despite the same registered request seed. Final task-preservation
comparisons therefore require within-allocation paired policy arms or repeated
policy samples; separate server launches are not an exact stochastic pair.
This does not change the strict NO-GO because clean task-3 E42 physically
collides and the other two clean cases time out.

Summary/validation file SHA-256 values are
`e8c3e0986ddde226330c1db182a59dbbeec7fb6d5d7d170c560df5b7defd48dd`
and `f58cd571e37701fc12607cff15df224fd42df07037d6121ddaf2f7cdcbdf9c43`;
payload SHA-256 values are
`3df38b493281aac8af34d27704a8f76d0a537b6d808c62985236137094447ae8`
and `0e9c6e90487c6561dc8b09a0ec31d68502d328fe91216dcf656fd3e2bc27fa55`.

## ADR-0141: Test the registered normal candidate bank at the first unsafe later warning

- Status: complete; strict proxy-certified recoverability NO-GO with physical-authority positive control
- Date: 2026-08-15

ADR-0140 identifies task-3 E42 step `105` as the first later warning where the
first-warning radius `0.75` is already forecast unsafe. Raw L6 contact begins
at step `108` only because the diagnostic deliberately executed that known
unsafe proposal. Before implementing a complete adaptive episode, test whether
the unchanged nine-point normal bank contains any known-safe action at this
changed physical/controller state.

The source is immutable clean result `40357` case 01 with file SHA-256
`4afe40e0754f05749d4ea4aaa71e924018262bfd3d98283027699e53bb574142`
and payload SHA-256
`162109aacc5aa4fccd84c22fabe62e758894d64a723505346609d69f4d48f0d5`.
Because that artifact did not store a standalone step-105 MuJoCo snapshot or
the returned VLA tensor, the apparatus replays its exact executed actions
through step 104 and reconstructs the exact raw/post-AEGIS nominal chunk from
the recorded query-21 QP contexts. Replayed seven-row clearance and nominal
post-AEGIS actions must match the source within `1e-9` before any candidate
outcome is interpreted.

The candidate bank, temporal profile, action limits, original released AEGIS
EE filter, five-action prefix, complete fixed backup, seven-row `1 mm` target,
protected-contact authority, CAR threshold, and timeout censorship remain
identical to ADR-0139. All nine candidates are evaluated because the earlier
task-3 curves are non-monotonic. Selection is by minimum realized post-AEGIS
L2 command change among known exact-safe candidates, not by requested radius.
This gate performs no selected-action execution. If no normal candidate is
safe, only then may a future-witness/controller-aware direction audit be
preregistered. Learning, denoising, QP changes, closed loop, generalization,
and CBF claims remain forbidden.

Initial producer `40365` and independent validator `40366` completed candidate
evaluation but failed the source-state provenance gate. The exact nominal
post-AEGIS chunk reproduced bit-for-bit, and the seven source clearances were
identical as an unordered physical set, but L5 slab rows 0 and 2 were reversed.
The cause is apparatus-only: slab templates are fitted lazily, and the
world-signed dominant PCA axis can reverse longitudinal part indices when the
first fit occurs after the link rotates. The source episode first fitted the
slabs at its initial state; the replay harness first fitted them at step 105.
The repair primes the same rigid body-local slab templates at the settled
initial state before replay. It changes no action, candidate, geometry union,
threshold, backup, physical veto, or selection rule. Attempts `40365/40366`
remain immutable apparatus history and cannot support a scientific verdict.

Replacement H100 producer `40368` and independent validator `40369` reproduce
the source state to `4.441e-16 m` and the nominal post-AEGIS action exactly.
The strict registered gate fails because none of the nine candidates maintains
the seven-row `1 mm` ellipsoid boundary. One radius-1.5 candidate times out;
the other eight are known proxy-unsafe, all with L6 row 3 as the active witness.
Consequently the specified proxy-governed system has no admissible selection
at this later warning and cannot yet support MLP training or closed-loop use.

The physical evidence must be reported separately. Requested radii 1.75 and
2.0 complete the five-action prefix plus fixed backup with zero protected
MuJoCo contact, negligible CAR (`<1.861e-12 m`), and a stable terminal. They
are rejected only because L6 row 3 reaches `-10.263 mm` and `-6.346 mm`
ellipsoid clearance, respectively. Thus the registered normal bank has enough
physical authority to prevent this collision, while the conservative slab
proxy cannot certify the physically safe candidates. The next decisive gate
must audit or replace the row-3 quantitative target against compiled collision
geometry before any learned future-risk model is trained. It is not a larger
normal magnitude, QP, denoising, or learned direction experiment.

Result/validation file SHA-256 values are
`f23b934ab4ddc2c0ddf830538561550f51bf1a7336ddef0e2c28b371e45c193d`
and `7745df484c178daea6c5435486fc219cdc908fb0ba9ab16f658685f12ca88df3`;
payload SHA-256 values are
`78cce6f36970a01d44d898ba21ae3ba84cf98d9619a3053f7470b8f26e4051c5`
and `1a469cf99cdfaa2d26ed019b7df409e1c47fd46cdc9677f4cb8f437ba5cc5bc6`.

## ADR-0142: Audit a compiled-box future-risk target before learning

- Status: completed; strict target gate NO-GO
- Date: 2026-08-15

ADR-0141 proves that the registered normal bank has physical collision
authority at task-3 E42 step 105, but the released single-obstacle perception
MVEE rejects both stable contact-free candidates. Do not fit an arbitrary
millimetre offset to one opened state. Instead, keep the seven certified robot
slab ellipsoids fixed and replace only the obstacle MVEE in a read-only label
audit with the exact union of contact-capable compiled MuJoCo boxes.

For robot ellipsoid `j`, obstacle box `b`, and internal substep `k`, define
the exact bound-constrained quadratic and dimensionless risk

```
d_norm(j,b,k) = sqrt(min_{x in box_b} (x-c_j)^T S_j^-1 (x-c_j)) - 1
q_box = -min_{j,b,k} d_norm(j,b,k).
```

Positive `q_box` means exact solid overlap between the registered robot
ellipsoid and at least one compiled obstacle box. It is deterministic and
continuous piecewise, but it is not metric clearance and must not be reported
in millimetres.

The fixed diagnostic population contains eight existing, post-AEGIS candidate
ledgers from two already opened L6 states. E44 step 110 contributes nominal,
radius 0.75, 1.25, and 1.75; E42 step 105 contributes radius 1.25, 1.50,
1.75, and 2.00. These provide three known-contact controls, four stable
contact-free controls, including at least three perception-MVEE-rejected
actions, and one timeout that remains unknown. Every exact prefix, selected
backup action, and terminal-hold action is replayed from its immutable source
snapshot while proxy rows, raw contacts, CAR, and compiled-box overlap are
measured at every MuJoCo substep.

The gate requires exact source-state/proxy/CAR/physical-veto replay, compiled
non-overlap and zero raw protected contact at both fixed initial states,
compiled overlap for every raw-contact control, zero compiled overlap for all stable
contact-free controls, useful compiled-box-safe support in both states, rescue
of at least three stable actions whose perception-MVEE solid overlap is
negative at zero buffer, and no timeout labeled safe. The source's separate
one-millimetre operational buffer is logged but is not used to manufacture a
geometry mismatch between the two obstacle representations.
If it passes, this authorizes only a simulation-oracle candidate-governor
target test. If it misses a raw contact, the target is unsafe. If it still
rejects stable candidates, the fixed robot slabs or compiled-box target remain
too conservative. Learning, calibration, QP, denoising, selected-action
execution, closed loop, metric-distance, deployment, and generalization claims
remain forbidden regardless of outcome.

H100 producer `40387` and independent validator `40388` reproduce both source
states, all proxy minima, CAR outcomes, and physical vetoes. Both initial
states are compiled-box non-overlapping and contact-free. The compiled-box
target detects all three raw-contact controls with zero physical false-safes
and retains a known safe candidate in both states. It rescues E44 radius 1.25
and E42 radius 2.0: their perception-MVEE minima are `-6.390 mm` and
`-6.346 mm`, while compiled-box normalized slacks are `+0.022739` and
`+0.036349`.

The preregistered three-rescue gate nevertheless fails. E42 radius 1.75 is a
stable, CAR-pass, raw-contact-free rollout, but the fixed L6 row-3 slab still
overlaps the exact obstacle boxes in 24 internal samples and reaches normalized
slack `-0.015667`. Thus replacing the obstacle MVEE fixes two false rejections
but not the complete target: the conservative fixed robot slab remains a
boundary-recall problem. Do not train the future-risk model on this target.
The next geometry gate must replace or refine the L6 robot-side proxy with
compiled collision geoms or validated tight primitives while retaining the
same contact controls. Result/validation file SHA-256 values are
`430980069b0b7e5a1ae9f4b19affce876b44b2ffffe559f209a9326bda0ca463`
and `77226287dca3a0f79eb38cd3ed4d85b809d90274d7a0268225e1a8c6e9314ac2`;
payload SHA-256 values are
`b80c81aacd9d954d892492a4ffc3fd0ae0d2c498ea23d6de147e458f63ff763b`
and `d49fa478d0a14d840c6e30a69fab78024f6de0c28b4423361a6c123272b7b60f`.

## ADR-0143: Calibrate a small empirical L6 proxy shrink as infrastructure

- Status: completed; empirical mechanism gate passes
- Date: 2026-08-15

The geometry mismatch is not the intended research contribution. Calibrate a
transparent development-only heuristic on the immutable ADR-0142 controls:
uniformly scale the two L6 robot-proxy semiaxes by the largest factor in
`[1.0, 0.995, 0.99, 0.985, 0.98]` that detects every raw-contact control,
accepts every stable contact-free control, keeps safe support in both opened
states, and leaves the timeout unknown. L5, L7, obstacle boxes, candidate
actions, backup, contacts, CAR, and AEGIS remain unchanged.

For original normalized radial slack `d`, uniform semiaxis scale `s` gives the
exact transformed value

```
d_scaled = (d + 1) / s - 1.
```

This finite sweep is permitted only because the user explicitly treats proxy
geometry as heuristic infrastructure. Any `s < 1` invalidates the original
mesh-enclosure certificate and must be labeled an empirical non-enclosing
proxy. The maximum permitted shrink is two percent. Passing authorizes only
freezing that opt-in scale for a later simulation mechanism pilot with raw
MuJoCo contact as final authority. It does not authorize a formal safety/CBF
claim, deployment, sealed-test use, MLP training, QP, denoising, or closed
loop.

H100 producer `40389` and independent validator `40390` reproduce the original
scale-1 stable recall of `3/4`. Scales `0.995`, `0.99`, and `0.985` also retain
only `3/4`; scale `0.98` is the largest registered value that accepts all
`4/4` stable contact-free controls while still detecting all `3/3` raw-contact
controls, retaining safe support in both states, and leaving the timeout
unknown. Freeze `0.98` for L6 rows 3--4 only in the opt-in empirical risk-label
config. The original AEGIS EE constraint and L5/L7 geometry remain unchanged.

This resolves the opened-state proxy blocker for a simulation mechanism pilot,
not for a safety certificate. The scaled ellipsoids are intentionally
non-enclosing; every later selected action must still be evaluated with raw
MuJoCo L5--L7 contact and CAR, and newly reserved episodes are required before
any generalization claim. Result/validation file SHA-256 values are
`11109d129b9e988d78eb85fb1cdd2c263520fba67aa65349080264dfd1cfe797`
and `2738d6250e780f517a448c3e6347572527805b9ee15cba7ae5899d9ec04e0a33`;
payload SHA-256 values are
`f175c28699eb8f08eb6e6dcc32be1de57b581f10e1d59f9e481943d1d849fdc0`
and `c25e5f48a85276bcd6ad5e9717b0906f8874f7b758d83d89bb8972d54734fe85`.

## ADR-0144: Test minimum-intervention finite-bank governance before learning

- Status: completed; strict oracle mechanism gate passes
- Date: 2026-08-15

The empirical L6 geometry gate removes the known opened-state label mismatch,
but it does not establish that adaptive normal magnitude is useful. Before
training a future-risk model, replay the complete nine-candidate normal banks
at four already opened warning states: task-2 E42 step 25, task-3 E42 step
100, task-3 E44 step 110, and the later task-3 E42 step 105 state. Each bank
contains the nominal action and registered magnitudes `0.25, ..., 2.0`; no
candidate or controller command is regenerated or changed.

Each immutable post-AEGIS five-action prefix and complete fixed backup is
replayed at every internal MuJoCo substep. Candidate risk uses exact compiled
obstacle boxes, the frozen empirical `0.98` scale for L6 rows 3--4, and the
unchanged L5/L7 rows. Raw protected MuJoCo contact and CAR remain physical
vetoes, the original AEGIS EE projection remains in every source action, and
unknown timeouts remain inadmissible. The oracle chooses the physically stable
candidate with nonpositive empirical risk and minimum realized post-AEGIS
action correction, not minimum requested magnitude.

Passing requires exact source replay, initially safe states, detection of all
stored raw-contact controls, acceptance of all stable contact-free controls,
safe support in all four states, a safe radius-2 control in all four states,
and a strictly smaller selected realized correction than radius 2 in every
state. This is a no-learning, no-execution mechanism test of the proposed
finite candidate governor. It does not authorize an MLP, selected-action
execution, denoising guidance, closed loop, deployment, generalization, or a
formal safety claim. If it fails, the complete-candidate future-risk governor
is not ready for learning even with the repaired geometry target.

H100 producer `40404` and independent validator `40405` pass every registered
gate. Across all 36 candidates, the empirical target detects all `10/10`
raw-contact controls, accepts all `16/16` stable contact-free controls, and
keeps all ten timeouts unknown. Every state has safe support and a safe radius-2
control. The minimum realized-correction choices are:

| State | Requested magnitude | Realized post-AEGIS L2 | Radius-2 L2 |
|---|---:|---:|---:|
| task-2 E42 step 25 | 1.25 | 1.065836 | 1.795073 |
| task-3 E42 step 100 | 0.75 | 0.711527 | 1.933886 |
| task-3 E44 step 110 | 1.25 | 1.104189 | 1.827801 |
| task-3 E42 step 105 | 1.75 | 1.376942 | 1.600365 |

Mean realized intervention falls from `1.789281` to `1.064623`, a `0.724658`
absolute or `40.50%` relative reduction. Thus the exact future-risk candidate
governor adds clear value over always applying maximum repulsion on these four
opened states. This supports the core algorithmic decomposition—generate a
small physical bank, predict complete candidate-plus-backup risk, and select
the least-modifying safe candidate—but does not yet validate a learned risk
model, selected-action task preservation, or generalization. The next gate is
grouped independent-state candidate-risk learning with raw contact/CAR
verification; a QP and denoising guidance remain unnecessary.

Result/validation file SHA-256 values are
`9b3454d60498f52bb1713ef645328c4434bb92582c2ef3da828b77fe113a8283`
and `9404193775f57b19384d65cd4301fa56186be08bee28c8fcadb97ce30dcebc54`;
payload SHA-256 values are
`287fdf6547f28bd44e4ed2dc3bbbd247107522f12db942861fa33af0f2307bc6`
and `45179b197f9eb4f897ec2167106a7a2de22a2da214693211d9c4a0bad0c2dabc`.

## ADR-0145: Collect grouped empirical candidate-risk curves before training

- Status: preregistered; H100 collection pending
- Date: 2026-08-15

ADR-0144 establishes that exact finite-bank governance reduces intervention on
four opened states. The next unresolved question is whether the scalar future
risk of each complete candidate is learnable across independent physical
states. Freeze the same nine normal magnitudes, released AEGIS EE projection,
five-action prefix, complete fixed backup, exact compiled obstacle boxes, and
empirical `0.98` L6 row scale. No direction, action, backup, or threshold may
change during collection.

Use the existing clean action-risk cohort but open only its ten training and
three validation episodes. Diagnostic and test episodes are excluded. For each
episode, select one actual five-action VLA query boundary by

```
floor((first_relevant_contact_step - 3) / 5) * 5.
```

This preserves real query alignment while placing the nominal five-action
chunk close to the known collision. Reconstruct the archived nominal action,
generate magnitudes `0, .25, ..., 2`, pass every proposal sequentially through
the unchanged AEGIS EE filter, and execute the exact prefix plus fixed backup.
Capture complete robot/controller context and the exact final five actions that
reach OSC. Relabel every internal substep using compiled obstacle boxes and the
frozen empirical robot proxy; raw MuJoCo L5--L7 contact and CAR remain physical
authorities.

The strict dataset gate requires 13 cases and 117 candidates, exact replay,
empirically safe initial states, detection of every raw-contact control,
acceptance of every stable contact-free control, two-sided known safe/unsafe
support at every state, recoverability in all ten training and all three
validation states, and censored timeout handling. Passing authorizes only a
prediction-only scalar candidate-risk MLP with episode-grouped splits. Failure
requires target or state-coverage repair; it must not be hidden by dropping
states or timeouts. Test episodes, selected-action execution, QP, denoising,
closed loop, metric-clearance, deployment, generalization, and formal-safety
claims remain forbidden.

The first apparatus canary, H100 job `40429`, stopped before candidate
simulation because the reused source collector still required the deprecated
perception-MVEE initial clearance to exceed one millimetre. ADR-0145 defines
initial eligibility using the compiled-box empirical target precisely because
the old proxy is being replaced. Amend the collector once to bypass only that
old proxy preflight in this opt-in path. Initial raw contact and CAR checks stay
active, and the independently replayed empirical initial-state gate remains
mandatory. No state, candidate, action, AEGIS behavior, backup, target, or
acceptance threshold changes. Job `40429` is apparatus history, not a
scientific result.

Replacement apparatus canary `40432` completes on one H100. It reproduces the
task-2 E19 source exactly, captures a non-null complete controller context,
four compiled obstacle boxes, seven empirical robot rows, and all nine exact
post-AEGIS five-action chunks. The empirical initial state is non-overlapping
and raw-contact-free. The bank contains three known unsafe contact candidates,
four censored timeouts, and two stable safe candidates. This validates the
collection and relabel apparatus without contributing scientific population
evidence. Full producer array `40437` and dependent independent validator
`40438` are submitted from exact commit
`e373a43090c6dc785a99a31a70a5d9fa240e87b8`.

## ADR-0146: Treat monotonicity as a validated regularizer, not a substitute for state

- Status: planned; blocked on ADR-0145 validation
- Date: 2026-08-15

The learned target remains the controller-conditioned scalar risk
`q*(z,A)`. Complete physical/controller context and the exact five actions
reaching OSC are mandatory inputs: no regularizer can identify different
risks for aliased states. Candidate selection also remains unit-consistent:
among candidates predicted below the registered safety threshold, select the
smallest realized post-AEGIS action change. Do not add risk and action norm in
one weighted objective.

Before training, audit every same-state, known non-timeout curve. Requested
alpha is not the monotonic coordinate after sequential AEGIS. Reconstruct the
realized correction relative to alpha zero, project it onto the registered
normal temporal profile, and report its off-axis residual, clipping, and AEGIS
status. Only pairs that preserve the registered direction may support the
physics hypothesis that increasing effective outward correction does not
increase risk. Unknown timeouts and direction-changing pairs are excluded from
the monotonic loss but remain reported.

If ADR-0145 passes and the training curves empirically support this ordering,
run one matched ablation: boundary-weighted risk regression plus AdamW versus
the identical model with a soft pairwise monotonic hinge. Keep inputs, grouped
splits, normalization, architecture, optimizer, seeds, and schedule identical.
Report ordinary/near-boundary error, monotonic violations, threshold error,
false-safes, safe recall, and recoverable-state support. A one-sided optimistic
error penalty is a separate later ablation only if prediction is otherwise
useful; combining it immediately with monotonicity would confound attribution
and could achieve zero false-safes by rejecting every action. QP, denoising,
closed loop, and sealed tests remain blocked.

Priority refinement: isolate ordinary weight decay before adding the monotonic
term. On the identical authorized dataset, representation, grouped split,
normalization, architecture, initialization seed, optimizer, learning rate,
schedule, and boundary-weighted risk loss, compare AdamW `weight_decay=0`
against the established `weight_decay=1e-4`. This is the first training
ablation because it changes only generic capacity control and does not assume
monotone physics. The realized-curve audit remains useful but moves downstream:
only if weight decay alone is insufficient and the audit passes may the
monotonic arm run. No hyperparameter sweep is authorized on validation data.

ADR-0145 H100 producer array `40437` and independent validator `40438`
complete all 13 cases and 117 candidates, but the strict dataset gate is
NO-GO. Raw-contact detection, stable-control acceptance, empirical initial
safety, timeout censoring, split counts, and 10/3 recoverability all pass.
Only four states have known safe and unsafe candidate-plus-backup outcomes;
nine states contain no known unsafe candidate, so two-sided boundary support
fails. Weight-decay and monotonic training are therefore not authorized.

The exact-replay gate also fails only on task-3 cases 07--10. Their simulator
state hashes, initial clearance, and CAR replay exactly, while the deprecated
proxy-clearance comparison differs by at most `1.7459027434885144e-7 m`, above
the frozen `1e-9 m` equality tolerance. Audit this numerical proxy discrepancy
separately; relaxing it cannot repair the independent two-sided-support
failure. Preserve every case and timeout. Collect distinct states where the
complete fixed backup yields both a known unsafe nominal/weak correction and a
verified-safe stronger correction before any training or loss ablation.

## ADR-0147: Test weight decay on the older frozen 354D future-risk MLP

- Status: complete; ordinary weight decay rejected as root-cause repair
- Date: 2026-08-15

The user's regularization question refers to the validated older model whose
complete 354D physical/OSC input achieved about `0.113 mm` training RMSE but
`11.428 mm` grouped-validation RMSE. It does not refer to training on the new
ADR-0145 dataset. Because all required labels already exist, this is a bounded
prediction-only diagnostic and requires no new simulator collection.

Freeze the 43 training and 29 validation labels, complete input, episode
groups, candidate actions, training-only normalization, 32--32 SiLU network,
symmetric boundary-weighted Huber loss, seed `20260814`, AdamW optimizer,
learning rate `1e-3`, gradient clipping, 2000 epochs, and fixed final epoch.
Compare only `weight_decay=0` with the established `weight_decay=1e-4`. The
established arm must reproduce the immutable job-40181 predictions within
`1e-9` or the experiment stops as apparatus failure.

Call weight decay materially helpful only if it lowers grouped-validation RMSE
by at least 10%, strictly reduces false-safes, and does not reduce recoverable
state support. Regardless of outcome, this two-arm diagnostic cannot pass a
safety or control gate. New simulation, monotonic or optimistic loss,
calibration, QP, closed loop, sealed tests, deployment, and neural-CBF claims
remain forbidden.

H100 producer `40489` and independent validator `40490` reproduce the frozen
`1e-4` arm and both stored models with zero prediction error. The matched
results are:

| Metric | Decay 0 | Decay `1e-4` |
|---|---:|---:|
| Train RMSE | 0.112900 mm | 0.112910 mm |
| Grouped-validation RMSE | 11.429146 mm | 11.427894 mm |
| Near-boundary RMSE | 7.218239 mm | 7.220164 mm |
| Validation false-safes | 13 | 13 |
| Supported recoverable states | 3/4 | 3/4 |
| Exact-safe selections | 1/4 | 1/4 |

The established decay improves grouped-validation RMSE by only `0.001252 mm`
(`0.01095%`), worsens near-boundary RMSE slightly, and changes no safety or
support outcome. This falsifies ordinary weight decay as an explanation for
the large grouped-state generalization gap. Do not tune decay further on the
same validation episodes. The next intervention must address independent
state coverage or state/action-response representation; monotonic loss remains
a separate hypothesis and cannot create missing unseen-state information.

Result/validation payload SHA-256 values are
`d466cd3b211ea7b460a2b21de9b773c9eef269cad1017e72cd5fcb398150d978`
and `3620f87398a7f30b457b571a57f64c64147f3fccac01fee8ed0c5aee5b2db6fd`.

## ADR-0148: Begin gradual input ablation with EE start and commanded endpoint

- Status: preregistered; H100 matched ablation pending
- Date: 2026-08-15

Test the user's smallest proposed input before adding any state group. For each
existing candidate define

```
x = [P_start, P_end]
P_end = P_start + 0.05 * sum_k(A_post_AEGIS[k, xyz]).
```

The exact final five post-AEGIS commands are known before execution. The
registered `0.05 m/action-unit` Cartesian scale is applied to displacement
only; no normalization mean is subtracted. Do not use the observed executed
endpoint or another future rollout quantity, which would leak the target.

Freeze the same 43 training and 29 grouped-validation risk labels, episode
split, 32--32 SiLU architecture, symmetric boundary-weighted Huber loss, seed,
AdamW `1e-4`, optimizer, learning rate, 2000 epochs, training-only
normalization, and final checkpoint as the immutable 354D comparator. Call
the endpoint representation promising only if it improves grouped-validation
RMSE by at least 10%, strictly reduces false-safes, and does not lose supported
recoverable states. Regardless of outcome, it cannot pass a safety or control
gate.

If the six-dimensional arm is insufficient, add exactly one group next:
obstacle-relative position and dimensions. Do not jump directly to joint,
controller, geometry-row, waypoint, rotation, or gripper features; this staged
order is intended to identify which physical input first improves transfer.
New simulation, loss changes, hyperparameter sweeps, calibration, QP, closed
loop, sealed tests, deployment, and CBF claims remain forbidden.

H100 producer `40494` and independent validator `40495` complete the matched
6D test with zero replay error. Endpoint-only training RMSE is `0.814920 mm`,
already 7.2 times the frozen 354D `0.112910 mm`; grouped-validation RMSE is
`16.068010 mm`, 40.60% worse than `11.427894 mm`. Near-boundary validation
RMSE is `13.871146 mm` versus `7.220164 mm`. Endpoint-only reduces false-safes
from 13 to 6 by becoming more conservative, but it does not improve supported
recoverable states (`3/4`) or exact-safe selection (`1/4`).

Therefore start/end alone is insufficient. Preserve the staged ablation and
add exactly one physical group next: represent commanded start/end relative
to the obstacle center and append the three obstacle semiaxes, for 9 total
features. Do not yet add obstacle rotation, joint state, controller memory,
link rows, intermediate waypoints, rotation actions, or gripper state.

Result/validation payload SHA-256 values are
`e9c31a9b314d7c0d0ed7b8e64813466f766281a30399db591ad6652f4ad5a601`
and `6c38ec9946591a6a12d71e229e79741b4ce4f0d87480e1898f9665f1fd973ec0`.

## ADR-0149: Add only obstacle-relative endpoint geometry

- Status: preregistered; H100 matched ablation pending
- Date: 2026-08-15

The 6D endpoint arm is insufficient but reduces false-safes through
conservatism. Test the smallest physical context likely to disambiguate the
same Cartesian endpoint across scenes:

```
x = [P_start - c_obstacle,
     P_end - c_obstacle,
     obstacle_semiaxes].
```

This is nine dimensions. Keep all labels, splits, architecture, loss, AdamW
`1e-4`, seed, optimizer, schedule, and checkpoint fixed. Obstacle rotation,
joint/controller state, L5 row transforms/normals, intermediate waypoints,
rotation/gripper commands, and future rollout state remain excluded. The group
is useful only if validation RMSE improves by at least 10% over the 6D arm,
false-safes do not increase, and support does not decrease. If insufficient,
append only the three current L5 clearances next. No new simulation or control
is authorized.

H100 producer `40496` and independent validator `40497` complete ADR-0149
with zero replay error. The compact relative representation obtains
`0.866652 mm` train RMSE, `7.305681 mm` grouped-validation RMSE, and
`6.150915 mm` near-boundary RMSE. It improves validation RMSE by `54.53%`
over 6D and by `36.07%` over 354D. Validation false-safes remain 6, while
safe support improves to `4/4`. Thus obstacle-relative endpoint geometry is a
genuinely useful input group, not merely a source of conservatism.

The arm is still not a safety filter: selecting the minimum predicted-safe
correction is exact-safe in only `1/4` states. Continue the staged attribution
by appending only current L5 row-0/1/2 clearances, producing 12 dimensions.
Keep all other exclusions and hyperparameters fixed.

Result/validation payload SHA-256 values are
`ddad0193514daec83f074b9db047bc70734f7ebbe46b3cad8ef6e63122307f8a`
and `551b2c3537419a074fb7b1b30c8dac2f491fdca0b1537cb5fed5dfc0733ae5a7`.

## ADR-0150: Append only current L5 clearance

- Status: preregistered; H100 matched ablation pending
- Date: 2026-08-15

Add the smallest state quantity directly tied to the prediction target. The
new input is the validated 9D obstacle-relative endpoint representation plus
current ellipsoid clearances for L5 rows 0, 1, and 2. It is 12D total. These
values are measured at the current state and do not leak the rollout target.

Keep data, groups, architecture, loss, AdamW `1e-4`, seed, optimizer, schedule,
and checkpoint fixed. The group is useful only if it lowers validation RMSE by
at least 10%, does not increase false-safes, and retains `4/4` support. All
remaining inputs and every control experiment remain blocked pending this
attribution result.

H100 producer `40498` and independent validator `40499` reject the clearance
addition with zero replay error. The 12D model obtains `0.621411 mm` training
RMSE but `21.860198 mm` grouped-validation RMSE and `14.856301 mm`
near-boundary RMSE. It has 3 false-safes and exact-safe selection in `2/4`
states, but loses safe support in one recoverable state (`3/4`). These apparent
safety gains come from stronger rejection, not better transferable risk
prediction.

Stop the feature-growth sequence here. The 9D obstacle-relative start/end and
semiaxis representation is the best current arm: it has the lowest validation
and near-boundary error and support in `4/4`, although it still has six
false-safes and only `1/4` correct minimum-correction selection. The next step
is a no-training train/validation range and state-distance audit of the three
clearance inputs. The result does not prove that clearance is physically
irrelevant; it shows that adding absolute clearance to this sparse grouped
dataset creates harmful state extrapolation.

Result/validation payload SHA-256 values are
`3d24979b3f54590e194a0f36aac34d821aa8890f1afecfc775ccbb3956e9d5ee`
and `defe40572bd7f339d5c501f653debf59cb967ab0fce3999cfa4935b70c62e67f`.

## ADR-0151: audit clearance support at the physical-state level

- Status: preregistered; allocated audit and independent replay pending
- Date: 2026-08-15

Do not add another input or retrain after the failed 12D arm. First audit the
train/validation distribution of its three added clearances using immutable
9D and 12D artifacts. Collapse candidate actions by `state_id` and require the
clearance vector to be bitwise identical inside each state. Use distinct-state
training moments and ranges for validation z distances; report the original
candidate-weighted normalization separately because unequal candidates per
state can distort it.

Relate this support audit to frozen per-state errors and safe support. If the
state that loses support, or the states where 12D degrades most, are outside
the training range or far from all training states, retain 9D and prioritize
new independent state coverage. If validation clearances are well supported
but 12D still degrades, investigate representation/model interaction instead.
Either outcome is diagnostic only: four validation states cannot establish a
causal relationship, and the audit authorizes no training or control.

Allocated H100 audit `40500` and independent validator `40501` support the
first branch only for the catastrophic E45 failure. E45 row-1 current
clearance is `7.792309 mm`, outside the training range
`22.185141--74.254913 mm` and `-2.39844 sigma`; 12D raises state RMSE by
`38.427610 mm` and removes the safe candidates found by 9D. The other five
validation states retain their prior support behavior, and two other
out-of-range states do not degrade comparably. Pearson correlation `0.8049`
is therefore not sufficient as a broad claim; rank correlation is `-0.0857`.

Conclude that clearance extrapolation explains the 12D support loss, not the
complete remaining generalization problem. Keep the validated 9D
obstacle-relative endpoint representation as the current best arm. If and
only if current clearances are revisited, collect independent states below the
present row-1 minimum and fit state-balanced normalization/sampling before a
matched rerun. Do not add another feature, train, calibrate, or control from
this audit.

Result/validation payload SHA-256 values are
`923c7dc1458ceb4e17b7ec9f1cb9293c9ded55b23fa7e4deacc90f857315db5a`
and `aa4ddd904266aff15db73ca7ef65f1569b523e53aba291dae08020ee415e7c8a`.

## ADR-0152: audit four-task/two-level cohort availability before collection

- Status: preregistered; allocated immutable-artifact audit pending
- Date: 2026-08-15

The requested train-on-three/test-on-fourth protocol cannot be inferred from
the current 18 clean episodes: they cover only Goal Level-II tasks 0, 2, and
3. Before launching new simulation, inspect all 400 immutable Goal-suite AEGIS
results and their hash-bound contact artifacts. Classify each task/level case
with the existing clean L5 rules. Define four folds a priori; holding out a
task excludes both Level I and Level II from training.

Require at least one clean test case in each held-out level and clean training
support from at least two other tasks. If a fold lacks support, do not train or
manufacture balance with repeated actions. Generate new outcome-blind AEGIS
trajectories only for the missing task/level cohorts, then repeat this audit.
If all folds have immutable support, proceed to independent-state two-sided
boundary collection with state-balanced sampling. The availability audit uses
no new rollout and cannot establish prediction or safety.

H100 jobs `40505/40506` independently reproduce a strict population NO-GO.
The 400 immutable Goal-suite cases contain 12 clean task-successful L5 cases,
all in Level II: task 0 has 2, task 2 has 8, task 3 has 2, and task 1 has 0.
All four Level-I cohorts have 0. Thus no held-out task has clean L5 test support
at both obstacle levels, and none of the four folds is scientifically
evaluable as registered.

Do not proceed to state-balanced 9D retraining. Repeated candidate actions
cannot create missing task/level states. Also do not silently search new
policy-noise seeds merely to manufacture failures; that changes the population
and may select on the outcome. The next study-design decision is external to
the model: prospectively expand obstacle/task configurations with outcome-
blind collection and a newly reserved test cohort, or narrow the first claim
to the naturally supported Level-II tasks. The current evidence supports the
latter as a mechanism pilot, not four-task generalization.

Result/validation payload SHA-256 values are
`f343348e281b6db7f60f7dc3a7fb1f00700e54a64c98420aa6f119d66d2700fc`
and `cf8cd4b52d6db5983ab93b7d861a548aea37f5041e5de68f3af42158f8488e06`.

## ADR-0153: audit physical EE primitives before replacing the released proxy

- Status: preregistered; allocated immutable-artifact audit pending
- Date: 2026-08-15

Do not infer that the oversized released EE ellipsoid supplies more learnable
data merely because AEGIS has many CAR failures.  First scan all 1,600
immutable Table-1 AEGIS contact artifacts and classify exact active-obstacle
contacts into palm, finger-1, finger-2, L5, L6, and L7 groups.  Preserve raw
geom identities and report unmapped robot contacts rather than forcing them
into a registered group.

Clean geometry controls must be complete native-task-successful, proxy-valid,
initially safe, and free of active-obstacle contact by dynamic task or other
objects.  Physical robot contact is authoritative even without paper CAR.
Each candidate primitive needs at least three clean contacts from two distinct
task/level groups and three matched clean contact-free controls before a
tighter-geometry audit is allowed.  This audit does not fit semiaxes or choose
a clearance threshold after observing the population.

If palm or finger coverage passes, the next gate fits and independently
validates tighter physical primitives against the frozen contact/contact-free
controls.  Only then may independent two-sided candidate-plus-backup boundary
collection begin.  The released EE proxy stays unchanged in the baseline arm;
training, QP, closed loop, and whole-arm/CBF claims remain blocked.

H100 jobs `40509/40510` validate the full audit with zero case mismatch. Palm,
L5, and L6 pass only the availability gate: clean contacts/cohorts/controls
are `3/2/40`, `20/5/113`, and `16/3/56`. Finger-1 base has `2/1/14`,
finger-2 base `0/0/0`, and L7 `0/0/0`, so those do not pass. The result does
not authorize boundary collection or training.

The unmapped-contact requirement correctly identifies 296 finger-1-pad and
151 finger-2-pad events. They cannot be ignored or admitted as contact-free
controls. ADR-0154 therefore amends only the physical taxonomy: palm,
finger-1 base, finger-1 pad, finger-2 base, finger-2 pad, L5, L6, and L7 are
separate exact geom groups. The same 1,600 immutable artifacts and thresholds
must be replayed before any geometry fit. This correction is motivated by
complete compiled-geom coverage, not by improving a learned outcome.

Result/validation payload SHA-256 values are
`a861c47e5a854fb6b278f2659aa55627a7c16689e8819f9bbbd7c9eb39af594e`
and `5ff6d96eb21c64d83d72d08bdd33bb3288fa113eb60c677a31cc8df11e4973cc`.

## ADR-0154: split finger base and pad primitives before geometry fitting

- Status: preregistered; allocated immutable-artifact replay pending
- Date: 2026-08-15

Use one constraint identity per contact-capable compiled EE geom. A finger
base and its tip pad are separate rigid physical primitives; combining them
into one enclosing ellipsoid could recreate the released proxy's empty-space
problem. Replay the unchanged 1,600-case availability audit with all four
finger geoms explicitly registered. Require zero unmapped robot contact events
before any palm/finger fit. Population, clean eligibility, support thresholds,
and all downstream prohibitions remain unchanged.

H100 producer `40513` and validator `40514` pass the corrected taxonomy with
zero case mismatch and zero unmapped contact events. Palm/L5/L6 remain the
only availability-ready groups. Palm is only minimally supported (`3` clean
episodes, `2` task-level groups); L5 and L6 have `20` and `16` clean episodes.
No finger pad/base except finger-1 base has even one clean task-successful
contact cohort, and finger-1 base has only `2` episodes in one group. L7 also
has no clean support.

Do not treat the larger raw EE failure count as a larger supervised dataset.
Proceed with a tighter palm geometry audit while retaining finger/L7 as
analytic and raw-contact diagnostics. A future shared constraint-conditioned
pilot may claim learned palm/L5/L6 risk only after palm geometry and two-sided
boundary support pass; it cannot claim learned fingers, L7, or whole-arm
safety from this population.

Result/validation payload SHA-256 values are
`a8f41767e98d7c03076f626ce1471d7377883bfed5cb3c36871edd1f6fc12466`
and `b62a81428d5e110e77a7fd1fb5a835048bb28c9c3df72c9fa151016c874cce88`.

## ADR-0155: fit palm geometry from the compiled collision geom only

- Status: preregistered; allocated replay pending
- Date: 2026-08-15

Fit a tight palm primitive from `gripper0_hand_collision` compiled mesh
vertices without inspecting contact labels. Freeze the three clean palm
contact cases and all 40 matched clean contact-free controls selected by the
validated v2 availability audit. Replay the exact released-AEGIS actions and
measure the fitted palm primitive, released EE proxy, and raw palm contact at
every internal MuJoCo substep.

Require a certified enclosure of all compiled palm vertices, exact replay of
the frozen action-boundary contact ledger, reproduction of three internal
palm-contact episodes and 40 internally contact-free controls, and zero
physical false-safe samples for the tight primitive. Report contact-free
acceptance for both tight and released proxies without tuning geometry to the
controls. A pass authorizes only two-sided boundary collection for palm, L5,
and L6; training still waits for grouped boundary coverage. A failure keeps
collection blocked and identifies whether the remaining error is replay,
robot primitive, or the unchanged obstacle perception proxy.

Canary `40531` resolves that ambiguity before the full array. E09 contains 28
internal raw palm-contact samples and every contact point lies inside the
certified tight palm primitive, but the unchanged static obstacle MVEE is
separated at all 28 samples. The first contact has tight/static-MVEE support
gap `+25.427 mm`; the released EE proxy is also falsely separated by
`+9.257 mm`. Conversely, the contact-free E01 control changes from
`-36.861 mm` under the released proxy to `+5.673 mm` with the tight palm.
Therefore ADR-0155 is a strict end-to-end NO-GO caused by the obstacle side,
not evidence against the compiled palm fit. Do not spend 43 replays on a gate
that the first positive already disproves.

## ADR-0156: isolate palm with live compiled-obstacle primitive union

- Status: preregistered; two-case allocated canary pending
- Date: 2026-08-15

Keep the ADR-0155 palm fit, cohort, actions, replay, and raw-contact authority
unchanged. Replace only the static released obstacle MVEE in the primary
diagnostic with a live union of certified bounds for every contact-capable
compiled obstacle geom. Meshes receive outcome-independent compiled-vertex
MVEEs; sphere/ellipsoid/capsule/cylinder/box primitives use the registered
exact or Loewner enclosure. Retain both released proxies as comparators.

This is privileged simulator geometry and can validate only whether the palm
primitive supplies usable quantitative simulation labels. Run E09/E01 first;
only if it has zero physical false-safes, exact replay, and accepts the control
may the unchanged 3/40 population and independent replay proceed. Even a full
pass authorizes boundary collection for palm/L5/L6, not training or control.

Compiled-union canary `40539` passes contact detection but fails usefulness:
E09 has zero false-safes and `-70.002 mm` minimum primary gap, while contact-
free E01 is rejected for 768 samples with `-21.824 mm` minimum. This is the
expected conservatism of intersecting separately enclosing volumes, not a
license to tune their scales from two outcomes. Stop this arm before the full
43-case array.

## ADR-0157: track the frozen obstacle MVEE pose without changing its shape

- Status: preregistered; two-case allocated canary pending
- Date: 2026-08-15

Keep the tight palm, released obstacle-MVEE shape, cohort, actions, raw-contact
authority, and zero-false-safe requirement unchanged. At the settled state,
record the obstacle MVEE center and rotation in the active obstacle root-body
frame. During replay, update only that rigid pose from exact MuJoCo state. Do
not change semiaxes, threshold, clearance buffer, or fit based on contacts.

This privileged pose-tracking audit directly tests the canary diagnosis that
the released static obstacle proxy became stale after physical displacement.
Require the E09/E01 canary to reproduce replay, detect every palm contact, keep
E01 predicted safe, and improve control acceptance over the unchanged released
proxy. Only then run the 3/40 independent replay. This remains a simulation
label mechanism, not deployable perception or learned safety.

H100 canary `40541` is a strict NO-GO for ADR-0157. E01 is accepted at
`+5.673 mm` with zero contact, but every one of E09's 28 raw palm-contact
samples remains false-safe and the first-contact gap stays `+25.427 mm`.
Therefore exact pose tracking does not repair the released obstacle shape.
Stop before the full cohort and do not inflate/shift the obstacle MVEE from
these outcomes.

The next scientific choice is now explicit. If the intended target remains a
quantitative future violation, first integrate and validate a consistent
compiled-geometry distance backend such as FCL/GJK. If the intended online
controller is the finite candidate governor without gradients or QP, a
separate experiment may instead learn constraint-conditioned future physical
contact probability from raw complete candidate-plus-backup rollouts. Do not
silently mix these two targets. In either case, the certified tight palm may
remain as relative-geometry input, but it is not an end-to-end safety value
with the released perception MVEE.

## ADR-0158: Freeze exact compiled-box radial slack for palm/L5/L6

- Status: preregistered; paired H100 canary pending
- Date: 2026-08-15

Do not fit another obstacle ellipsoid. The existing active-set solver already
computes, without iterative tolerance, the exact represented-geometry value

```
h(E, B) = sqrt(min_{u in [-1,1]^3}
  (c_B + R_B D_B u - c_E)^T S_E^-1
  (c_B + R_B D_B u - c_E)) - 1.
```

This value is dimensionless: positive means the represented robot ellipsoid
and solid compiled box are separated, zero is their exact boundary, and
negative means overlap. The obstacle target is the minimum over every live,
contact-capable compiled MuJoCo box. No perception MVEE and no enclosing
ellipsoid union participates. Unsupported obstacle geom kinds fail closed
before replay; a later convex-distance backend must be preregistered if the
population contains them.

Keep the outcome-independent certified tight palm MVEE and the frozen
certified L5/L6 slab primitives. The paired canary replays immutable E09 and
E01 from the same Table-1 state/action ledgers at every internal MuJoCo
substep. E09 contains observed palm, L5, and L6 contacts; E01 is contact-free.
The canary asks only whether every observed group contact has `h <= 0`, every
primitive certificate and exact replay passes, and the control retains
positive represented clearance. Raw MuJoCo contacts remain the independent
physical authority.

Only a passing canary permits the unified final cohort: all 31 unique clean
palm/L5/L6 contact episodes plus three shared matched contact-free controls.
Boundary collection and
the three-output future-risk MLP remain blocked until the complete geometry
gate passes. Gradient correction, QP, denoising, closed loop, deployment,
metric-clearance, and formal safety claims remain forbidden. This is a
privileged-simulation mechanism target; perception geometry is downstream.

H100 canary array `40548` passes the paired represented-geometry mechanism
gate. E09 reproduces 28 palm, 169 L5, and 2 L6 contact samples; every contact
has nonpositive exact radial slack, with group episode minima
`-0.056608/-0.322074/-0.037365` for palm/L5/L6. Contact-free E01 has positive
minima `+0.753981/+0.047423/+0.217761`, all primitive certificates pass, and
replay is exact. Thus the obstacle MVEE and enclosing-union failures are
removed without threshold fitting. This authorizes only an optimized exact-
arithmetic replay of that 31/3 cohort and independent validation; learning and
control remain blocked.

Optimization attempt `40556` failed its allocation-side unit preflight before
simulation because Numba does not permit an import opcode inside a compiled
kernel. Replacement `40558` failed the clean-source preflight because the
submitted full commit string was mistyped. Both are retained apparatus
history. Correct replacement `40560` passes the scalar-versus-batched exact-
arithmetic test and reproduces every canary value, reducing E09/E01 wall time
to `1:45/1:40`. No scientific setting changed.

The final unified producer `40562`, dependent independent replay `40563`, and
aggregate verifier `40564` are submitted from exact clean commit `2c84f96`.
The cohort contains all 31 unique availability-audit clean palm/L5/L6 contact
episodes and three shared matched contact-free controls. Arrays are capped at
two H100s; training remains blocked until verifier `40564` passes.

Producer `40562` is retained as apparatus history rather than scientific
evidence. Three cases stopped before replay measurements because an unused
released-obstacle MVEE comparator supplied an orthogonal basis with determinant
`-1`, which the proper-rotation data class rejected. Canonicalizing the column
sign (and only negligible serialization drift) preserves the ellipsoid shape
and cannot alter the exact compiled-box target. Do not combine output across
the old and repaired commits: restart the complete 34-case producer and its
independent replay, then apply the unchanged aggregate gate.
