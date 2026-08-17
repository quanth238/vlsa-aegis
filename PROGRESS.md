# AEGIS SafeLIBERO table reproduction

## Grouped real-query boundary coverage gate (2026-08-14)

The next data gate is implemented without executing candidates or authorizing
learning. It enumerates every real pi0.5 query boundary before the first
protected contact in the 15 diagnostic/train/validation episodes while
excluding the previously inspected test split. At each boundary it restores
the immutable released-AEGIS trajectory state, requires the current seven-row
proxy margin to be at least +1 mm with zero protected contact and paper CAR,
and evaluates the exact next five archived AEGIS commands over all 25 internal
MuJoCo substeps per action. A state is retained only when this nominal prefix
violates a seven-row proxy constraint, protected contact, or CAR.

The producer records the bound policy query index/seed/hash, complete source
state hash, seven current clearances, seven exact nominal-prefix row minima,
and the active row/action/substep witness. The validator recomputes retention
and aggregates explicit L5 slab 0--2, L6 slab 0--1, and L7 slab 0--1 counts.
Unsupported rows remain zero; nominal far-safe values cannot fill boundary
coverage. Candidate execution, MLP training, calibration, QP, and closed-loop
control are forbidden by the versioned contract. Focused local tests and shell
syntax pass. The local full suite reaches all tests but the desktop runtimes
lack optional CVXPY; the H100 allocation preflight uses the registered safety
runtime. The next command is an allocated-H100 E05 canary, followed by the
15-episode array only if the canary reproduces query 37 / action 185 and its
known L5 row-1 nominal warning.

Allocated-H100 canary `39628` passed on `worker-2`: E05 retained exactly
action 185 / query 37, with a current minimum of `+12.277061 mm` and exact
nominal-prefix minimum of `-16.075686 mm`. The active witness was L5 slab row
1 at five-action offset 3, internal substep 1. Grouped H100 array `39629`
then completed all 15 non-test episodes with exactly one valid retained query
boundary per episode. Independent H100 validator `39650` reproduced all
retention and row counts.

Coverage is a strict NO-GO for seven-output training. Active witnesses occur
only on rows 1, 2, and 3: L5 row 1 has 1 active witness / 5 violated states;
L5 row 2 has 7 / 11; L6 row 3 has 7 / 7. Rows 0 and 4 have zero violation or
active-witness support, and both L7 rows 5--6 have zero support. The validated
artifact interpretation is `query_boundary_coverage_incomplete_do_not_train`.
Validation result file SHA-256 is
`56ad37586f33f426e6fe920f6a5c5e67527acee81bb59f1d3140cf6d313dbf0c`;
payload SHA-256 is
`2c47f1bb17c5de676197ed58c320521c6f5a1d3cf0042f565dbf1ba0a06e14d7`.
The next allowed action is frozen 37-candidate-plus-complete-backup collection
on these 15 retained states, reporting safe/unsafe/near-boundary and active
combined-risk witnesses per row. Training remains unauthorized.

## Query-aligned E05 five-action risk diagnostic implementation (2026-08-14)

The versioned replacement apparatus is implemented without modifying released
AEGIS or any Table-1 artifact. It restores the exact E05 state before real
pi0.5 query boundary 185 (query index 37), binds the archived five actions that
released AEGIS actually sent to OSC, and applies a frozen 37-member structured
translation bank: nominal plus paired obstacle normal/up/side directions,
constant/front-loaded unit-L2 temporal profiles, and radii 0.5/1.0/2.0.
Rotation and gripper commands are preserved, endpoint cancellation is not
imposed, and every candidate executes all five actions.

After a candidate prefix, the apparatus repeatedly executes a deterministic,
VLA-ledger-independent registered geometry backup. Each backup step evaluates
hold, world-axis, and local normal/tangent one-action branches with a fixed
five-hold lookahead, executes exactly one selected action, and recomputes from
the changed state. It terminates only as `SAFE_TERMINAL` after +5 mm release
and a verified ten-action stable hold, `UNSAFE_CONTACT_OR_CAR`, or
`UNKNOWN_TIMEOUT` after ten backup actions. Fixed k0 is an eligibility check,
not part of action-dependent prefix risk. Seven candidate-prefix, backup, and
combined Monte Carlo risks are stored separately; raw contact and CAR remain a
physical veto. The full diagnostic cannot authorize training by construction.

Local static checks pass (`py_compile`, JSON contract, pure candidate/risk
checks, shell syntax, and `git diff --check`). The next step is an H100 Slurm
canary with nominal plus the first away-normal candidate, followed by the full
37-candidate E05 gate only if snapshot, terminal, and internal-substep binding
pass. The scientific pass requires dangerous nominal risk, mixed safe/unsafe
support, at least one safe terminal, no timeout labeled safe, and no
proxy-safe physical collision.

Allocation-backed canary `39607` completed on H100 host `worker-1` in 1:34.
The explicit CPU canary did not consume another user's allocated accelerator;
the artifact records all eight host H100 identities and remains non-scientific.
The apparatus passed: two repeated nominal five-action rollouts were bitwise
identical in complete dynamic state, every clearance sample, contacts, and
CAR. State 185 was initially safe at `+12.277061 mm`. The nominal chunk became
unsafe at `-16.075686 mm`; the first registered radius-0.5 normal correction
improved it to `-13.105013 mm` but remained unsafe. Both correctly terminated
as `UNSAFE_CONTACT_OR_CAR`, with no timeout mislabeled safe and no proxy-safe
physical collision. This authorizes the frozen 37-candidate diagnostic, not
training. Full allocated-H100 job `39608` is queued from commit `b458bf5`.

The full mechanism gate passed and was reproduced. CPU-bound Slurm diagnostic
`39613` evaluated the frozen 37-member population in 6:03; allocated-H100 job
`39618` reproduced it in 6:15. After removing only allocation/source/timing
provenance, the two JSON artifacts are byte-identical. The H100 result file
SHA-256 is `da347b1ab8ee4a9615b80f866aa3628aad80b2777b7a22e267ccda6f719bfb00`
and its payload SHA-256 is
`f0a6732968e2dc8ad0561b324d64ab94c4f361240b599ae1f08d345b1dfbc615`.

The population is nonvacuous: 2/37 candidates reached `SAFE_TERMINAL`, 32/37
hit protected contact or CAR, and 3/37 reached the fixed timeout and remained
fail-closed. Nominal fell to `-16.075686 mm`. The two safe candidates were
positive obstacle-normal radius-2 corrections: constant achieved
`+1.009345 mm`, while front-loaded achieved `+2.193830 mm` with an applied
norm of `1.961183` after action clipping. Both required zero backup actions
and passed the ten-action stable hold, raw protected contact, and CAR gates.
Radius 0.5 remained around `-12--13 mm`; radius 1.0 improved to about
`-6--7 mm` but did not recover within the registered backup horizon.

Independent validator `39621` passed candidate identity/order, immutable
Table-1 and policy-query binding, exact 126-sample prefix traces, seven-row
risk composition, terminal semantics, physical veto reconstruction, timeout
handling, and mixed support. Validation file SHA-256 is
`53fcfbda01db3d14f421d4835958ef3466a8098194933a29a10fe9976f634769`;
payload SHA-256 is
`be2eaf86795f0f9178ad15fe9c771d1afa6ec8d6a2b3144e6af0dd6781943c10`.

This is a decisive pass for the corrected data object on one E05 L5 state,
not a learned-filter or L5--L7 generalization pass. Every one of the 37 active
combined witnesses was slab row 1 (L5); L6/L7 coverage is zero. Training
remains unauthorized. The next permitted gate is grouped episode collection
with real query-boundary discovery, the unchanged complete backup/terminal
semantics, immutable episode splits, and explicit per-row coverage. Unsupported
episodes and absent L6/L7 witnesses must remain visible rather than being
dropped or relabeled.

## Query-aligned five-action Monte Carlo risk redesign (2026-08-13)

The method draft and data memo supersede the one-action-plus-one-backup-plus-
hold pilot. The learned object for the next gate is a direct seven-output,
backup-policy-conditioned action risk for one complete five-action Cartesian
chunk. It is not a joint-trajectory predictor, binary contact classifier,
force imitator, TD target, or QP surrogate.

The prior collector is reproducible apparatus evidence only: it selects a
single action, executes one backup action, and then holds. That target is not
the specified
`five-action candidate -> repeated deterministic backup -> terminal status`
value. Pending corrected selector job `39598` was inspected and canceled
before allocation when this mismatch was identified. CPU-only Slurm preflight
`39599` independently confirmed that allocation-backed MuJoCo work can run on
the H100 worker pool without consuming a GPU device; no simulation ran on the
login node.

The replacement data unit starts only at a real pi0.5 query boundary. The
archived Table-1 artifact stores hashes for each returned ten-action tensor and
the exact five `nominal_raw` actions actually scheduled before the next query;
it does not store the unused five returned actions. Therefore archived pilots
may use only that exact scheduled five-action prefix. New collection must also
store the complete returned tensor. Every candidate resets the identical full
simulator/controller snapshot, preserves nominal gripper commands, executes
all five actions through the unchanged released AEGIS-plus-OSC stack, and is
then followed by the complete ledger-independent backup until exactly one of
`SAFE_TERMINAL`, `UNSAFE_CONTACT_OR_CAR`, or `UNKNOWN_TIMEOUT`.

The direct Monte Carlo target is seven fixed slab-row risks

```text
R_j(x,A) = max(candidate-prefix violation j,
               complete-backup violation j).
```

Timeout is never safe. The dataset stores prefix and backup values separately,
terminal successor state, every internal joint/margin/contact/CAR sample,
active witnesses, clipping/normalization, policy-query identity, snapshot and
replay hashes. Ellipsoid risk remains the learned target; raw protected contact
and CAR are a distinct physical veto. Any proxy-safe physical collision is a
geometry-cohort failure, not a negative example to hide.

Warning-state discovery is deterministic but adaptive: enumerate real query
boundaries before first contact and retain an initially safe state only when
the archived nominal continuation is dangerous and the frozen structured
candidate bank contains both safe and unsafe outcomes. Report every episode
without such support. Candidate residuals use paired signs, fixed radius
ladders, smooth temporal bases, and separate obstacle-normal/tangent/
translation/rotation arms; independent per-timestep noise is forbidden.
Physical displacements use scale only, never mean subtraction. Previously
inspected E05/E10/E42/E44 remain diagnostic. New complete episodes must be
reserved before final model selection.

Before collection scale-up, one diagnostic H100 oracle must prove all of:
query/snapshot determinism, mixed candidate support, a complete terminating
backup, no timeout labeled safe, and nonvacuous near-boundary witnesses. Per-
row witness coverage must be reported; if L7 has no active/unsafe support, the
claim must remain L5/L6 rather than silently generalizing to L7. Only a
validated grouped dataset can authorize the simple seven-output Monte Carlo
MLP. TD bootstrapping, Poisson geometry, calibration, QP, and closed loop remain
downstream.

## Clean task-successful action-risk dataset (preregistered, 2026-08-13)

The next learned-filter gate uses only cases where released AEGIS completed the
native SafeLIBERO task yet failed paper CAR through an L5/L6/L7 contact.  The
immutable screen excludes settled contacts, gripper contacts, task/other-object
contacts, incomplete or invalid MVEEs, the known E38 proxy mismatch, and any
allocation-audited state below the +1 mm initial boundary.  E05/E10 are
diagnostic only; three untouched milk-obstacle episodes are the final test.

Each of 18 episodes contributes one state three actions before first
protected contact.  Exactly 26 proposals per state execute one action, then
recompute and execute the fixed ledger-independent backup before its registered
25-action hold.  Seven positive-is-unsafe worst-future proxy risks,
raw protected contacts, and CAR are recorded at all internal MuJoCo substeps.
Episode groups never cross splits.  Training remains blocked until every
requested state is initially safe and has an exactly verified safe candidate.
QP and closed loop remain blocked until a later seven-output MLP has zero
observed false-safes and useful safe-action support on untouched episodes.
The prediction-only trainer is itself fail-closed: it refuses to start unless
the independently validated exact dataset records `MLP_training_authorized`.
Its fixed 167D input, 256/256/128 SiLU architecture, boundary-weighted Huber
loss, grouped splits, and untouched metrics are registered in the same config.

Exact next command after committing and syncing clean source:

```bash
EXPECTED_GIT_COMMIT=<commit> RUN_ID=clean-action-risk-20260813a \
  sbatch slurm/distal_clean_action_risk_collect.sbatch
```

H100 canary `39465` failed closed before writing labels because separately
cloning a proposal successor omitted wrapper/controller execution state: the
selected backup suffix differed when embedded after the proposal. This is an
apparatus failure, not a dataset result. The registered repair evaluates every
`proposal -> backup action -> hold` branch uninterrupted from the same saved
state and selects the fixed backup using that branch's successor suffix.
H100 composition canary `39471` passed the corrected apparatus at E05 step 167
in 1:54: all 25 branches shared the proposal successor within
`8.326673e-17 m`, the nominal proposal plus selected local-normal backup kept
`+147.467828 mm` minimum proxy clearance, zero protected contact, and
negligible CAR. The artifact is correctly marked `scientific_result=false`
and cannot authorize training. The full case receives a four-hour allocation
based on this measured rollout cost.
Full diagnostic E05 job `39474` passed the apparatus in 53:05, but all 104
proposal-plus-backup candidates were exact-safe. This validates early-state
recoverability but gives no unsafe classification boundary, so the planned
MLP gate would be vacuous. Do not scale that population. Move the registered
states to 5/3/2/1 actions before first protected contact, retain strict safe
`k=0`, and additionally require at least one unsafe candidate per state.
Near-contact E05 job `39487` then isolated the clean mixed boundary: step 182
had 26/0 safe/unsafe outcomes, step 184 had 21/5, step 185 had 0/26, and step
186 was initially below +1 mm. Freeze one state at the three-action lead
(E05 step 184) for all episodes. E05 is diagnostic-only; untouched test
outcomes were not used to select this offset.
Grouped H100 array `39490` completed all 18 allocations. E05/E10 again passed
with 21/5 safe/unsafe candidates. Most milk/L5 states were already outside the
+1 mm initial proxy boundary and were retained as scientific NO-GOs; two milk
cases passed mixed support. Seven milk/L6 cases stopped before atomic output
because a separately stepped proposal endpoint did not match the same proposal
inside a composed branch. These are apparatus failures, not dataset outcomes.
The repair derives successor clearances and ellipsoid centers from the first
uninterrupted hold composition and requires all remaining branches to match it.

H100 L6 canary `39512_10` then passed the repaired binding in `1:40`: all 25
embedded backup branches reproduced the proposal successor with exactly zero
clearance error. Full grouped array `39516` completed all 18 case artifacts on
the H100 pool. The rollout apparatus now works for both L5 and L6: for example,
E09 produced 11 safe/15 unsafe candidates and E34 produced 16/10, with every
proposal successor exact, every label finite, and no source-state mutation.

Independent structural validator `39565` nevertheless issued the registered
dataset NO-GO. Seven of 18 requested states were already outside the strict
+1 mm initial proxy boundary. The surviving split contained 6/10 train states,
1/3 validation states, and 2/3 test states. Moreover, several accepted L6
states were uniformly safe (26/0), including both surviving test states, so
the final false-safe test would be vacuous. Totals were train 115 safe/41
unsafe candidates, validation 16/10, diagnostic 42/10, and test 52/0. The
validation payload SHA-256 is
`6c50d9a3851a8ff7fd07e7326725142d0a012f676a4bb30b6a7f73b7bd6610af`;
the file SHA-256 is
`3c26f7771beb5ee3a36828bb3673935e92efd6218e11834ad8b7711cb3967e43`.
`MLP_training_authorized=false`; QP and closed loop remain forbidden.

The scientific root cause is the state-selection rule, not the exact backup:
three actions before contact ranged from already unsafe in most L5 episodes to
roughly +20--32 mm in many L6 episodes. Time-to-contact is not a consistent
geometry warning coordinate. The next gate must audit a clearance-based,
strictly pre-contact warning-state rule using diagnostic/train/validation
episodes only, then reserve new complete episodes for the final untouched
test. The already inspected E42/E44 episodes remain diagnostics and cannot
support a later untouched generalization claim.

The replacement selector is preregistered before its outcomes are observed.
For each diagnostic/train/validation episode, replay the immutable raw AEGIS
ledger only until its first protected contact, find the first state whose
seven-row proxy minimum is strictly below `+1 mm`, and select the state exactly
two actions earlier. Reject the case rather than shifting the state if that
selected state is below `+1 mm`, already has protected MuJoCo contact, or has
more than `1 mm` active-obstacle L1 displacement. This H100 audit evaluates no
candidates, trains no model, and excludes the current test split. It passes as
a population selector only if it yields strictly safe pre-contact states
without case-specific offsets; exact mixed candidate support remains a later
gate. After freezing the rule, new complete episodes—not E42/E44—must be
reserved for the final untouched test.

Exact next command after committing and syncing clean source:

```bash
EXPECTED_GIT_COMMIT=<commit> RUN_ID=clean-warning-state-canary-20260813j \
  sbatch --array=0%1 slurm/audit_distal_clean_warning_states.sbatch
```

H100 canary `39572` reproduced E05 exactly: the first boundary crossing was
state/action index 186 and the selected index 184 had `+26.138684 mm`, zero
protected contact, and negligible obstacle motion. Full array `39576`
completed all 15 artifacts, but review found that its trace stopped at the
state before action `contact_step-1`; it omitted the still-pre-contact state
immediately before the contact-causing action. Five apparent L6 warning misses
are therefore apparatus-only and are not a scientific proxy NO-GO. Pending
validator `39594` was inspected and canceled before allocation. Version 2
includes state index `contact_step`, retains the identical frozen selector,
and must replace the invalid array before interpretation.

## Matched task-successful proposal counterfactual (preregistered, 2026-08-13)

H100 producer `39376` and fresh action-ledger validator `39378` executed 300
exactly reproduced actions with zero contact/CAR and +19.888 mm minimum proxy
clearance, but the newly sampled pi0.5 mode made zero task progress. No warning
activated and no repulsion was applied. The next focused counterfactual freezes
the task-successful but action-223-colliding proposal ledger from job `39354`
and changes only the repeated normal backup. This removes stochastic policy
sample confounding; it is not live VLA feedback after intervention.

Matched producer `39380` and replay validator `39381` applied six contact-free
repulsive windows but falsely stopped at action 262. The nominal and every
normal candidate shared the identical +4.496559 mm minimum because the metric
included immutable `k=0`; the latch was still active below its +5 mm release.
This is a measurement-definition NO-GO, not lack of action authority. The
registered correction keeps `k=0` for current-state eligibility and evaluates
candidate improvement only on future samples after command authority begins.

Producer `39382` confirmed the same +4.496559 mm plateau even after excluding
`k=0`: the full five-action nominal stayed physically safe and did not reduce
clearance. The remaining stop was a hysteresis logic error. The +1 to +5 mm
deadband must keep the backup armed while allowing a freshly verified safe
nominal; it must not demand a nonzero repulsive improvement.

Final matched H100 producer `39384` and fresh action-ledger validator `39385`
executed 288 exactly reproduced actions with eight normal-repulsion windows,
zero protected contact, paper CAR pass, and +1.037969 mm minimum executed
L5--L7 proxy clearance. The immutable task-successful proposal ledger was
exhausted at action 292 without native task completion. This is a validated
safety-mechanism pass and safe-task-success NO-GO: repeated normal repulsion
can keep this proposal sequence inside the tested safe set, but the frozen
post-intervention suffix is not a task-recovery policy. Result and validation
payload SHA-256 values are
`06f03ae1b8879c82a104fde47cfca63f4eddaec42f04e4f87fddfae22646fdec`
and `3bb1561c2258d65a6be74b4e44a378c25a937f7652026aefa2a5a9a54b4df62a`.
H100 single-environment media replay `39393` matched all 288 next-state hashes
and produced 289 uncorrupted frames; its video receipt payload is
`af9a69f4e6b74941733ff3b60cc2abc754e26ffc414b4472aef0cb47ede187f2`.

## Fixed PNCBF-style backup-policy oracle (preregistered, 2026-08-13)

The next E02 gate fixes one reproducible continuation policy before learning a
maximum-over-time policy value. It retains the registered contact-free prefix,
then uses fresh frozen pi0.5 plus original AEGIS and a hysteretic five-action
L5--L7 warning. While active, it tests only constant obstacle-normal repulsion
at six frozen magnitudes and executes the smallest physically safe buffered
push, falling back to the best physically safe clearance improvement. There
are no detour modes, endpoint constraints, model training, or new QP.

## One-shot repulsive prefix: task recovery passes, safety fails (2026-08-13)

H100 producer `39354` and independent stochastic replay `39358` completed from
clean commit `39a78e7`. The five-action repulsive prefix was contact-free and
the frozen VLA completed the task in both runs. With no further L5--L7
intervention, L5 contact began at action 223 in both runs and CAR failed. The
primary run completed at action 287; replay completed at 248. The prefix
therefore delayed the original collision and preserved task recoverability,
but did not make the continuation safe.

The next gate is not a larger one-shot correction or endpoint-preserving
detour. Construct an offline policy-conditioned recoverable set from complete
prefix-plus-continuation rollouts, then test hysteretic warning-triggered
reapplication. Exact ellipsoid/compiled-box overlap began at action 190, 33
actions before raw contact, while the more conservative Loewner proxy became
negative at action 183. Keep these warning and physical authorities separate.

The two matched-seed pi0.5 servers produced different action chunk hashes.
Future data must record the realized chunk and final claims must account for
policy randomness.

## One-shot repulsive-prefix/live-VLA recovery gate (preregistered, 2026-08-13)

Execute the independently validated raw-contact-free `soft_free_5d` prefix at
E05 actions 182--186 without endpoint preservation, then query frozen pi0.5
from the changed observation and continue with the unchanged released AEGIS EE
QP. No additional L5--L7 repulsion, candidate rejection, or stopping is
allowed. This isolates whether live policy feedback absorbs the displacement.
All internal substeps after activation report raw protected contacts, CAR,
Loewner-proxy margins, and exact ellipsoid/compiled-box overlap separately.

Exact next command after committing and syncing clean source:

```bash
EXPECTED_GIT_COMMIT=<commit> RUN_ID=soft-prefix-live-replan-e05-20260813a sbatch slurm/distal_soft_prefix_live_replan_e05.sbatch
```

H100 job `39352` stopped before simulation because the evaluator imported the
protected-link predicate from the baseline module instead of its registered
replay module. The run contains only apparatus-failure evidence. The import was
corrected and an evaluator-import check was added to allocation preflight.

H100 job `39353` also stopped before simulation: the registered file SHA was
correct, but the payload hash copied into the preregistration was incomplete.
The immutable artifact reported
`3ac21494b469de9a7d60f702e9b6a19f12830ee73e2b2412f309c446496c7e98`.
The registration was corrected and a pre-server payload check was added.

## E05 Cartesian controllability/manifold oracle (preregistered, 2026-08-13)

The active no-learning gate first audits the archived action-182 state at all
25 internal MuJoCo substeps per action. It separately records the conservative
compiled-box Loewner margin, exact robot-ellipsoid/compiled-box overlap, raw
protected contacts, CAR, and L5--L7 center motion under ten paired basis
probes. Search proceeds only if the initial state is physically safe, command
influence precedes raw contact, pre-contact authority reaches 1 mm, and the
nominal continuation contains a genuine protected contact.

Conditional on controllability, compare a five-dimensional soft/free
task-relative continuous oracle, a four-dimensional endpoint-preserving
control arm, and a fixed 54-candidate discretization. A candidate must have no
exact overlap or protected contact at any internal substep, pass CAR, stay
within 15 mm terminal EE deviation, and retain at least 50% nominal progress.
The Loewner proxy is reported but cannot veto a physical pass. Learning,
ranking, QP, primary corrected execution, live VLA and closed loop are frozen.

Clean H100 producer `39333` completed 702 cloned-OSC rollouts in `57:59` on
`worker-1`; independent H100 validator `39338` freshly replayed each arm's
best candidate and passed in `2:21`. The initial state was proxy-safe
(`+17.229 mm`), exact-overlap-free and contact-free. Command influence began
at internal sample 2, well before the first exact overlap at sample 132, and
paired probes moved L5--L7 centers by up to `14.244 mm` before violation. The
nominal future violation was genuine: 311 exact-overlap and 170 protected-
contact samples, with minimum proxy `-58.287 mm`. Thus timing and Cartesian
authority pass.

The candidate-support gate fails. All ten best soft/free finalists were raw-
contact-free, CAR-safe and progressing, but none met exact ellipsoid/box
nonoverlap or the 15 mm terminal bound; the best had 202 overlap samples and
`20.926 mm` terminal error. All ten endpoint-preserving finalists passed the
terminal/progress gates but retained contact and failed CAR. Exhaustive
internal checking found no contact-free or CAR-safe candidate among the fixed
54. The soft optimizer exhausted its registered eight generations, so this is
a bounded-search NO-GO rather than a proof of global nonexistence. Training
remains blocked; next change candidate horizon/continuation, not the model.

Result/validation payload SHA-256 values are
`3ac21494b469de9a7d60f702e9b6a19f12830ee73e2b2412f309c446496c7e98` and
`d4713cefffed96c1f8de67b8f46dbf1d335e594c20877faa90fad186f99dfa71`.

## E05 Moka frozen ranker audit (preregistered, 2026-08-13)

The next gate freezes the validated direction-conditioned scalar checkpoint
from job `39312` and changes only evaluation directions. At the identical
action-182 state and radius `0.0125`, generate 64 new directions using seed
`2026081410`, replay both signs through cloned OSC, and compare predicted with
exact twenty-action L5--L7 ordering.

Require at least `0.85` branch accuracy, exact one-sided binomial `p < 0.05`
against random sign, and positive mean exact gain on the six nominally near-
active L5 witnesses. Separately test Best-of-N ranking using predicted top
candidates at `N=4/8/16/32/64`, including exact percentile and regret. The
model, input, state, radius, geometry, horizon, and loss are immutable; no
training, QP, corrected execution, or closed loop is allowed.

Clean H100 producer `39319` completed on `worker-2` in `217.824 s` from
commit `5ee79369593c49180c0841c6a0201efc097b825a`; independent H100 validator
`39324` accepted all `129` rollouts, frozen-model/source bindings, new-
direction disjointness, primary gates, Best-of-N arithmetic, and hashes.

The frozen model selected the safer branch in `49/64` cases (`0.765625`),
which is strongly better than chance (one-sided binomial
`p=1.21823e-5`) but below the registered `0.85` gate. Near-active L5 accuracy
was identical. Mean selected near-active L5 gain was slightly negative at
`-0.002563 mm`, so the required positive-gain gate also failed; random sign
was worse at `-0.046616 mm`.

The model did select the exact best candidate at `N=16/32/64`, including the
best of all 128 signed physical branches at N=64. That best branch improved
only `0.093389 mm` and remained deeply unsafe at `-58.178324 mm`. Preserve
this as a one-state ranking diagnostic, not avoidance efficacy. The validated
interpretation is `frozen_paired_ranking_signal_fails_large_direction_gate`.
No training, QP, correction, or execution is authorized from this result.

Result/validation file SHA-256 values are
`c841ea3cbae7cc8cb02e286ec1dfa2fc94c85cb106d6d263fe956ac5a30e4f03`
and `caa06f768357493c97139c77325dce881bdfd9736dd1b2be336ccb1eb3e379d0`;
payload SHA-256 values are
`8b97733bba6491563cb1b92ad48b613704fd38e4f35ebffbe8ab5e49031333e9`
and `fb4c67621bd3c68d7f580865afd07a1b7378728dd5f778cc28eabf35cff94ba3`.

## E05 Moka local nonlinear prediction gate (preregistered, 2026-08-13)

The next gate retains the compact 25D physical witness input and validated
radius-`0.0125` action-182 protocol. It replays the exact 32 directions from
job `39305`, trains on directions 0--19, selects checkpoints on 20--23, and
keeps 24--31 untouched for the scientific test.

Two matched 40-input scalar MLPs isolate output representation. The first
predicts a central-secant response conditioned on the query direction. The
second predicts nonlinear absolute margins conditioned on signed action
deltas; its paired values imply a secant. Model width, depth, optimizer, seed,
state, geometry, and horizon are identical. No additional state input, QP,
corrected action, or closed-loop execution is allowed.

Clean H100 producer `39312` completed on `worker-2` in `130.628 s` from
commit `86d23589bf07f07d6bf4be372ca5f095b34325e5`; independent H100 validator
`39313` accepted all `65` cloned-OSC rollouts, the matched 21,889-parameter
architectures, exact split/direction binding, metrics, hashes, and gates.
Neither model passed.

On untouched directions, the direction-conditioned arm reached only
`0.067122/0.653571` all-witness cosine/sign and
`0.609441/0.833333` on six near-active witnesses. It nevertheless selected
the safer positive/negative branch in `7/8` cases and had `0.073 mm` worst-
margin RMSE. The latter is not meaningful absolute-value improvement: a
zero-response nominal-margin baseline is slightly better at `0.070 mm`.

The nonlinear local action-value arm was worse for steering: all-witness
cosine/sign `0.386933/0.586607`, near-active `0.063869/0.479167`, safer-
branch selection `3/8`, and worst-margin RMSE `1.224 mm`. Direct value
regression again fits common margin structure without recovering the small
action-dependent physical response.

The validated interpretation is
`neither_scalar_nor_nonlinear_local_representation_passes`. Retain only the
`7/8` paired-ordering result as a hypothesis for a future grouped-state
preference test. It is not a usable gradient, and QP/control remain blocked.
Result/validation file SHA-256 values are
`6d9d49bb7511be8ee325ebbcf94956724ccc7fd3c3cdab0319688d48d7bbbec4`
and `a691953a2f464738a8f1bc06a1c1fb5b21281ed78b2141a9fb9b229d450a7248`;
payload SHA-256 values are
`95937f109842d3adf80a38db2501b81b2b947d7e6a050004ec55a3a94761bef7`
and `19cab581d87761445807dd512967aeeabd41255ddb64cef6882e6d5e5317a29b`.

## E05 Moka compact-input secant-radius ablation (preregistered, 2026-08-13)

The next gate keeps the validated 25D one-state input and every model,
geometry, horizon, loss, optimizer, seed, and direction setting fixed. It
reuses the exact radius-0.05 direction vectors but executes paired cloned-OSC
rollouts at `0.025` and `0.0125`. This isolates physical secant locality from
direction support.

For each radius, fit the local ridge teacher and compact MLP independently.
Require both to predict the same eight fresh directions with cosine at least
`0.8` and sign accuracy at least `0.85`; also retain the compact fit and
near-active-row gates. No new state input, QP, correction, or closed-loop
experiment is allowed.

Clean H100 producer `39305` completed on `worker-2` in `222.103 s` from
commit `b91d81d6815855de06ad0e4027ee80e1275cc21e`; independent H100 validator
`39307` accepted all `129` cloned-OSC rollouts, immutable direction binding,
full-rank designs, hashes, and registered arithmetic gates. Neither radius
passed.

At radius `0.025`, compact-MLP held-out cosine/sign were
`0.383901/0.748214`; the local ridge ceiling was `0.646382/0.790179`. At
radius `0.0125`, the MLP improved to `0.708877/0.824107`, but ridge remained
only `0.664620/0.769643`. Both remained below the registered `0.8/0.85`
requirements. Within-radius near-active MLP/ridge agreement stayed high
(`0.963438/0.969130`), while cross-radius near-active ridge cosine was only
`0.764505`.

The strict interpretation is
`smaller_secants_do_not_rescue_affine_response_teacher`. The result does not
justify a larger MLP, additional state inputs, QP, or control. The one-state
twenty-action response itself is direction- and radius-sensitive; the next
prediction gate should replace the full-vector affine target with a
direction-conditioned scalar response or matched nonlinear local action
value.

Producer result file/payload SHA-256 values are
`cb7e2aeb65a031f7b665b4aad2d5fa2840f313e51fa4ac52e0752a321dca79a6`
and `d74bd14ea877dd9e9b0ae350909691bb23425160d898868b30e8860139aada8e`.
Validation file/payload SHA-256 values are
`0a89b39a331443fec92bd4a945373865f861c271d8802797e565b23c8b8222a2`
and `72e9392118ade961c937e50a742e74c63a693a4f3b7f5143671760f17f96e1f7`.

## E05 Moka compact-input memorization ablation (preregistered, 2026-08-13)

The next active gate changes exactly one factor from the validated `39230`
NO-GO: model input. It replays only original training action 182 and trains the
unchanged 128-unit value/response MLP from the nominal first-five XYZ chunk
plus time/link identity (`25` inputs rather than `1,055`). All optimizer, loss,
seed, direction, perturbation, geometry, and rollout settings remain fixed.

This is a capacity/memorization test. It compares the MLP with the attainable
per-state local ridge response on the same 24 fit and eight held-out paired
directions, and reports near-active rows separately. It cannot establish
state-conditioned learning because the physical context contains one state.
No additional state input, weighting change, QP, correction, or closed-loop
execution is allowed.

H100 attempt `39293` stopped before simulator construction or training. Its
allocation shell preflight reached `worker-0`, but that node's live
`/usr/bin/nvidia-smi` is a zero-byte file: the Python allocation-identity
helper failed with `Exec format error`. The retry excludes only this broken
node and changes no input, model, label, split, seed, loss, or gate.

Clean H100 producer `39295` completed the one-state ablation on `worker-2` in
2:10 from commit `f3f7188956e3129287deba90e465345926b48659`;
independent H100 validator `39298` accepted all 65 replay rollouts and every
registered arithmetic gate. The strict result is NO-GO, but it isolates two
causes rather than one.

The compact 25D model improved step-182 fit response cosine from `0.490390`
to `0.880229`, sign accuracy from `0.759226` to `0.932143`, and value RMSE
from `32.750 mm` to `0.701 mm`. Its six near-active L5 rows reached mean
cosine `0.931639` with the local ridge rows. This confirms that the old 1,055D
redundant input materially caused training underfit.

The prediction gate still fails. Compact held-out response cosine was only
`0.408501`; the full-rank local ridge teacher was also only `0.449661`, with
condition number `480.847`. The compact model's fit cosine and all-row ridge
cosine were `0.880229/0.862107`, below the registered `0.9` thresholds. Thus
input reduction alone cannot establish a usable response gradient. The next
one-factor gate should keep compact inputs and test smaller paired secant
radii before adding state variables, QP, or control.

Producer result file/payload SHA-256 values are
`a55fd62fe391724c662296ae7e59f67328f163c3980486b37465fdec7955aab1`
and `5384c6893e0fefaa6e19bd59c8e13c210a1fe25e04a9e18eb00468325281ecae`.
Validation file/payload SHA-256 values are
`f5ca20f14e70f16b5978067e95382eb1e9edf3424b39d78f5cbfab532656db39`
and `6c2a4c50561026d0c72276892653c948a3ed001949eccd402baf901d15609492`.

## Frozen E05 Moka response-model audit (preregistered, 2026-08-13)

The next gate is read-only diagnosis of the validated `39230` NO-GO. It
replays the identical nine state groups and 32 paired directions per state,
reconstructs the exact frozen model from its immutable artifact, and forbids
training, parameter changes, larger corrections, QPs, corrected execution,
and closed loop.

The audit reports response accuracy separately for actions `178--182` train,
`183--184` validation, and `185--186` test, including directions excluded
from each state's local ridge fit. It decomposes state support into simulator,
auxiliary, clock, controller, nominal-action, and obstacle-geometry groups.
For all 140 action-boundary witnesses per state it also records link, time,
closest compiled Moka primitive, closest/second-closest primitive separation,
value error, response-row cosine, and near-active membership within 2/5 mm of
the state minimum. Raw protected contacts and exact box-overlap counts remain
separate physical diagnostics.

This can distinguish training underfit, state-coverage failure, and
near-active witness representation/weighting failure. It cannot authorize a
new model or control experiment; the diagnosed intervention requires a new
preregistered gate.

Submission was first rejected before allocation because the live cluster no
longer exposes the historical `studentbatch` partition. Live preflight found
the default `main` partition with H100 resources and `normal` QOS. The
launcher-only repair targets that current allocation interface; no audit
input, frozen model, replay, metric, or interpretation changed.

H100 attempt `39244` stopped before simulator construction because the
isolated clean worktree was not exported on `PYTHONPATH`. The launcher now
uses the same repository/SafeLIBERO module and LIBERO configuration contract
as the accepted producer. This is an apparatus-only fix; no scientific result
was produced.

Clean H100 producer `39265` completed the final frozen-model audit on
`worker-1` in 16:44 from commit
`f6355f490c5b5780e1bdc6ed9dcc0edeec85a993`; independent H100 validator
`39272` accepted all 585 replay rollouts and the immutable model hash. The
primary diagnosis is `training_response_underfit`: MLP held-out response
cosine/sign were only `0.312181/0.572679` on train,
`0.340293/0.627232` on validation, and `0.098247/0.598661` on test. Value
RMSE was `34.292/33.266/36.290 mm`; the train value head had 21 false-safes.

State coverage is secondary. The 1,055D input has only five distinct training
states, 875 dimensions constant across train, and duplicated controller
state. Test action chunks are `3.617/4.346` nearest-train RMS z, with
31.4--33.6% of coordinates outside range; action 186 auxiliary state is
`19.198` nearest-train RMS z. Obstacle geometry is constant and supported.

Near-active switching is rejected as causal. All 54 rows within 2 mm of the
worst value are the same L5 slab/Moka `g4` pair over actions 188--193; paired
primitive-switch fraction is zero and the runner-up remains 5.15--6.65 mm
away. The MLP overestimates these rows by 48.025 mm on average with response
cosine 0.509. They form only 4.29% of train witnesses. The local ridge teacher
has useful sign `0.877/0.808/0.779` but cosine only `0.663/0.427/0.379`, so
finite-radius long-horizon nonlinearity is an additional issue.

Do not add a QP or larger correction. The next gate should require one-state
and aggregate train fit using compact nonduplicated physical inputs and
scale-normalized near-active supervision, then separately test secant-radius
or direction-conditioned response fidelity.

Final result/validation file SHA-256 values are
`19b754da4c26a9950038463ca1f49e5543f42042ea259764885a84e269450e27` and
`803a7e09252a114bedfdc9b5a3a8b03d2cacb7db3034dff85f2f86eef932df50`;
payload SHA-256 values are
`5e0e80b6163c90e6c28a8b76ef61dd8dcfb112f95c9439322d5d62aff78e178d` and
`ca73b28abcc82b87c744100cc8a53d730bb795d3508d9850b904af24f3fe59c7`.
Earlier `39247/39261` is superseded because it lacked the final teacher and
primitive-switch checks.

## E05 Moka learned response-field gate (preregistered, 2026-08-13)

The next learning pilot deliberately excludes E38 because its released single
obstacle MVEE remains negative after physical contact has disappeared. It
returns to the task-valid primary Moka case `vlsa-t1-goal-ii-t0-e05`, keeps
the frozen VLA, OSC, archived Table-1 actions, action bounds, and accepted
seven L5--L7 robot ellipsoids unchanged, and replaces only the obstacle
optimization proxy with all 15 compiled Moka collision boxes. Raw MuJoCo
protected contact and exact box-union overlap remain separate physical
diagnostics.

This does not repeat the rejected binary classifier, direct joint predictor,
or scalar monotone row-weight MLP. For each future action and robot row, one
state/witness-conditioned MLP predicts the quantitative multi-primitive margin
and its 15D response to the first five XYZ actions. Paired `+0.05/-0.05`
twenty-action cloned-OSC rollouts supervise `v^T g_j`. Complete state groups
are frozen as actions 178--182 train, 183--184 validate, and 185--186 untouched
test. The later action-197 collision is inside every outcome horizon.

At each test state, learned steering is compared at equal norm with fixed
multi-primitive repulsion, a privileged exact local-secant ceiling, and 64
matched random directions. The gate requires direction cosine at least 0.8,
held-out sign accuracy at least 0.75, at least 0.5 mm exact proxy gain, at
least 0.1 mm gain over fixed repulsion, matched-random `p<=0.05`, and no new
raw protected contacts at both test states. No corrected action executes, and
no QP, closed-loop, task-completion, population-generalization, deployable
perception, CBF, or formal-safety claim is authorized by this pilot.

H100 attempt `39229` was stopped during startup before any scientific output.
The pinned evaluation PyTorch reports CUDA architectures only through sm_86
and cannot launch MLP kernels on H100 sm_90. The MLP has only 128 hidden units,
so the compatibility retry trains it with eight CPU threads inside the same
H100 allocation; simulator execution, data collection, and training remain
off the login node. No input, label, split, seed, loss, comparator, threshold,
or model parameter changes.

Clean H100 producer `39230` completed the frozen retry on `worker-1` in
24:44 from commit `a505434ea7aed7e55f596e0b61d389e5ac593ed1`; independent
H100 validator `39238` accepted the immutable artifact. The producer ran 987
paired/evaluation cloned-OSC rollouts. The learned-versus-exact direction
cosine was only `0.676880/0.313475` at untouched actions `185/186`, and
held-out directional sign accuracy was `0.679464/0.517857`; both states miss
the registered `0.8/0.75` gates.

The learned field improved the conservative multi-primitive minimum at every
tested radius, but action 185 was consistently worse than fixed repulsion.
At action 186 it beat fixed repulsion by `0.223611/0.428832 mm` only at radii
`0.25/0.5`. Only the action-186 radius-0.25 row passed its per-radius checks;
the complete state gate did not. No candidate removed the approximately
`58.272 mm` conservative violation or its protected contacts. Therefore the
validated interpretation is
`one_task_moka_controller_conditioned_response_strict_no_go`, test-state pass
count `0/2`. No learned correction executed, and QP/closed-loop work remains
blocked pending a split/support and near-active-witness response audit.

Result/validation file SHA-256 values are
`1a8f78a486812fd2cc22fe5e75a40198d13c074be1024abfcae160c36e09ed93` and
`f7858a73123eb1f9215c75d698ec5172965113bdc228552bbba86e7e762c99e8`;
payload SHA-256 values are
`3df335aaa06ff54b99b24beabb04cbcf5c693c3349cad7f9bf54f25dacafbde2` and
`29b97422329786177bf2cc763801d5fad1803cc36fe6d4bad9f242683a8ff530`.

## Post-detour live-continuation gate (preregistered, 2026-08-12)

The next experiment is intentionally limited to the missing continuation
question exposed by Gate 0. Starting from the identical pre-action-182 state,
it internally verifies and executes only the registered compound actions
182--186. At the resulting measured action-187 state, one fresh frozen
`pi0.5` query uses registered query index 37 and its immutable noise seed.
The first five raw actions are converted by the unchanged released AEGIS EE
QP along their nominal cloned-OSC path, then frozen as one paired continuation
chunk.

Three local arms are compared from the identical action-187 state: the
registered compound actions 187--191 as a binding positive control; the fresh
`pi0.5` plus released-AEGIS chunk; and, only if that live chunk is unsafe, the
same chunk after the registered iterative 2 mm smooth all-witness secant
field. Every arm has exactly 126 measurements: the initial state and every one
of 25 MuJoCo model steps inside each of five OSC actions. Acceptance remains
at least 1 mm L5--L7 ellipsoid clearance, zero raw protected contact, and
paper CAR at most 1 mm. Field acceptance additionally requires every fitted
iteration to pass held-out cosine 0.8 and sign accuracy 0.75.

This is a local mechanism gate, not a complete receding episode. A safe fresh
VLA arm shows that live feedback can continue the detour locally; a safe field
arm shows that repeated filtering can repair the first new continuation. If
both fail while the registered continuation passes, the current five-action
live proposal/field family is a strict local NO-GO. Only a pass authorizes a
complete receding episode. No MLP, QP, CBF, task-completion, or formal-safety
claim is allowed at this stage.

H100 attempt `38989` passed allocation and unit preflights, internally verified
the registered prefix, and then stopped before the policy query on an
over-strict bitwise comparison between the rendered primary environment and
the image-disabled instrumented probe's complete auxiliary/controller state.
It produced no scientific result. The retry retains that numerical difference
as apparatus evidence while preserving exact pre-prefix synchronization,
bitwise action identity, and all-substep clearance/contact/CAR authority. No
policy input, action, field setting, or acceptance threshold changes.

Clean H100 producer `38990` completed from commit
`6b04a2c5c6deee09ce818d77da52cfe6c548677c` on `worker-1` in 2:45;
independent H100 validator `38993` recomputed every gate in one second. The
registered compound prefix and continuation passed at `+15.441969 mm` and
`+8.151976 mm`, respectively, with zero protected contact and paper CAR. This
confirms that the post-detour state has safe five-action continuation support.

The fresh query-37 `pi0.5` plus released-AEGIS continuation was unsafe at
`-12.433323 mm`, with 35 protected-contact samples, although CAR remained
within threshold. Ten smooth-field iterations used 851 cloned-OSC rollouts,
consumed the full registered path budget `1.0`, and produced correction L2
`0.905002`. Every independent direction audit passed (minimum cosine
`0.975932`, minimum sign accuracy `0.875`). The exact final rollout eliminated
all raw protected contacts and passed CAR, improving the margin by
`12.186052 mm`, but its hard minimum remained `-0.247271 mm`. It therefore
missed zero clearance by `0.247271 mm` and the registered +1 mm buffer by
`1.247271 mm`.

The focused gate is a strict local NO-GO. Live feedback alone did not preserve
the safe detour, and the current single smooth descent nearly removed contact
but could not reach the clearance gate within its fixed budget. Do not run a
complete receding episode or train an MLP from this field. The important next
question is no longer whether the local safety direction is accurate--it is--
but whether the fresh live chunk needs a larger/nonlocal or multimodal
continuation proposal. Any next oracle should compare the registered safe
continuation with derivative-free or structured proposals at this exact
action-187 state, rather than adding more local field fitting.

Producer result payload/file SHA-256 values are
`318f3e64701db6c5afd893b1b7c733c395fbd5e6812d7dfa3e84b66739a73bf1` and
`ada4c854bc650bb2ef7fa3ed356b11acde2f9069b5d3f2d60370ad1957dd4976`.
Validation payload/file SHA-256 values are
`2d8cf0c050435e27dbddcd3dd45adfaa49575b3a24b311ecec02b2abc0187474` and
`1dae3663491a3b76684fdfd120d6bf00a33a6b9b34314e13add92cdae2571f7e`.

## Action-187 persistent-route oracle (validated, 2026-08-12)

Clean H100 producer `39029` and independent H100 validator `39032` tested the
same measured action-187 state and frozen fresh pi0.5/released-AEGIS chunk from
the post-detour gate. The nominal reproduced `-12.433323 mm`; the registered
continuation reproduced `+8.151976 mm`. Four geometry-defined persistent route
modes (left, right, projected-up, and obstacle-normal retreat) were tested at
correction L2 bounds `0.5/1.0/1.5/2.0`, followed by one smooth-field refinement
and a matched full 15-D derivative-free feasibility control. All acceptance
used every internal MuJoCo substep, seven L5--L7 rows, a +1 mm buffer, zero
protected contacts, and paper CAR.

Persistent retreat was safe at norms `1.0/1.5/2.0`, with margins
`+2.071672/+8.650534/+9.400398 mm`. Left and projected-up were harmful; right
improved the margin but remained unsafe. Best-route plus smooth refinement
reached `+4.314759 mm`; the derivative-free control reached `+10.822065 mm`.
Therefore a generic physical route family can enter a verified safe corridor
at action 187, and route selection matters more than simply increasing a fixed
repulsive force. This authorizes a complete receding E05 oracle, but not MLP
training, policy-value claims, QP claims, or neural-CBF terminology.

Producer and validation payload SHA-256 values are
`52efbd3c18331a4c200bc23baa8c6b257220ff4e2307e07e6ac821f10804a19b` and
`b04f65a722fd07a620f46ad019935d3d673902da3835fc3e7c348c135f7915b0`.

## Raw-AEGIS multi-start Gate 0 (preregistered, 2026-08-12)

Before launching a radius or branch search, Gate 0 tests whether the registered
raw fixed-suffix problem actually contains the previously cited positive
control. From the identical pre-action-182 state it compares: immutable raw
AEGIS actions 182--201; the full earlier-detour-plus-smooth compound trajectory;
and the compound actions 182--186 transplanted onto immutable raw AEGIS actions
187--201. The transplant is defined by physical normalized-action displacement
using scale only, never by subtracting a normalization mean. Requested and
applied corrections, coordinate clipping, and bitwise prefix/suffix hashes are
recorded.

All three arms are measured at the initial state and after every one of 25
MuJoCo model steps inside every OSC action. Acceptance requires at least 1 mm
ellipsoid clearance, zero raw protected contact, and paper CAR at most 1 mm.
The full compound may reveal either binding error or an action-boundary versus
internal-substep transient if it fails this stricter authority.

The radius/multi-start search is conditional. It is authorized only if raw
AEGIS is unsafe, the full compound is safe, the transplanted prefix with raw
suffix is also safe, no clipping occurred, and both action identities match
bitwise. If only the full compound is safe, Gate 0 stops the experiment and
classifies the five-action prefix as insufficient without an adaptive or
modified continuation. No MLP, live VLA, QP, CBF, task-completion, or formal
safety claim is allowed.

H100 attempt `38977` passed its allocation and unit-test preflight, then stopped
before any rollout because the transplant reconstructed the registered prefix
as `raw + (compound - raw)`. The subtraction/addition round trip was
numerically equivalent but failed the required bitwise identity assertion.
This is an apparatus failure with no scientific outcome. The retry assigns the
registered compound prefix directly and continues to record its displacement
from raw plus any clipping; no action value, comparator, or gate changes.

Clean H100 producer `38978` completed from commit
`79a33fcb7427999f0a2675229c4059266d72b914` on `worker-1`; independent H100
validator `38979` reproduced every registered gate. All arms contained exactly
501 measurements: the initial state plus 25 MuJoCo model steps for each of 20
OSC actions. Raw AEGIS reproduced `-16.075686 mm`, 193 protected-contact
samples, and `22.833455 mm` CAR displacement. The full compound trajectory
passed strictly at `+1.673938 mm`, zero protected contact, and effectively zero
CAR displacement.

The identical compound actions 182--186 transplanted onto the bitwise raw
AEGIS suffix failed at `-6.789842 mm`, 85 protected-contact samples, and
`8.289781 mm` CAR displacement. Requested and applied correction L2 were both
`1.849546`; no coordinate clipped, and both prefix and suffix hashes matched.
The prefix delayed first L5 contact from action 187 to action 191 but could not
prevent actions 191--194 from returning L5 into the obstacle. A second L5 slab
also reached `-1.154244 mm` while all L6/L7 slab minima stayed positive.

Gate 0 therefore blocks the radius/multi-start search. The raw fixed-suffix
problem does not contain the previously cited positive control, even with its
full `1.849546` correction. This result rules out a simple “previous radius too
small” diagnosis for the registered five-action/raw-continuation family. The
next oracle must permit actions after 186 to adapt: correct a longer horizon or
execute a short verified prefix and requery the frozen VLA. Do not train an MLP
or use the compound result to claim a directly recoverable raw five-action
detour.

Producer result/file SHA-256 values are
`eb673729c4c95b33d5551137c88e7e089b9eab4efc584dacbb996039e0c78b65` and
`b7f7410b307e0bedef225d710b6dcd4ed06be13ff23fa3a0988f9705d7bf193f`.
Validation payload/file SHA-256 values are
`5c0cf046fc0d88aff2422e62a3b42b127333651c8acf62674d20b2b8e4af29ca` and
`a2750f7983dad7bbea6c205247f08677d7f8f75d27b2071fcec198b03e7a5853`.

## Direct smooth-field attribution gate (preregistered, 2026-08-12)

The next gate separates the successful smooth proposal from the earlier
five-action task-rejoining detour. Starting from the identical archived
pre-action-182 state, it compares the immutable raw Table-1 AEGIS suffix, the
earlier detour, the detour plus the validated smooth second-stage correction,
and a freshly fitted smooth counterfactual secant field applied directly to the
raw AEGIS suffix. All comparisons use actions 182--201 and modify only XYZ in
the first five actions. The direct field retains the registered 32 paired
`+0.05/-0.05` probes, `2 mm` smooth-min temperature, `0.1` trust steps, exact
hard-margin line search, and radius/path budgets of `1.0`. Eight independent
paired directions per iteration must achieve cosine at least `0.8` and sign
accuracy at least `0.75`; these held-out probes do not fit the field.

The field continues to be fitted from the seven action-boundary ellipsoid rows
over twenty actions, avoiding an unregistered 3,500-row change in the proposal
model. Every comparator and every candidate considered for final scale
selection is then replayed with instrumentation after all `25` MuJoCo model
steps inside each 20 Hz OSC action, including the initial state. The attribution
gate requires at least `1 mm` internal-substep ellipsoid clearance, zero raw
L5--L7 contact, and paper CAR. A `0.025` grid reports the smallest verified
scale along the discovered correction ray only; it is explicitly not a global
minimum-norm claim.

This is a single-state attribution experiment. It contains no live VLA query,
closed-loop execution, task-completion claim, MLP, QP, CBF, policy iteration,
or formal safety result. Passing proves only that the smooth field can directly
repair raw AEGIS under the registered continuation and held-out directional
gate. Only then may a separate receding frozen-VLA recovery experiment begin.

Clean H100 producer job `38974` completed on `worker-1` from commit
`7d617fa801cad4684a386e4dd92c34dca04e516d` in `7:36`; independent H100
validator job `38976` accepted the immutable result. Internal instrumentation
recorded the initial state plus exactly `25` MuJoCo model steps for every one
of 20 OSC actions. Raw AEGIS reproduced a minimum of `-16.075686 mm`, `193`
protected contact samples, and `22.833455 mm` paper-CAR displacement. The
earlier five-action detour alone also remained unsafe at `-14.305567 mm` with
`41` contact samples and `1.070811 mm` CAR displacement.

The direct smooth field improved exact internal-substep clearance by
`9.066542 mm`, reaching `-7.009143 mm` after seven accepted steps with path
length `0.7` and net correction L2 `0.668480`. It nevertheless retained `125`
protected contact samples, missed CAR at `18.110775 mm`, and had no safe scale
on the registered `0.025` ray grid. Held-out secant prediction passed strongly
for the first seven fits (`0.994--0.999` cosine), but the eighth local fit fell
to cosine `0.694629`, below the registered `0.8` gate; exact line search then
found no improving step. The direct-attribution gate is therefore a strict
NO-GO.

The compound detour-plus-smooth comparator reproduced `+1.673938 mm`, zero
protected contact, and paper CAR. Its smallest registered second-stage scale
was `0.975`, with `+1.173397 mm` clearance, and its total correction from raw
AEGIS had L2 `1.849546`. This establishes that the earlier positive smooth arm
was a second-stage repair of a much larger detour, not a standalone correction
found directly from raw AEGIS. Do not start live closed-loop recovery or train
a field MLP from this result. The next mechanism question is whether a richer
proposal family can recover the detour mode from raw AEGIS; merely smoothing
the scalar risk or learning its local secants is insufficient.

Producer result/file SHA-256 values are
`54469ee5dcd5ac8263df5a0cbaf833fb028d17ef8f0587ee501d0b4959865f55` and
`49799eb0fb2ce45e916d3f6f7b630a4ed27ccf2c41fcee364d03ddf02f995f9e`.
Validation payload/file SHA-256 values are
`b6a90330d948c76516d29075c902430b322e1838d3c8e3faadd20cbf3032b5dd` and
`b23fc9c3b4d29b4071787b44e35bbf61c115a48f63a8590087bbec2b2fee403c`.

## Multi-witness counterfactual field gate (preregistered, 2026-08-12)

The next avoidance-only test keeps the same immutable E05 action-182 state,
five corrected XYZ actions, and twenty-action outcome through the known
action-197 collision. It addresses the fixed-step result's identified failure:
the hard worst link/time witness switched on every paired probe, so one
collapsed gradient zigzagged despite known radius-1 safe support.

Every arm now estimates separate counterfactual action sensitivities for all
140 `(future action offset, L5--L7 slab row)` clearances from 32 paired
`+0.05/-0.05` cloned-OSC rollouts at each visited center. The comparison is:

1. the single currently worst clearance row;
2. a `2 mm`-temperature smooth minimum across all rows;
3. a coordinated epigraph solve over at most eight rows within `5 mm` of the
   worst witness; and
4. the immutable radius-1 derivative-free safe result from job `38955`.

The coordinated arm minimizes the maximum linearized violation plus a small
quadratic action penalty under a `0.1` local trust radius, normalized action
bounds, total correction L2 at most `1.0`, and total accepted path length at
most `1.0`. Every arm uses the same three exact line-search fractions, accepts
only an exact twenty-action hard-margin improvement, and recomputes at every
accepted center. Task penalties, endpoint preservation, attraction, MLP, flow
guidance, and closed-loop execution remain excluded.

The multi-witness gate requires positive exact L5--L7 clearance, zero raw
protected contact, paper CAR, the shared path budget, and a better hard margin
than the single-witness arm. Failure while the derivative-free control remains
safe rejects local multi-witness linearization for this detour and points to
nonlocal/multimodal candidate planning. Passing authorizes only later learning
of separate link/time-conditioned rows; it does not authorize an MLP that
outputs one collapsed vector.

H100 attempt `38964` passed all allocation tests and completed substantial
paired rollout work, then stopped without a result when one visited action lay
within `0.05` of a normalized coordinate bound. The legacy random-direction
sampler kept rejecting symmetric pairs until its fixed attempt limit. This is
an apparatus failure with no scientific outcome. The retry freezes only those
coordinates lacking the preregistered `+0.05/-0.05` bidirectional headroom and
samples the same count, radius, seeds, and smooth random basis in the largest
remaining coordinate subspace. No optimizer, witness, budget, or gate changes.

Clean H100 producer `38966` completed `1,814` cloned-OSC rollouts in `778.612`
seconds from commit `325e285`; independent H100 validator `38967` reproduced
the immutable result. The nominal hard margin was `-14.263205 mm`. The current
hard-min row reached only `-7.944801 mm`. The top-eight near-active epigraph
improved slightly further to `-6.678093 mm`, but retained three L5 contacts,
missed paper CAR at `1.196774 mm`, and exhausted `0.95` of the shared path
budget. The preregistered multi-witness gate is therefore a strict NO-GO even
though the immutable derivative-free control remains safe at `+7.892343 mm`.

The matched `2 mm` smooth-maximum-of-risk arm produced a distinct positive
mechanism result: after seven `0.1` updates it reached `+1.673938 mm` exact
clearance with correction L2 `0.622085`, zero protected contact, and effectively
zero obstacle displacement (`0.000000023 mm`). Its terminal EEF deviation was
`19.576451 mm`, diagnostic only in this avoidance experiment. This shows that
continuous weighting of all future link/time witnesses can avoid the archived
collision where hard row selection still switches or omits useful impending
witnesses. It is one-state oracle evidence, not learned steering, task recovery,
population generalization, or a formal safety result. Do not train the proposed
hard top-M row model from this failed gate. The next independently registered
test should either validate the smooth field across additional dangerous states
or execute this verified five-action correction and test live frozen-VLA
replanning; no task-completion claim is made here.

Producer result file/payload SHA-256 values are
`639b4928efdb5bb27b2dcd4ea5d1c224f5fd6116264d29d316940933a7c96f58` and
`eb7bc1d97c6821b9fc66b15380383eb56917712d45af25613a7c1ea4ef3f0d77`;
validation file/payload SHA-256 values are
`a9e80073eade8cbbf3e151f112eae129efaefe9452f1e4bb9e3be03b5573df2a` and
`e12329c67be9fb5ff35da220db416adf71ee0bdac9b3e1da0a4e172bc2c0af41`.

## Fixed-step long-horizon avoidance field gate (preregistered, 2026-08-12)

The active E05 mechanism test now isolates whether the paired-rollout action
field itself can find the safe five-action detour already demonstrated by the
relaxed analytical and derivative-free oracles. It replays the same immutable
state at action 182 and evaluates every corrected five-action prefix through
the same fixed continuation to action 201, including the known action-197 L5
contact. The risk target is only `V_H=-min h` over seven L5--L7 slab rows and
twenty actions. Task penalties, endpoint preservation, attraction, QP, MLP,
flow guidance, and closed-loop execution are excluded.

At every iteration, 32 paired `+0.05/-0.05` cloned-OSC rollouts estimate a new
15-dimensional XYZ clearance direction. The method must take one full
normalized `0.1` action-space step without line search or backtracking, then
re-estimate at the new center. It stops only after exact positive clearance
with zero protected contact and paper-CAR pass, after ten registered steps, or
when a full step violates physical action bounds. A matched comparator follows
the first estimated direction under the identical step schedule. Previously
validated radius-1 analytical and derivative-free safe candidates supply the
shared-budget feasibility ceiling without new simulator search.

The gate passes only if recomputation reaches verified safety within the same
radius-1 budget and improves on the fixed initial direction. If known safe
baselines remain positive but this field stays unsafe, the single-gradient
representation is rejected and no field MLP is authorized. A pass authorizes
only the next live-VLA rejoining test; it is not task-completion, learned,
population, or formal-safety evidence.

Clean H100 producer `38961` completed 656 deterministic cloned-OSC rollouts on
`worker-2` in 5:22 from commit `5a81ecd`; independent H100 validator `38962`
recomputed the immutable receipt. The nominal hard margin reproduced
`-14.263205 mm` with L5 contact at action 197. Recomputing the long-horizon
field after every full `0.1` step improved the best margin to `-7.125095 mm`,
substantially better than following the initial field direction
(`-12.861372 mm`), but remained unsafe after all ten registered steps. The
best recomputed action still contacted L5 at actions 196 and 197 and exceeded
paper CAR at `1.043142 mm`.

All 32 paired branches switched active witness in every one of the ten field
fits, alternating the worst second-L5-slab witness between continuation
offsets 14 and 19. The recomputed directions therefore zigzagged: ten full
steps consumed a registered path length of `1.0` but produced net correction
L2 only `0.749880`. This is not empty support. The same radius-1 evidence
contains verified analytical and derivative-free corrections with
`+16.023852/+7.892343 mm` margins, zero protected contact, and CAR pass.

The avoidance-only gate is a strict NO-GO for a single hard-min gradient field.
Repeated recomputation is useful but insufficient because the nonsmooth worst
future witness switches across modes. Do not train its MLP and do not run live
VLA recovery from this candidate. The smallest justified next representation
is multi-mode: retain separate active link/time field directions and choose or
optimize among them, rather than average them into one local vector.

Producer result file/payload SHA-256 values are
`3b68cb6eaa0ba54144d3306b712c3a5a04d3cb27571b422ade04b8a7ead40be9` and
`8d14d053bb85dcff43c55c55a5f71637ff85a558b5237f10f3de8d2dc7e5c0fb`;
validation file/payload SHA-256 values are
`3d143616f4c1a767cf5bb0181b8b7fcf198678d855f877367391ee6cdc82ea65` and
`48b2c351e58ef245d43ba3158cdd12ef74df3e18da5126436fa000f6990312d5`.

## Receding five-action exact-oracle gate (preregistered, 2026-08-12)

The active E05 gate now repeats the validated five-action task-rejoining
optimization across the episode. It replays immutable actions 0--181, uses the
immutable 182--186 window for its first decision, and thereafter requests a
fresh frozen pi0.5 Cartesian chunk from every measured state. Each window is
evaluated with the exact cloned OSC and seven accepted L5--L7 slab clearances;
an unsafe nominal window triggers the same five-iteration finite-difference
SQP used by job `38874`. Only the first action of a freshly verified five-action
window may execute, after which the horizon shifts by one action. The success
gate requires native task completion, zero robot/L5--L7 contact, paper CAR,
and no infeasible window. This is an exact simulator oracle, not a deployable
learned controller or a formal safety claim. MLP training remains blocked.

Clean H100 producer `38903` and independent validator `38907` completed this
gate. At action 182 the receding oracle reproduced the `-9.313104 mm` nominal
window and selected the validated `+1.747199 mm` detour, executing only its
first action. Fresh nominal windows at actions 183 and 184 were exactly safe.
At action 185, however, the new nominal five-action window reached
`-7.611232 mm`; all five SQP iterations remained below the registered `1 mm`
buffer. The best exact candidate reached only `+0.369923 mm`, so the controller
failed closed before contact, CAR, or task completion. This is a strict oracle
NO-GO: receding evaluation detects the later danger, but the present five-action
zero-sum candidate family has no accepted support. Validation SHA-256:
`014d2deb0c1fffe3cad4930e3b32379d3fbdd4d5a7fc87c8eddb489cdac262ea`.
Do not train the direct correction field. The next gate must intervene before
action 182 or expand the task-rejoining horizon/candidate family.

## Early five-action detour apparatus retry (2026-08-12)

H100 attempt `38873` passed allocation tests, replayed the immutable prefix to
the pre-action-182 state, and stopped before candidate generation because the
new evaluator hard-coded a nonexistent `robot0_grip_site`; this SafeLIBERO
model names the authoritative site `gripper0_grip_site`. The attempt is retained
as apparatus failure. The compatibility retry uses the repository's existing
`_eef_site_id` resolver and changes no scientific setting.

Clean H100 job `38874` found and executed a verified endpoint-preserving
five-action detour: the archived minimum improved from `-9.313104 mm` to
`+1.747199 mm`, XYZ residuals summed exactly to zero, terminal EEF error was
`11.669 mm`, and nominal progress ratio was `0.8966`. The frozen policy later
completed the task at action 283, showing that early task rejoining repaired
the competence failure of the late emergency correction. However, unfiltered
post-detour AEGIS contacted L5 at action 197 and failed paper CAR at 199. The
detour is therefore promising but not sufficient; the next matched composition
keeps the same detour and adds exact receding cloned-OSC filtering only after
the detour.

H100 composition job `38876` replayed the identical prefix and identical
five-action detour, then enabled the existing exact one-step cloned-OSC filter
at action 187. The filter evaluated 87 candidates but found no exactly verified
safe action, so it failed closed before contact and before task completion.
Independent H100 validator `38881` reproduced both outcomes, decoded all
`285` and `188` video frames, and accepted the paired interpretation: the early
detour preserves task competence but is not persistently safe, whereas the
one-step filter has no safe support immediately after it. Validation SHA-256:
`e726da999a6e166f1826ac287f3f01999487c584d675245a699df834ba6adf74`.
The primary problem remains unsolved. The next gate is a receding five-action
task-rejoining detour, not learning or stronger one-step repulsion.

## Field-recovery executable gate apparatus retry (2026-08-12)

H100 array `38848` was canceled after exact inspection when the first completed
arm exposed an apparatus error: the generic one-step SITL activation modified
archived action 184, so the registered action-185 field oracle never ran. Its
partial/result artifacts are retained as non-scientific apparatus evidence.
The retry passes released AEGIS actions 0--184 byte-for-byte and activates only
at the archived pre-action-185 state.

Corrected H100 array `38854` and independent validator `38867` completed the
four-arm executable gate. All arms reproduced the dangerous nominal margin
`-9.313104 mm`, produced a fresh positive-clearance two-action repair, executed
only its first action, and used live frozen pi0.5 plus released AEGIS replanning.
Fixed, normal-plus-tangent, and unrestricted ran 300 actions; normal-only failed
closed at action 209. All four had zero active-obstacle robot contact, zero
protected L5--L7 contact, and paper-CAR pass, but none completed the native
task. Thus late post-hoc geometry can prevent this collision, but the required
large correction destroys task recovery. Normal-plus-tangent did not beat the
fixed baseline, so the field-mixing MLP gate is closed. Validation SHA-256:
`6c1011d2a74d6087cad2840e065ee40e7f556f91748041ef50a3c0fe8b27b38c`.

## Barrier-free EmbodiSteer baseline fidelity pilot (preregistered, 2026-08-08)

Before reconsidering any L5/L6 geometry or collision guidance, the active
`E02` work now isolates the paper's first two baseline groups on the primary
case: ordinary Cartesian denoising without guidance (`EE`) and joint-space
denoising without guidance (`Joint`). Both arms use the same frozen
`pi05_libero` checkpoint, the same settled simulator state, observation,
policy-noise seed schedule, five-action execution horizon, physical obstacle
scene, and 300-action limit. Every ellipsoid, barrier row, clearance query,
and QP is disabled in both arms.

The joint arm follows EmbodiSteer Eqs. (3), (4), and (8)--(10): Cartesian
Gaussian initialization is lifted around the chunk-start Panda configuration;
each reverse pi0.5 Euler step maps the joint trajectory through exact MuJoCo
FK, queries the unchanged Cartesian denoiser, and maps the resulting pose
residual back with the damped Panda Jacobian (`alpha=0.1`,
`lambda_pinv=0.001`, joint clip `0.5 rad`). The resulting joint targets are
executed directly with SafeLIBERO `JOINT_POSITION`; the gripper channel is
retained from the same denoised chunk.

This is a paper-derived adaptation, not an exact author-code reproduction.
The paper uses 10D DDPM actions expressed as poses relative to the chunk-start
pose; `pi05_libero` uses 7D incremental OSC flow actions. We therefore compose
incremental deltas into chunk-start targets for the paper equations, then map
back to incremental deltas whenever the frozen pi0.5 denoiser is queried. The
initial live gate requested `5e-5` raw-unit equivalence; allocation-only
calibration below replaces that inapplicable fused-versus-split JIT threshold
with preregistered physical pose and gripper-sign bounds before simulation.

The competence gate is deliberately prior to safety efficacy: if the joint
arm does not preserve useful task behavior relative to the Cartesian arm,
collision guidance cannot be credited. Separately, the current L5/L6 MVEE
representation is no longer treated as EmbodiSteer-faithful geometry. The
paper uses multiple link-attached cuRobo collision spheres with a top-4
smooth maximum, not one ellipsoid per link; the old ellipsoids remain disabled
until a raw-geometry audit is complete.

Initial H100 submission `37049` stopped during its allocation-side unit gate,
before policy startup or simulation, because Python 3.8 eagerly evaluated one
new `tuple[...]` type-alias expression. The immutable run root contains an
apparatus-failure receipt and no result or video. The repair changes only that
annotation to `typing.Tuple`; no experiment parameter or algorithm changes.

Retry `37050` passed the allocation unit gate and loaded `pi05_libero`, but the
live ordinary-sampler equivalence check rejected the chained one-step endpoint
before either simulation arm. This second immutable attempt is also an
apparatus failure with no scientific result. The acceptance tolerance is not
weakened; the next diagnostic records the exact maximum/mean discrepancy and
index so the one-step implementation can be corrected against raw evidence.

Diagnostic retry `37052` measured maximum action discrepancy `0.001953` and
mean discrepancy `0.000375`; the maximum occurred in the gripper channel, not
an arm-pose dimension. The gate remains closed pending per-dimension pose-unit
errors and first-five gripper-sign equivalence. Job `37051` is a one-second
submission error from a mistyped expected commit and never created a run root.

Per-dimension calibration job `37054` measured maximum translation and
rotation discrepancies of only `45.767 micrometers` and `0.000124 rad`; its
first-five gripper signs were identical. The largest raw action discrepancy,
`0.002441`, remained in the gripper channel. Exact bitwise equality is not
available because EmbodiSteer's external FK/Jacobian update necessarily splits
the fused pi0.5 while-loop into separately compiled Euler calls. Before any
simulation outcome, the apparatus gate is therefore revised to physical
equivalence: raw error at most `0.005` action units, translation at most
`0.1 mm`, rotation at most `0.001 rad`, and identical executed gripper signs.
All measured values remain recorded; this is not a task- or collision-outcome
tuning decision.

Clean H100 job `37055` completed both barrier-free arms and the original
shape/count validator from commit `600cefc`. The Cartesian arm executed 300
actions, never moved the bowl or satisfied the goal, first moved the obstacle
beyond the paper CAR threshold at step 19, and contacted link 7 at step 236.
The joint arm also never moved the bowl or satisfied the goal and recorded no
contact or CAR. This pair is not accepted as an EmbodiSteer-fidelity result.
The audit found that SafeLIBERO's `JOINT_POSITION` input is a bounded delta,
but v1 encoded each absolute `Q_0` target through only `0.05 rad`: 220/300
steps saturated, mean post-step target error was `0.240 rad`, and maximum was
`0.963 rad`. The rendered/policy camera stream also developed visible
high-frequency corruption after initially valid frames; a shape-only decoder
check incorrectly passed it.

The v1 artifacts remain immutable evidence of these apparatus failures. The
Cartesian MP4/JPG are visually valid. The joint MP4/JPG are rejected despite
having 301 decodable frames and must not be presented as simulation evidence.
The result file, payload, Cartesian MP4, and rejected joint MP4 SHA-256 values
are `b58504c3df32854d555db1cb5cb0dda6824b60154be89bfcf30d4b4be8b8a92b`,
`9bc55a0194ad19a5a4b6a3632d0601f5be46bd9a522e2491fd1dc465d083870e`,
`2bc0256ad56ad7f9fd9a0c5fbc7cf1d84ca05bf71e2a0d5a1a56de55a57b1b4d`,
and `afb293d529ec153f1024e228a9e10cb0fe347459e4c96ce55adb62656d760bf5`.

Protocol v2 changes only the execution and evidence apparatus before a new
outcome: it uses the delta controller's full `6 rad` encoding range so every
bounded Panda `Q_0` configuration is represented as the exact absolute target,
runs EE and Joint in fresh evaluation processes under the same H100 allocation,
and rejects any source or decoded frame whose horizontal/vertical adjacent
pixel MAD exceeds `8`. All ellipsoids, SDFs, barriers, and QPs remain disabled.
Acceptance additionally requires zero joint-target encoding saturation and
reports target-tracking error explicitly.

H100 job `37058` completed the v2 Cartesian worker, then the fresh-process
Joint worker failed closed before a valid result. Its fixed agent-view stream
flipped by 180 degrees at action 10 and exceeded the preregistered adjacent
pixel MAD gate at action 11 (`54.698/255`). The pair therefore has no accepted
Joint outcome and cannot be compared with the paper. This also disproves the
earlier hypothesis that the visual failure was caused only by reusing one
OSMesa context across the two arms.

Protocol v3 is frozen before its H100 run. It corrects two additional
paper-rate mismatches: EmbodiSteer controls at `10 Hz` and executes the full
predicted chunk, so this pi0.5 adaptation executes all ten available actions
at `10 Hz` instead of five at `20 Hz`. The paper's actual horizon is 16;
pi0.5-LIBERO exposes only ten, so v3 remains an adaptation. The physical
obstacle is retained because this is the primary-case stress test; it is not
misreported as the paper's obstacle-free `Joint` population. Visual integrity
now also requires each frame to remain at least `5/255` closer to the initial
upright fixed-camera image than to its 180-degree rotation. A failure writes
raw/processed JPGs, the complete action and joint-state prefix, and the
initial/failure MuJoCo camera poses before terminating as apparatus failure.
All collision geometry, CBF rows, and QPs remain disabled.

Initial v3 submission `37059` passed allocation tests and the sampler gate,
then stopped before action zero because the installed native MuJoCo model does
not expose the legacy wrapper method `camera_name2id`. Its immutable run is an
apparatus failure with no baseline outcome. The compatibility repair uses
native `mujoco.mj_name2id` when the wrapper method is absent and changes no
control, policy, pairing, geometry, or acceptance parameter.

Job `37060` is a one-second submission typo: the expected full commit hash was
mistyped, so the clean-source gate rejected it before creating a run root.
Retry `37061` completed the valid 10 Hz Cartesian worker, then failed the
Joint visual gate at action zero. The atomic trace makes the cause precise:
the joint target had zero encoding saturation, peak post-step velocity was
only `0.299 rad/s`, and the MuJoCo `agentview` camera pose was bitwise
unchanged, while the processed frame was almost exactly the 180-degree
rotation of the initial frame (`0.000618/255` rotated-reference MAD versus
`71.202/255` upright-reference MAD).

The Joint worker still constructed a second rendered SafeLIBERO environment
for read-only FK/Jacobian queries; this changes the process-global OSMesa
context even though the paired arms themselves use fresh processes. The next
retry removes that second environment. It performs exact FK/Jacobian queries
on the live MuJoCo model, restores the complete flattened state before every
executed action, and requires bitwise state equality after restoration. No
policy, joint target, controller, rate, pairing, geometry, or visual threshold
changes.

H100 job `37062` confirmed that removing the second environment fixed the
renderer: the 10 Hz Cartesian arm completed, and the Joint stream remained
upright through multiple actions. The Joint worker later stopped on a
different apparatus assertion before producing a result: the server's
float32 affine normalize/decode round trip exceeded the original fixed
`1e-7` raw-action tolerance. That tolerance was not derived from the actual
float32 path and can reject a faithful value solely as its magnitude changes.

Before the next retry, the round-trip gate is frozen to `32 * eps_float32 *
max(1, |input|, |decoded|)`. This is an IEEE-precision-scaled serialization
bound, not an action or outcome tolerance. Every reverse step records its
actual maximum error, magnitude, and bound. The trajectory supplied to the
denoiser and the resulting joint targets are unchanged.

Clean H100 job `37067` completed the final v3 pair on `worker-2` in
`00:05:21` from commit
`d1b3e7b6ec9a6044a795bb6858a3005bda62b758`. Allocation validation passed
the complete result and both 301-frame upright videos. Both arms used the same
settled state, checkpoint, and noise schedule; every ellipsoid, collision SDF,
barrier row, and QP remained disabled. The maximum FK-pose float32 round-trip
error was `1.559e-7` raw action units, only `3.21%` of its precision-derived
bound.

The result does not resemble EmbodiSteer's task-competent `EE`/`Joint`
baseline relation. Both arms executed 300 actions and failed the native goal.
Cartesian `EE` first failed paper CAR and made robot contact at step 7, then
contacted the obstacle with link 6 at step 193; maximum obstacle displacement
was `0.185724 m`. `Joint` likewise failed CAR/contact at step 7 and contacted
with link 5 much earlier at step 79; maximum obstacle displacement was
`0.509158 m`. Both did close the gripper (first positive commands at steps 97
and 41), so this is not the former never-close bug.

The direct-joint execution itself fails the competence gate. It had zero
target-encoding saturation, but mean/maximum post-step target errors were
`0.255/1.225 rad`, maximum joint speed was `3.503 rad/s`, and target changes
reached `0.555 rad` per 0.1-second boundary step. It moved the bowl by as much
as `0.593 m` without satisfying the bowl-on-plate goal. This is evidence that
the pi0.5 incremental-action lift plus SafeLIBERO joint-position adapter is not
a matched EmbodiSteer `Joint` baseline, not evidence against the paper.

Result file/payload/validation-receipt SHA-256 values are
`d690616cda5ff5895fe4d5af585c2a2e364f4a428e7ef319622d116366b8bb8b`,
`a0a33571574fe273fad986e66940e8e151fe3e97635cddab70d7d6930db5ca02`,
and `77501d0ab7414bc287681530460a13f42dca32222e80120aa2127ded6c24fd48`.
Cartesian and Joint MP4 SHA-256 values are
`49273784754407074d84b90287c1c2b3311d80067e150b48b25ef205ec8d7a13`
and `d4ddf8016d7266cb3b396b2a427750135c554ceb9a630e1c10c95251feafc0dc`.
The current L5/L6 MVEEs remain disabled and unsuitable for an EmbodiSteer
claim; the paper uses multiple link-attached collision spheres with top-four
smooth SDF aggregation.

Unresolved risk: no task-competent, natively joint-executed policy/controller
has been identified for this SafeLIBERO task. No further Slurm experiment is
authorized until that choice is preregistered. The exact next command is the
read-only evidence check:

```bash
sha256sum /mnt/data/quanth/experiments/vlsa-embodisteer-joint-baselines-e05/paired-v3-20260808e/result.json /mnt/data/quanth/experiments/vlsa-embodisteer-joint-baselines-e05/paired-v3-20260808e/arms/*/episode.mp4
```

## EmbodiSteer-inspired task-metric multi-CBF flow (completed negative test, 2026-08-08)

The next active `E02` subexperiment references Wang et al., *EmbodiSteer:
Steering Embodiment-Agnostic Visuomotor Policies with Joint-Space Guidance for
Zero-Shot Cross-Embodiment Deployment* (arXiv:2606.12965). The paper's
single-constraint QP in Eqs. (6), (14), and (15) uses
`H = J_EE^T W J_EE + lambda I` to move away from collision while minimizing
end-effector disturbance, and Eq. (16) applies weak guidance to early noisy
samples and strong guidance late in denoising.

The smallest honest SafeLIBERO adaptation is frozen in
`configs/vlsa_embodisteer_multicbf_e05.v1.json`. At each live pi0.5 query it
identifies both four-body clearance and end-effector-trajectory Jacobians from
the complete cloned OSC transition. One task metric contains all 40
L5/L6/L7/end-effector trajectory-CBF definitions plus action bounds. A
metric-Dykstra projection is applied after every one of pi0.5's ten Euler
updates with the paper's fixed schedule `gamma=1`, `beta=50`, `c=0.7`; the
final chunk must still pass exact cloned ten-step verification before its first
five actions execute. The optimizer clearance remains `D_opt=10 mm`, gripper
commands are unchanged, and only XYZ may change.

This is an OSC action-space surrogate, not full EmbodiSteer joint-space
denoising: SafeLIBERO's released task-competent arm accepts Cartesian OSC
commands and does not expose joint redundancy to the policy sampler. The
four-group evidence is therefore reported without claiming a matched
EmbodiSteer reproduction: (1) immutable Cartesian EE baseline, (2) retained
direct-joint diagnostic whose different checkpoint was baseline-incompetent,
(3) retained Cartesian post-hoc three-row multi-CBF, and (4) this new
denoising-time task-metric four-body projection. Only group (4) is a new H100
run; completed Table 1 and prior negative artifacts remain read-only.

Clean H100 job `37042` completed on `worker-1` in `00:03:13` from frozen
commit `b87f1fcbc12848321b83ce70224884f80b6475c6`. Its first nominal chunk
SHA-256 `05f3960fdb6e97e0750a2f29d012bdee4dd680c1a9125c46f22e4f6538f99a6d`
exactly matches the immutable task-success Table 1 episode. All 60 queries and
300 actions completed. Eight queries at steps `0--35` activated guidance,
giving nine task-metric projection attempts; all accepted chunks passed exact
cloned-OSC verification. Raw MuJoCo evidence contains no robot contact, no
paper CAR, and maximum obstacle displacement `2.275e-11 m`.

The arm nevertheless never moved the bowl (maximum bowl-center displacement
`6.375e-13 m`), never satisfied the native goal, and timed out, so
`primary_problem_solved=false`. The initial nominal horizon violated only the
end-effector trajectory constraint (`-12.015 mm`); its L5/L6/L7 minima were
positive `79.391/103.491/73.052 mm`. Across live query states, minimum
L5/L6/L7 buffered clearances remained `103.705/127.446/99.333 mm`, while the
end-effector proxy reached the `20 mm` activation region. Thus all meaningful
intervention again occurred before the historical distal-link collision
regime.

The metric and schedule improved task preservation relative to the earlier
Euclidean hard projection but did not recover the task. The closest
end-effector-to-bowl-center distance improved from `256.941 mm` to
`184.234 mm`; the immutable successful baseline reaches `40.121 mm`. Guidance
also stopped after step 35 instead of step 75, and cumulative XYZ deviation
from the baseline through 75 actions decreased from `5.292` to `4.173` action
units. This is still insufficient: a task-preserving metric cannot preserve a
nominal task route that the hard end-effector proxy itself declares infeasible.
Result file SHA-256 is
`857e3d07d2f6f4f7712f8e08f27fe188be862525ca45e1a483abc4fff2158fdc`;
payload SHA-256 is
`3582095f2844246d186f06f37c9978b85f127112bf21b89e574468135a1a27e1`.

H100 visualization job `37044` then replayed the accepted 300-action ledger
on `worker-1` from clean commit
`fd1219b7f27009a4696ad1a1fc75212e5209f462`. Its receipt verifies exact
simulator state, end-effector, obstacle-displacement, contact, CAR, and task
traces; all 301 decoded frames; six distributed pixel-fidelity samples; and
visible L5/L6/L7/EE plus obstacle wireframes. The verified MP4 SHA-256 is
`ec5ee58889cb2a27c6a22466211aa476b6f572133d77e989e13c9a8122539c38`,
the JPG SHA-256 is
`e9ee91299e3e0f6a8b35c0e295c5e09573c50d6e38ee3e7ec2d1486e580f401b`,
and the receipt SHA-256 is
`933f6a34211456e68c553b160e27e800d27147af5c9584f0c01a63e85e1ea117`.
The final local structural gate passes 219 tests with 27 dependency-optional
skips.

## Predictive L5--L7 plus end-effector flow guidance (completed negative capability test, 2026-08-08)

The active `E02` follow-up is frozen in
`configs/vlsa_predictive_flow_guidance_e05.v1.json`. It retains the accepted
three distal MVEEs and adds the released AEGIS end-effector ellipsoid, giving
40 horizon constraints over ten future actions. A cloned SafeLIBERO rollout,
including the stateful OSC_POSE and gripper transition, supplies the local
trajectory derivatives. The correction is applied after every one of the ten
pi0.5 Euler updates and the final chunk must pass exact cloned trajectory
verification before execution. One relinearization is allowed.

Implementation, default-sampler regression, and the primary paired H100 run
are complete. Independent artifact validation remains pending. The completed
Table 1 tree remains read-only.
Initial H100 submission `37014` stopped during allocation-side unit preflight,
before policy startup or simulation: the Python-3.8 evaluation runtime could
not evaluate an existing `int | None` annotation while dynamically loading
the server helper, and a synthetic test expected the later-step residual
instead of the smaller first-step residual. The retained run root contains an
apparatus-failure receipt and no result or video. The repair postpones type
annotation evaluation and corrects only that synthetic expected value; no
experiment setting changed.
Retry `37019` passed both allocation preflights and loaded pi0.5, but stopped
before its first action because the nominal chunk no longer matched the
immutable Table 1 chunk byte-for-byte. This is retained as a second apparatus
failure, not accepted as a different live baseline. The repair restores the
ordinary `sample_actions` implementation as a separate untouched method and
moves all projection logic into a distinct opt-in compiled sampler selected
only when the reserved guidance envelope is present.
Second isolation check `37021` still failed the same pre-action hash gate. The
ordinary sampler body was exact, but policy construction still wrapped the
guided bound method alongside the ordinary method. The accepted live shadow
run `36757` constructed only the ordinary wrapper and exactly reproduced the
Table 1 chunk and 237-action task success. Guided wrapping is therefore now
lazy: it cannot occur until after a validated guidance request, while the
first nominal request follows the accepted construction path exactly.

Clean H100 job `37024` completed on `worker-1` in `00:04:42` from commit
`852070b2f2c4ed86b98411e17363d0ea22f89df2`. Its first live pi0.5 chunk has
SHA-256 `05f3960fdb6e97e0750a2f29d012bdee4dd680c1a9125c46f22e4f6538f99a6d`,
exactly matching the immutable task-success Table 1 episode. This resolves
the pre-action pairing gate without waiving it. The 300-step result and video
have SHA-256 `739adbca0287193bac477b737a7131e6fa9ed55c452ba9e7eea1113e5e4649c1`
and `df49d0d4f5a9659ad0c44f445e83ded4379995bebd2eb586e9d823933b4594d0`.

The predictive controller made 18 guidance attempts across the first 16
queries (steps 0--79), using 1,098 cloned horizon rollouts and 10,980 cloned
`env.step` calls for finite-difference identification. All accepted chunks
passed exact ten-step verification. The minimum accepted exact trajectory-CBF
residual was `2.760e-7 m`; two first proposals failed exact verification and
were safely relinearized. No unsafe proposal was passed through. The sampler
projection residual was at worst `-1.510e-9`, within the frozen tolerance.

Raw MuJoCo evidence contains no robot contact and no paper CAR over all 300
executed actions, with maximum obstacle displacement `2.275e-11 m`. However,
the bowl never moved, no native goal atom was ever satisfied, and the episode
timed out. Thus `primary_problem_solved=false`: safety by diverting the robot
away from the task is not useful SafeLIBERO completion.

The per-body audit is decisive. Minimum observed buffered clearances were
`108.667 mm` for L5, `129.095 mm` for L6, `95.291 mm` for L7, and only
`0.333 mm` for the end-effector proxy. Every intervention was therefore
caused by the predictive end-effector constraint before the later L5/L6
collision regime was reached. At query zero the exact nominal horizon had a
minimum trajectory-CBF residual of `-12.015 mm`; the accepted correction put
it at `+0.000276 mm`. After query 15 guidance deactivated, but the changed
closed-loop observations had already diverted the policy: the end effector
finished near `y=0.515 m` and the bowl pose remained unchanged. The immutable
baseline instead moved the bowl by step 75, contacted L5/L6 beginning at step
187, and completed the native goal at step 236.

The original job-`37024` MP4 is not accepted as visual evidence: although its
container opened and frame zero was readable, later decoded frames were
corrupted. H100 exact-ledger visualization job `37031` repaired the evidence
without changing or rerunning the controller. It replayed all 300 accepted
actions from the read-only result (SHA-256
`739adbca0287193bac477b737a7131e6fa9ed55c452ba9e7eea1113e5e4649c1`),
matched the recorded simulator state and end-effector traces with zero maximum
error, and overlaid the live L5/L6/L7/end-effector ellipsoids plus the frozen
obstacle MVEE. The replacement H.264/yuv420p MP4 has 301 decoded frames and
SHA-256 `34fcbb4e2ca83cdce203a8c553d2963d5fb5a4e2af8a30667e311b6e1d7e9b77`;
six distributed decoded samples have maximum mean absolute source-pixel error
`2.612/255`. The JPG SHA-256 is
`23a5f037bba9265f63e5cf1d27fa7a76a48b7de373a4c26f1444c4a44ef0677c`.
Preceding job `37029` produced the same verified bytes but ended `FAILED`
during legacy OSMesa process teardown, so it is retained as an apparatus
failure; the renderer now exits only after closing artifacts and atomically
writing the receipt, and clean retry `37031` ended `COMPLETED (0:0)`.

This result shows that cloned OSC dynamics and exact horizon verification fix
the earlier certificate-to-`env.step` mismatch for the actions they accept.
It does not show that predictive full-body guidance solves the original
task-completing failure: the released AEGIS end-effector proxy plus a ten-step
hard predictive barrier is too restrictive for this local flow-projection
controller on the primary route. Model identification alone averaged
`9.094 s` per guided attempt and guided inference averaged `0.891 s`, so the
implementation is also an oracle capability test rather than a real-time
filter. `E02` remains active and no KKT/VI or learned approximation is
authorized by this negative result.

## Cloned-step discrete L5--L7 multi-CBF (completed negative oracle test, 2026-08-07)

The next active `E02` subexperiment retains the accepted negative continuous
three-row QP and tests the same L5/L6/L7 ellipsoids with a discrete transition
model. At every archived action it estimates a `3 x 3` next-clearance Jacobian
from one nominal and six centrally perturbed cloned `env.step` calls, solves
one three-row XYZ QP, and executes only a candidate whose exact cloned next
step keeps all buffered clearances nonnegative within `1e-6 m`. Fixed fallback
scales move the QP candidate toward stop; failure to verify is retained rather
than passed through.

The full primary next state must match the accepted clone within `1e-10`, and
the ordinary AEGIS and accepted continuous multi-CBF paths remain unchanged.
Clean H100 rerun `36873` reconfirmed that comparison: all 237 three-row QPs
were valid, with 28 interventions and mean QP total time `1.690 ms`, but L5
contact and paper CAR both occurred at action 189 and the task did not finish.
Result file SHA-256 is
`bf25c0210a5b55e1ae7caa97ff3a4d7113a73a3f6b1c58c5728c3a618d4e139e`;
payload SHA-256 is
`bc257a906121e48a5a09f972ebd21c4e48ff098d483db7db8cc7c480b4bef27c`.

Initial submission `36870` failed before simulation or artifact creation
because worker-0 exposed an empty GPU-name query despite its H100 GRES record.
Allocation diagnostics `36871` and `36872` isolated the live node issue;
scientific jobs were then restricted to worker-1/worker-2. This is retained as
an apparatus failure and did not change an experiment setting.

Preliminary cloned-step H100 job `36874` exactly matched every one of its 185
executed primary transitions to the accepted clone and had no robot contact,
no paper CAR, and maximum obstacle displacement `2.275e-11 m`. It stopped
without executing action 185 because the valid three-row QP candidate and all
preregistered scales toward stop failed exact next-step verification. The
controller result is retained, but its artifact omitted the detailed terminal
filter record and is evidence-incomplete. The replacement changes only that
serialization, not a controller parameter.

Authoritative H100 job `36875` reproduced that outcome on `worker-1` from
clean commit `5821bfb49ecf284c8ae61b75c23ba83343c7efa4`. Result file SHA-256
is `5f03bc323949fc9f13a3169b28fbe11ce7fe83669ce5b5807dc6cf6f79ecf3d4`;
payload SHA-256 is
`b656837a543699850e746dedf4f5f445b0a323a18ebccfdf452c5d7e8931b687`.
All 186 attempted QPs contained exactly three valid constraints. All 185
executed transitions exactly matched their accepted clones (maximum state
error zero), with no robot contact, no paper CAR, and maximum obstacle motion
`2.275e-11 m`; however, the task was unfinished when the controller refused
action 185.

At the terminal step, nominal next L5 buffered clearance was `-8.487 mm`.
The QP predicted its large corrected action would reach exactly `0 mm`, but
the exact cloned step measured `-3.697 mm`. Scaling that action to zero made
the exact L5 result progressively worse, ending at `-6.852 mm`; therefore no
preregistered candidate was executable. This is a valid method failure, not a
collision-free task success: the one-step oracle acts as a reliable veto but
reaches the viability boundary too late to preserve useful motion.

The run used 1,493 cloned simulator steps. QP total wall time was mean
`1.182 ms`, p95 `1.483 ms`, maximum `2.319 ms`; complete filtering was mean
`131.895 ms`, p95 `169.939 ms`, maximum `208.820 ms`, exceeding the `50 ms`
period of the 20 Hz controller. The strict primary problem remains unsolved,
`E02` remains active, independent validation remains pending, and no learned
controller is authorized by this negative one-case oracle result.

## AEGIS Cartesian-to-joint bridge versus L5--L7 multi-CBF (completed pilot, 2026-08-07)

The preregistered repair retained the task-competent released AEGIS Cartesian
action ledger and replaced only the low-level execution interface. Each
archived XYZ command is converted at the live state to a seven-joint velocity
with a damped least-squares Jacobian bridge; the bridge-only arm executes that
nominal velocity, while the active arm executes the solution of one QP with
the three separate L5/L6/L7 ellipsoid constraints. Both arms restore the exact
archived OSC-settled MuJoCo state, use the same 237 actions and gripper
commands, and are exactly paired through the first material QP intervention
at action 200.

Producer H100 job `36824` completed on `worker-1` in `00:03:32` from clean
commit `25f91ebf224348a1d692bb0fb0d3cbc0be6af61f`. Result file SHA-256 is
`8b1b5e077a2eddeef9c06757faa0d4d6e9ed651cf5dd5b2f71156b2582628e14`;
canonical payload SHA-256 is
`1ad2097a173cae9364abd7b3af900c367476668e5e90278cf48de36d103cb961`.
Independent H100 validator job `36826` completed on `worker-1` from clean
commit `fe3f04a4f92417cd201ab067af7eb126149b4980`, decoded both 238-frame
videos, checked all 474 action records, recomputed raw simulator outcomes,
and verified every three-row QP residual. Validation file SHA-256 is
`a95c2fdbe2d4fe2b9d6f80452d517377581fc6a404219b8b9c0f4fb90886bf5b`;
receipt payload SHA-256 is
`4abfbe076541503256b03a34a1f7a529af6097cbf98c66727b0a6ab25bfd3af4`.

The competence gate fails. Neither arm satisfies the native task. Both first
cross paper CAR at action 157 and first contact the obstacle at action 159,
but the contacting geoms are `gripper0_finger1_collision` and
`gripper0_hand_collision`; no protected L5/L6/L7 contact occurs. Maximum
active-obstacle displacement is `0.244348 m` for bridge-only and `0.245901 m`
for active multi-CBF. The collision mechanism therefore changed before the
safety comparison: replaying feedback-dependent Cartesian actions open-loop
through a different joint controller does not preserve AEGIS task competence.

All 237 QPs per arm are valid, with minimum independently checked raw residual
about `-2.78e-17 m/s`. The active filter intervenes once at action 200. Its QP
time is mean `1.972 ms`, p95 `2.550 ms`, maximum `5.048 ms`; total filter time
is mean `2.884 ms`, p95 `3.783 ms`, maximum `6.273 ms`. Bridge-only video
SHA-256 is
`ab0f07e06a6eee733f3bd759f0b8165f24974aed5ca789f82bee1806c161ce9d`;
active video SHA-256 is
`5226a081152e121c96a0a4a79bec6882e32150e6d4fa67c6020723cfbcb2c992`.

The retained interpretation is
`bridge_incompetent_no_safety_efficacy_claim`; `safe_problem_solved=false`.
No bridge scale, QP gain, clearance, or geometry is tuned after this result.
The unresolved risk is closed-loop policy/controller compatibility: the next
experiment must be newly preregistered and query a task-competent Cartesian
policy on the joint-controller arm's live observations before any L5--L7
safety claim or KKT/VI training. The exact read-only handoff command is:

```bash
jq '{status, interpretation, baseline, active}' /mnt/data/quanth/experiments/vlsa-aegis-cartesian-joint-velocity-bridge/vlsa-aegis-cartesian-jv-bridge-e05-20260807c/validation.json
```

## Direct joint-velocity baseline versus L5--L7 multi-CBF (completed pilot, 2026-08-07)

The preregistered paired pilot replaced the policy/control interface with the
published `pi05_droid` checkpoint's direct seven-joint-velocity output, while
keeping the active safety arm to exactly three independent L5/L6/L7
ellipsoid constraints. Both arms restored the same primary-case simulator
state and ran the frozen 225-action, 15 Hz horizon. The first policy chunks,
nominal actions, and simulator states are exactly paired through the first
material intervention at action 71.

Producer H100 job `36802` completed on `worker-1` from clean commit
`9c3191a8d851d9ea18306257222307b995d07928`. Result file SHA-256 is
`e8469841fc55258d894e34493ce7809f2a2b95d3726c9c94b624e7ae9a8c2242`;
canonical payload SHA-256 is
`bda99a32fb7bacae15881b6e25cf09cc9740e7bf7170da58848ca176a1f4c118`.
The 12.43 GB checkpoint tree is bound by SHA-256
`9bef87f85aa1961e2756b89e2779500dd8433ff67ea88f219fba5702da2ecacb`.

The unfiltered baseline had no robot contact and no paper CAR, but it also
never closed the gripper in 225 actions and never satisfied the native
`on(akita_black_bowl_1, plate_1)` goal. The active arm made 43 material
joint-velocity corrections, briefly closed the gripper for two actions, and
also had no contact or CAR, but its maximum and final native goal fraction
remained zero. Thus `baseline_competent=false` and
`safe_problem_solved=false`: this pilot validates the direct joint-space
control and multi-constraint plumbing, but cannot establish safety efficacy
on the original task-completing collision case.

All 225 three-row QPs in each arm were valid. The active arm reduced the
worst predicted post-step optimizer gap from the baseline shadow's
`-0.033406 m` to `-0.003117 m`. Active QP time was mean `1.706 ms`, p95
`2.322 ms`, maximum `3.800 ms`; total filter time was mean `2.602 ms`, p95
`3.963 ms`, maximum `5.359 ms`. The baseline video SHA-256 is
`c0f557ca88d3c7bf7f9e922c0f39f802e11531181a864d7ea7a7f7a76fcae61a`;
the active video SHA-256 is
`bab5ae351e41d3af96e34100e76f8192a8769fdee260c4a55d7d9b95f9027033`.

Independent H100 validator job `36806` completed on `worker-1` from clean
commit `71b85da6f0a7c1bf8e7b212f25cc74bbd74faf4a`. It rehashed the result and
checkpoint, decoded 226 frames per video, checked all 450 action records,
recomputed raw contact/CAR/task summaries, and verified every QP residual.
Validation file SHA-256 is
`e63fd013e80735b111c899b908bb12dbe3cb97389fcc72362874bb60b1a592f8`;
receipt payload SHA-256 is
`f297c1888f886742add6c3492410d660f952ab1751973964926a5a7cd83d1dd7`.

The unresolved risk is policy/domain competence, not QP execution: the
zero-shot DROID checkpoint is not a competent SafeLIBERO policy for this
case. No KKT/VI training or further H100 experiment is authorized by this
negative pilot. The exact read-only handoff command is:

```bash
jq '{baseline_competent, safe_problem_solved, interpretation}' /mnt/data/quanth/experiments/vlsa-pi05-droid-joint-velocity-pair/vlsa-pi05-droid-jv-pair-e05-20260807a/result.json
```

## Distal three-ellipsoid refinement (active, 2026-08-07)

Per user correction, `E02-distal-three-ellipsoid-shadow` now targets exactly
link 5, link 6, and link 7; it does not claim whole-arm coverage. The local
implementation fits one close, certified MVEE to each compiled collision
mesh and passes exactly three analytic rigid-link constraints to the same
read-only OSQP. Allocation-backed visualization and primary-case timing are
still pending.

Visualization attempt `36771` stopped before environment construction because
the new experiment parent directory did not exist. It produced no simulation
or geometry evidence; the wrapper now creates that bounded parent first.

H100 visualization `36772` then completed, but visual inspection rejected the
covariance-shaped L5 envelope as too loose despite correct vertex containment.
That image is diagnostic only. The active fit is now Khachiyan MVEE followed
by exact farthest-vertex inflation; it requires a fresh H100 render.

H100 visualization `36773` completed from clean commit `36681ca`; its MVEE
payload SHA-256 is
`4d86c6a88aaa11bcf04de94b8c30fc421fe8a3042767025e44261f1337447b5a`.
Visual inspection accepts the separate L5, L6, and L7 surface fit.

Full-policy attempt `36774` completed 300 actions but failed the pairing gate:
the first GPU policy chunk already differed from the immutable archived chunk,
so the robot timed out rather than succeeding. The shadow never changes an
executed action, and this attempt is not scientific comparison evidence. The
replacement validation replays all 237 immutable archived `env.step` inputs,
requires stepwise simulator-state/contact/goal equality, and recomputes only
the read-only three-link QP. The rigid-link MVEEs are now fitted once and
cached in body coordinates rather than refitted at every action.

Exact shadow replay `36775` and independent H100 verifier `36776` passed. All
237 immutable actions supplied exactly three valid QP constraints; the shadow
first identified L5/L6 nominal violation at action 186, one action before raw
link contact. The current active subgate now applies those constraints to the
executable XYZ action while preserving zero rotation and the archived gripper
command. Its frozen pass condition requires no L5/L6/L7 contact, paper CAR,
and native task success within the same 237-action nominal horizon.

Preliminary active run `36781` passed those outcome checks, but OSQP emitted
sub-`1e-8` displacement from the exact nominal optimum on feasible steps. The
final replay snaps only those verified-feasible solutions to the exact AEGIS
XYZ and uses `1e-6` L2 as the material-intervention threshold. This is a
numerical identity correction; the CBF parameters and unsafe-step solution
remain unchanged.

Authoritative H100 replay `36782` completed on `worker-1` in `00:01:41` from
clean commit `ad05dac8ea3f51fff69f6857d1a819ffbe1be2b1`. Result file SHA-256
is `d28e29c06820817157d01c27d3f5fd366825992bcc75234086b36a868d9d80f0`;
payload SHA-256 is
`950628aaaa1f63ede455abdc044add2fa08a189468452c0444840f36a81e15b8`.
All 237 QPs solved with exactly three rows. The controller preserved the exact
archived action on 209 steps and made 28 material interventions, starting at
action 182. QP total wall time was mean `1.736 ms`, p95 `2.297 ms`, maximum
`4.831 ms`; solver-only mean was `0.0862 ms`.

The preregistered outcome is negative: `primary_problem_solved=false`. Direct
`robot0_link5_collision` contact begins at action 189 and persists through
193, with minimum raw MuJoCo distance `-0.0008023 m`. Paper CAR also first
fails at 189, maximum active-obstacle displacement reaches `0.0266153 m`, and
the native task never succeeds within the 237-action horizon. Relative to
released AEGIS, the filter delays link contact by two actions and CAR by one,
but does not prevent either and loses task success. Every solved action-space
constraint passed its QP residual check (minimum raw residual about
`-5.6e-17`), so the remaining gap is between the high-level resolved-rate
linearization and discrete OSC/simulator motion, not an ignored QP failure.

No gain, clearance, geometry, or fallback is tuned after this outcome. The
next controller experiment requires a new preregistration, most directly for
joint-space execution or discrete-time nonlinear rollout/step reduction.
Local rehash reproduced the result payload identity exactly, and the final
`./init.sh` gate passes 192 tests with 24 dependency-optional skips.

H100 job `36788` then replayed the accepted 237-action ledger on `worker-1`
and produced an annotated 20 fps MP4 with a four-times-slow interval around
actions 178--196. Its receipt verified all pairing fields plus exact
end-effector, obstacle-displacement, raw-contact, CAR, and task traces; all
three maximum replay errors are zero. Video SHA-256 is
`dc7df3d03b5e893b0566e97a57c90be9593775948f8984aa27c90a98b0e17df0`;
receipt SHA-256 is
`5bc7dd8a97b8f0b23e2dc88b1c4ac74f68e49924ac611021e59e29e472027883`.

The stepwise audit confirms a resolved-rate/discrete-transition mismatch. At
action 185 the L5 row predicts next `h_opt` near `+2.220 mm`, while the next
MuJoCo state is `-4.441 mm`. At action 186 the QP predicts recovery at
`+44.412 mm/s`, but the observed finite-difference rate is `-143.104 mm/s`.
The detailed record is in
`docs/distal_three_ellipsoid_multicbf_failure_audit.md`.

## Multi-link ellipsoid research branch (2026-08-07)

`E01-multilink-ellipsoid-shadow` is passing on branch
`codex/multilink-ellipsoid-qp`, based exactly on clean AEGIS reproduction
commit `1592aa59361f431ba96c6ddcbebcb596f6c20853`. The historical Table-1
population is not modified, resumed, or reinterpreted on this branch.

Read-only raw-artifact verification for primary case
`vlsa-t1-goal-ii-t0-e05` established the preregistered target:

- archived AEGIS `result.json` file SHA-256 is
  `273d77cba3fd5b1e457817ad20f8572b5628aeaa8b5b9f768b32e4f095fbad8b`;
- the released one-constraint OSQP is solved at action 187 with positive
  end-effector proxy barrier `h=0.009312042732980995`, while raw MuJoCo
  evidence records direct `robot0_link5_collision` contact with the active
  moka pot at that action and a link-6 pair later in the episode;
- paper CAR crosses at action 188, AEGIS executes 237 actions total, and the
  native SafeLIBERO goal first becomes satisfied at action 236. Thus the
  robot executes 49 actions after the paper collision and completes the task.

Implementation evidence before H100 execution:

- the ordinary evaluator remains unchanged unless
  `--multilink-ellipsoid-shadow-config` is supplied together with AEGIS mode
  and failure diagnostics;
- the observer constructs one certified enclosing ellipsoid for every
  contact-participating collision geom on `robot0_link1` through
  `robot0_link7`. Primitive sphere, ellipsoid, capsule, cylinder, and box
  bounds use closed-form enclosures; mesh/unknown geometry uses MuJoCo's
  conservative `geom_rbound` sphere;
- every link ellipsoid is paired with the same frozen obstacle MVEE used by
  released AEGIS. Analytical world-twist derivatives are mapped through live
  MuJoCo Jacobians into one joint-space constraint per pair;
- the exact OSQP minimizes a positive-definite end-effector-preserving
  seven-joint velocity metric subject to all pair constraints and physical
  velocity bounds. It records setup, solve, OSQP, and total wall time, and
  retains infeasible/failed outcomes explicitly;
- `D_opt=0.01 m` is only the optimizer support-gap buffer. `D_sim` remains
  raw MuJoCo contact and active-obstacle displacement evidence; neither is
  substituted for the other;
- the shadow QP never changes the action passed to `env.step`. The H100
  validator requires exact nominal, executed, virtual-direction, policy-noise,
  and returned-action ledger equality with the immutable archived AEGIS case;
- focused geometry/contract tests pass: 10 tests ran with the allocation-only
  OSQP dependency test skipped locally. The complete `./init.sh` gate passes
  185 tests with 22 dependency-optional skips in 205.773 seconds. The OSQP
  test remains mandatory inside the H100 allocation before simulation.

No learned steering or KKT/VI model is authorized in `E01`. No further
experiment is authorized on this gate; active execution requires a new `E02`
preregistration that addresses the observed hard-QP infeasibility.

H100 result (2026-08-07):

- Slurm job `36757` completed on `worker-1` in `00:02:59` from clean commit
  `4822fd416d7e979f6b51e15428ddd6db1b5b5985`. The allocation-side numerical
  preflight passed all 7 tests, including the coupled OSQP test.
- Validation receipt SHA-256
  `390683dae335f05ebcec8c339bb5fb8fc577a1a7129002c0139a333c9cdb22bb`
  has status `validated`. The candidate preserved the archived initial state,
  policy-noise schedule, and exact executed-action ledger.
- The live model exposed exactly one collision mesh on each of link 1 through
  link 7. Independent validation recomputed every bound from the recorded
  MuJoCo type, size, and `geom_rbound`; all seven bounds passed. Because all
  were meshes, each conservative ellipsoid is the corresponding
  `geom_rbound` sphere, with radii `0.175118`, `0.171559`, `0.164639`,
  `0.165694`, `0.193834`, `0.132899`, and `0.090883` m.
- Every one of 237 actions supplied exactly 7 joint-space barrier rows to one
  simultaneous QP. OSQP solved 182; the other 55 were explicitly reported as
  primal infeasible, beginning at action 182 as link 5 approached the
  obstacle. No failed solve was dropped.
- Total QP wall time was mean `1.478 ms`, median `1.282 ms`, p95 `1.931 ms`,
  maximum `2.009 ms`. Solver-only time was mean `0.0747 ms`, median
  `0.0651 ms`, p95 `0.0967 ms`, maximum `0.801 ms`.
- Raw MuJoCo evidence still records direct link-5/link-6 collision, first
  robot contact at action 187 with minimum contact distance `-0.0010594 m`,
  while released AEGIS reports positive end-effector barrier
  `h=0.00931204`. The unchanged robot travels another `0.172512 m` at the
  end effector and first satisfies the native task goal at action 236.
- Slurm visualization job `36767` completed on `worker-1` H100 in 22 seconds
  from clean commit `fbd5df6bb3f0b336f8bee9892aecf1731b252368`. It restored
  the same primary initial state, settled 20 actions, framed all seven live
  bounds in a MuJoCo camera, and published visualization payload SHA-256
  `676adc159496b7d8fadc78fd3799776c71ed59025be9ad1de272e59379a42993`.

This validates the whole-arm bound construction and multi-constraint QP
implementation/timing, and independently confirms that original
end-effector-only AEGIS misses the physical upstream-link collision while
useful task motion continues. It does not show that executing the proposed QP
is safe or task-preserving. The 55 infeasible hard QPs are the principal risk
for `E02`; KKT/VI learning cannot repair an infeasible oracle target.

## Objective

Reproduce the translational-action portion of AEGIS Table 1 from the authors'
release, retain videos for every SafeLIBERO rollout, and explain both collision
and task failures without silently excluding apparatus failures.

## Historical Table-1 gate (frozen branch context)

`A04-population` was active on the original reproduction branch. It is pending
and out of scope on this research branch so that `E01` is the only active
gate. The apparatus, capture, and action-invariant paired-canary gates passed
in dependency order; the text below is retained as historical context and is
not a new interpretation of completed Table-1 artifacts.

Local implementation evidence on 2026-07-18:

- The focused diagnostics/publisher suite passes 162 tests with one
  dependency-optional skip. The full `./init.sh` structural gate passes all
  175 tests with 18 dependency-optional skips.
- The immutable manifest contains exactly 1,600 cases in 32 groups of 50;
  paired evaluation requires exactly 3,200 terminal episode results.
- The immutable capture population and its independent validator passed for
  all 1,600 cases, and the actual settled MuJoCo state established exactly one
  authoritative in-workspace obstacle for every case.
- The outcome-blind Codex label manifest contains all 1,600 cases, has
  SHA-256
  `f9a862f28f168f02de4e0987e37d297de24b167ae50fb96c7f8243a76916880e`,
  and reuses the accepted ordinal-100 canary row byte-for-byte. Its
  publication receipt has SHA-256
  `e83611f46ce5fbb13c84f74db3825ab114bf7184db96b62be2965c7a0c5b9e20`.
- The revised canary runs four serial rollouts in one allocation: pi0.5 with
  diagnostics off/on, then pi0.5+AEGIS with diagnostics off/on. It requires
  exact action-byte, query-schedule, outcome, simulator-state, frozen-label,
  and decoded-video invariance between each off/on pair.
- The historical ordinal-100 action reference is frozen at SHA-256
  `1a06b4842b356eb0fd6671b214aaea7d63cb2d9878982aa6817d305a6489fdf1`.
  It also pins the validated baseline collision/time-limit and AEGIS
  collision-free task-success outcomes.
- The revised paired canary is frozen to GroundingDINO on CPU, matching the
  validated historical canary that produced the action reference. The exact
  Python 3.8 interpreter, ImageIO packages, and bundled FFmpeg binary are
  allocation- and receipt-bound.
- Checkpoint receipt task `28589_0` completed on `worker-1`. Receipt SHA-256
  `4336a202b2519f461c9b715dc8a30e09f897756a1efb27c0af096582b37e87d9`
  binds full checkpoint-tree SHA-256
  `7c81971fafcdbc677b0e8fd25b3bffd3d624abe4635323784f1918abdcef6f15`.
- Action-invariant canary task `28590_0` completed on `worker-2` in 7 minutes
  20 seconds from clean release commit
  `5105894faebd40d2e27e2011d33688b25c2578dc`. Allocation-side receipt
  SHA-256
  `6a80afce3ce019bf43090da65c85717344b6dafb4a983d5a9efcf6bbecd46d95`
  has status `validated`.
- Independent local regeneration validated all four artifacts. For both
  pi0.5 and pi0.5+AEGIS, diagnostics off/on preserved exact action bytes,
  policy-query schedules, simulator outcome semantics, and video bytes. The
  baseline collided at step 70 and timed out after 300 actions; AEGIS remained
  collision-free and completed the task after 145 actions. This is apparatus
  evidence for one frozen case, not population efficacy.
- The full population remains unauthorized until its publisher validates all
  per-case detector, point-cloud, MVEE, QP, contact, goal-progress, terminal
  frame, and video evidence without retaining all 3,200 large result objects
  in memory.
- The memory-safe publisher now validates one full pair at a time, then
  independently recomputes and byte-compares the final aggregate, exhaustive
  failure report, and indexed gallery from the immutable result tree.
- Contact explanations are now bound to the settled active obstacle, immutable
  BDDL goal arguments, full MuJoCo body/geom/joint topology, and an ordered
  raw contact ledger. Dynamics, mobility, and roles are independently
  reconstructed from frozen joint ownership/types/names. The validator also
  reconstructs the complete robot-body ID set from frozen body names and
  binds every task object and goal-site parent to its exact frozen MuJoCo root
  name. Direct goal objects, parented goal sites, and parentless fixed sites
  are handled separately; unknown roles fail closed. Rehashed attempts to
  omit the contacted robot or swap two task-object roots are rejected.
- Independent adversarial review gives a conditional GO for a fresh paired
  canary only, after the final clean release is frozen. Because the contact
  observer changed after task `28590_0`, the next allocation-backed steps are
  a new clean source-bound checkpoint-tree receipt and a new ordinal-100
  four-run action-invariance canary. Population remains unauthorized.

Checkpoint hash task `28467_0` completed on worker-2 in 16 seconds. Its full
content-tree SHA-256 is
`7c81971fafcdbc677b0e8fd25b3bffd3d624abe4635323784f1918abdcef6f15`;
the small receipt was independently revalidated locally.

Paired-canary attempt `28468_0` terminated during asset preflight on worker-1
after one second, before policy, simulator, GroundingDINO, or QP execution.
The checkpoint bytes had not changed. The hash receipt included Linux
`st_dev=1048662`, while the same shared file is exposed as `st_dev=1048723`
from another cluster mount namespace with identical path, size, inode, mtime,
and ctime. The repair removes only mount-local `st_dev` from the cross-worker
identity and retains the full tree hash plus stable file metadata. Run
`vlsa-table1-paired-canary-20260717a` is immutable and remains an apparatus
failure; it is never reused.

Retry-B task `28470_0` passed the repaired evaluation preflight, then stopped
before the first policy query/action because the Python-3.8 SafeLIBERO
environment does not implement `str.removesuffix`. The wrapper now uses the
equivalent suffix slice and statically excludes both Python-3.9-only string
helpers from all runtime reproduction modules. Immutable retry-B remains an
apparatus failure and is never reused.

The first capture case is frozen as ordinal 100,
`vlsa-t1-spatial-i-t2-e00`. This is the Level-I version of the qualitative
task shown by the user: pick up the black bowl on the stove and place it on the
plate. The capture is apparatus evidence only; it cannot establish either
baseline failure or AEGIS success.

Capture attempt `vlsa-table1-capture-canary-20260717a`, Slurm task `28460_0`,
terminated before reset on worker-1. Source/protocol preflight passed, but the
legacy Robosuite stack could not initialize EGL because the allocation cannot
open the host `/dev/dri` render devices. The run is preserved as an apparatus
failure with zero reset, settle, policy, perception, QP, or outcome actions.
The repair selects the OSMesa headless path already proven by this project's
VinUni workloads; it does not alter simulator state or policy behavior.

Capture retry `vlsa-table1-capture-canary-20260717b`, Slurm task `28461_0`,
successfully initialized OSMesa but terminated before reset when the
OffScreenRenderEnv constructor could not generate its temporary randomized
object placement. The author evaluator calls `np.random.seed(7)` before that
constructor; the wrapper had omitted this ordering while still calling
`env.seed(7)` afterward. The repair restores the author's pre-construction
NumPy seed. The failed run again executed zero reset, settle, policy,
perception, QP, or outcome actions.

Capture retry `vlsa-table1-capture-canary-20260717c`, Slurm task `28462_0`,
completed on worker-1 in 23 seconds from clean commit
`bce7737369328da06d62d3840ae577a52b46780f`. It executed exactly one reset,
one immutable-state restore, and 20 settling actions, with zero policy,
semantic-selector, GroundingDINO, filtering, MVEE, QP, or outcome calls. The
capture payload SHA-256 is
`4fea432ba3ebc62b2c115e0804ade2e28da7243520a04cbb2fe1adbadedc891e`.
Local revalidation independently reproduced that payload hash, the stored NPY
hash, and settled agent-view array hash
`b8bcd1a309fbfc900d18ed472853bbec56a9e8e1fa98c55462a4960782560e7b`.
Visual review identified the exact active obstacle as `blue moka pot`; the
one-row Codex label manifest was frozen before any outcome at
`labels/vlsa_table1_canary_labels.jsonl`, SHA-256
`2d4d1be5c0a4940c72eb452d00361f6a4935de3f5cbc1058c9671fff96a35a36`.
This is allocation-backed capture evidence only. No baseline or AEGIS outcome
has run.

Evidence established before implementation:

- Source begins at untouched upstream commit
  `57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b`.
- The Table 1 screenshot is the translational protocol, not the released
  full-action README command.
- SafeLIBERO contains 32 scenarios and 50 frozen initial states per scenario:
  1,600 episodes per method.
- The released translational evaluator cannot execute unchanged:
  it omits the required suite argument to `filtering_points`, rejects the
  `safelibero_long` suite name, and has an undefined infeasible-QP fallback.
- The released repository has no SafeLIBERO-only pi0.5 runner, no population
  manifest, no aggregator, and no structured result schema.
- The released GLM-4.5V selector has an empty API key. The user requested Codex
  labels instead; these must be frozen from pre-outcome images.
- VinUni has the pi0.5-LIBERO and GroundingDINO checkpoints. OpenVLA-OFT is not
  installed and the paper does not identify the exact public checkpoint used.

## Gate order

1. `A01-reproduction-apparatus` — manifest, capture, frozen-label contract,
   paired pi0.5/AEGIS evaluator, aggregation, tests.
2. `A02-one-case-capture` — allocation-backed settled-image capture only.
3. `A03-one-case-paired-canary` — one translational pi0.5/AEGIS pair with
   identical initial state and per-request flow-noise schedule.
4. `A04-population` — all 1,600 cases for both arms; every case terminal.
5. `A05-analysis` — Table 1 comparison, indexed videos, and failure taxonomy.
6. `A06-openvla` — optional OpenVLA-OFT row after its exact checkpoint and
   environment are preregistered.

No population job may launch before a paired canary from the final exact
release is validated.

## Known interpretation limits

- Paper CAR is operationalized by the release as active-obstacle L1
  displacement strictly greater than 1 mm. It is not a direct collision,
  clearance, or whole-arm-contact measurement.
- Codex replaces only GLM semantic selection. Results must be named
  `AEGIS conditioned on frozen Codex labels`, not exact end-to-end AEGIS.
- The published Long-suite translational CAR of 79.63% cannot be produced by a
  single 400-episode population, whose resolution is 0.25 percentage points.
  This is an unresolved reporting ambiguity.
- Full pi0.5-versus-AEGIS evaluation is 3,200 rollouts and is expected to take
  roughly 50--100 GPU-hours under the two-GPU user limit.

## Original AEGIS EE-only Cartesian/Direct-Joint pair (completed timing audit)

The next primary-case run disables every L5/L6/L7 ellipsoid and CBF row.  It
uses exactly the released Table-1 end-effector proxy (`0.06, 0.12, 0.11 m`),
the archived E05 obstacle MVEE, and the released six-variable translational
CBF-QP.  The archived MVEE is bound to immutable Table-1 result file SHA-256
`273d77cba3fd5b1e457817ad20f8572b5628aeaa8b5b9f768b32e4f095fbad8b`,
settled state SHA-256
`5a72a870b8368d0a6508428bb89dce75ccc6f28e86a2349edae35b2618918741`,
and settled agent-view SHA-256
`b66103d274cbd2db02ef7fe84771725819fb74062ef36de18b6af6119da7865f`.
Completed Table-1 artifacts remain read-only.

The Cartesian arm executes the released translational filter directly.  The
Direct-Joint arm first produces the same paper-derived joint-denoising chunk,
evaluates the same released Cartesian EE QP, and lifts only the resulting
translation correction to the joint target using the live 6x7 EE Jacobian.
This is explicitly a post-hoc controller adaptation, not EmbodiSteer's
denoising-time whole-body guidance.  Acceptance requires one and only one EE
row per executed action, zero L5--L7 geometry/rows, optimizer-row feasibility,
raw MuJoCo contact/CAR reporting, QP timing, and decodable upright videos.

H100 attempt `37077` passed allocation tests and sampler equivalence, then
failed before action zero because the fresh 10 Hz arm's re-rendered RGB bytes
did not equal the archived Table-1 capture.  Its flattened simulator state did
match exactly.  The run is retained as an apparatus failure.  The corrected
source gate keeps exact simulator-state equality and additionally requires
the active obstacle position to match the archived numeric position within
`1e-9 m`; it records both camera hashes but does not require render bytes from
fresh contexts to match when the archived MVEE is reused numerically.

Clean H100 retry `37080` completed on `worker-1` from commit
`fd6625844536da8ede9e28c79efca824cab3d8bd`; the allocation-side validator
passed all one-row QPs and decoded both upright videos.  The archived obstacle
position matched within `2.242e-12 m`.  Cartesian solved 105 QPs (mean
`6.409 ms`), completed the task at step 104, but directly contacted L5/L6 at
step 85 and failed CAR at step 86 (`0.016973 m` maximum displacement).  Direct
Joint solved 300 QPs (mean `6.440 ms`) but failed the task, failed CAR at step
8, contacted L7 at step 27 and eventually L5/L6/L7, and displaced the obstacle
by `0.288681 m`.  Its mean/max target tracking error was `0.289/1.029 rad`.

Therefore the original EE-only constraint does not solve distal-link safety
under either controller.  Cartesian retains useful task competence but is not
safe.  The Direct-Joint post-hoc adapter is neither safe nor task competent,
so it cannot support an EmbodiSteer efficacy claim.  Result/validation file
SHA-256 values are
`083c92e11c3b63b9275b8fee6d5b541ccfb87de5c3e4a341afdfeedd7ffb8073`
and
`e057126ec4ee3b6f0c30bb9dbec35eb4f4eb8ab63272ab035063757b90caaa7b`.
The exact next audit command is:
`jq '.geometry_isolation,.comparison,.arms|.' /mnt/data/quanth/experiments/vlsa-embodisteer-aegis-ee-pair-e05/paired-aegis-ee-20260808b/result.json`.

The subsequent paper audit withdrew the earlier inference-rate interpretation:
EmbodiSteer's reported `103 ms` / `9.61 Hz` is the time for a full guided
inference call, not evidence for a `10 Hz` simulator control rate.  Protocol
v2 restores the released AEGIS values (`20 Hz`, five executed actions/query,
`0.05 s` internal QP step) while leaving the v1 evidence immutable.

Clean H100 jobs `37083` and `37084` completed on worker-1 from commit
`72f54892658e357f65c9beff952b38cd35f43703`; both allocation validators passed
and decoded all four upright videos.  Job `37083` is the actual paper-derived
`Joint` ablation: all guidance is disabled.  Neither its Cartesian nor Direct
Joint arm completed the task.  Direct Joint contacted L5 at step 97, failed
CAR at step 11, displaced the obstacle by `0.231396 m`, and had mean/max target
tracking error `0.218/0.978 rad` despite zero target-encoding saturation.

Job `37084` is separately and accurately named joint denoising plus post-hoc
AEGIS EE correction.  Cartesian completed at step 231 but contacted L5 and
failed CAR at step 196.  Direct Joint completed 300 one-row QPs (mean
`6.575 ms`), did not contact L5--L7, but contacted the hand/finger at step 15,
failed CAR at step 16, displaced the obstacle by `0.034719 m`, and did not
complete the task.  Its mean/max tracking error remained `0.183/0.701 rad`.
The no-guidance result/validation SHA-256 values are
`2b1705fdba6e9627c48b5f2a63874c13f1b04ca9d9c2a0574b13509456fb799c`
and `9d575e706a3e412d6fd92272e41609e0167aef43227aaf918d6fe42637d7b1b1`;
the AEGIS result/validation values are
`5cc05303b889cf07a267a5c790e021abff32783d356096f1ff6b49390a8631fc`
and `517318edcf3accba5a485703f9dea1b9b9e16842e12caebc5816c7bebb921593`.

The implementation audit confirms the published Joint heuristics are already
present (`alpha=0.1`, damped pseudoinverse `0.001`, joint residual clip
`0.5 rad`).  What remains unmatched is the native policy/control stack:
pi0.5-LIBERO exposes incremental 7D OSC flow actions with ten Euler updates,
not the paper's chunk-start pose DDPM denoising and native joint action
interface.  No scalar paper heuristic repairs that incompatibility.  The
exact next audit command is:
`jq '.comparison,.arms[]|{action_count,raw_simulation_evidence,joint_target_execution,aegis_ee_qp_timing}' /mnt/data/quanth/experiments/vlsa-embodisteer-aegis-ee-pair-e05/original-rate-aegis-ee-20260808a/result.json`.

## Accepted tight L5--L7 ellipsoid geometry with released AEGIS EE retained

The original AEGIS end-effector proxy remains exactly as released: its center
and orientation come from the authoritative `robot0_grip` site plus the
released `-0.08 m` local-z offset, and its semiaxes remain
`[0.06, 0.12, 0.11] m`. It is not replaced, resized, or counted as one of the
distal link parts.

The first multi-part attempt used common-apex convex-hull facet groups. H100
job `37086` proved hull enclosure, but visual review rejected that fit because
one L5 ellipsoid still spanned nearly the complete link and retained excessive
empty space. That immutable artifact remains diagnostic evidence only.

The accepted opt-in v4 geometry cuts each compiled collision hull into
contiguous slabs along its dominant PCA axis: three for L5, two for L6, and two
for L7. Every clipped slab vertex is enumerated from original hull vertices
and hull-edge/cut-plane intersections. Its MVEE contains those vertices and
therefore the full clipped convex polytope. Adjacent cut bounds are identical,
so the seven-ellipsoid union covers the full L5--L7 collision hulls without an
unprotected gap.

Clean H100 job `37087` completed on `worker-1` in 20 seconds from commit
`9fb0cc8b48afe923ea0a6a9990954fe04873b6d0`. All 21 allocation numeric tests
passed. All seven certificates report maximum normalized quadratic below one,
complete clipped-polytope containment, contiguous boundaries, and full
hull-union containment. The major semiaxes changed as follows:

- L5: single `0.237611 m` to parts `0.084948/0.107840/0.083307 m`;
- L6: single `0.111524 m` to parts `0.083655/0.094118 m`;
- L7: single `0.081039 m` to parts `0.058616/0.054754 m`.

The accepted simulator JPG and metadata SHA-256 values are
`8924d99709b1704d928082d81b1d92329e905887d69d6ba1587e031a3e2c0be5` and
`23d29a4b27b0e8a37a74187116f983691cd51706265bffd85852634c579db73b`.
This result validates geometry only. The v4 bounds have not yet been promoted
into an active safety filter, so they do not alter the completed Table 1
artifacts and do not establish task or collision efficacy. The exact next
audit command is:
`jq '.source,.allocation,.released_aegis_end_effector_proxy_unchanged,.ellipsoids[]|{body_name,semiaxes_m,bound_source,enclosure_certificate}' /mnt/data/quanth/experiments/vlsa-distal-partitioned-ellipsoid-geometry/slabbed-l5-l7-ee-20260808c/visualization/visualization.json`.

## Preregistered E05 learned repulsive-force direction gate

The next bounded mechanism test starts from clean oracle-harness commit
`ced71857d3ede63f990755ba0427890ad69b2e87`, leaving the later execution-model
experiments and completed Table 1 artifacts untouched. It asks whether a tiny
monotone MLP can learn useful weights for physical L5--L7 push-away directions.
Paired cloned-OSC rollouts provide seven two-action clearance derivatives;
the MLP cannot invent their signs and is trained on quantitative ellipsoid
margins rather than contact labels.

Whole state groups are frozen before launch: E05 steps 182--183 train, step
184 validates, and step 185 is untouched test. At test, learned, fixed
analytical soft-min, and 256 antithetic random directions receive identical
normalized-action norms at radii 0.10 and 0.25. The learned arm must improve
exact two-action clearance by at least 0.5 mm, beat fixed repulsion by at
least 0.1 mm, and beat the matched-random distribution at add-one `p<=0.05`
for both radii. No QP, stop, task-success claim, or online deployability claim
is allowed. A two-action execute-and-replan continuation runs only after the
direction gate passes and must retain at least half the nominal end-effector
progress.

The workstation structural gate is clean except for three pre-existing
NumPy-dependent SITL tests, because the default local Python has no NumPy;
the allocation job reruns those tests under the pinned evaluation Python
before simulation. H100 attempt `38782` passed all 39 allocation numerical
tests, then stopped before the first candidate because the new evaluator built
the primary camera at 32 pixels rather than the archived Table-1 resolution;
the required initial-observation hash therefore differed. It is retained as
an apparatus failure with no scientific result. The compatibility repair uses
the archived render resolution for provenance, then disables images after the
pairing hash is accepted; no split, sample, model, comparator, or gate changes.
H100 retry `38783` passed pairing and completed candidate generation plus
training, but failed before atomic output because the model receipt retained
one NumPy input-scale array that JSON cannot serialize. It also has no
scientific result. The receipt-only repair converts that array to a plain
list; it changes no rollout, model, direction, comparator, or threshold. The
exact next command is to commit the serializer and independent validator,
sync the clean commit, and resubmit with a new immutable run ID.

Clean H100 job `38789` completed on `worker-2` from commit `4b4765b` in
83 seconds and wrote the first scientific result. The frozen learned force
beat all 256 equal-norm random directions at both radii (`p=1/257`), but was
almost identical to the fixed physical soft-min direction (cosine `0.997199`)
and slightly worse: learned-minus-fixed gain was `-0.00367 mm` at radius 0.10
and `-0.00651 mm` at radius 0.25. Neither learned radius supplied an exact
two-action safe candidate. The preregistered direction gate therefore failed,
the conditional continuation correctly did not execute, and
`primary_problem_solved=false`.

Independent validation attempt `38790` stopped before issuing a receipt
because its H100 provenance check looked for legacy `allocation.gpu` rather
than the producer's recorded `allocation.device.name`. This is validator
apparatus failure only. The receipt-only repair changes no producer result or
verdict. The exact next command is to commit that key repair and rerun only
the immutable-result validator.

Independent H100 validator job `38791` completed on `worker-2` and reproduced
the strict NO-GO. The nominal two-action hard margin at step 185 was
`-9.313104 mm`. At equal-norm radius 0.10, learned/fixed gains were
`1.003028/1.006700 mm`; at radius 0.25 they were
`2.509951/2.516463 mm`. Learned-minus-fixed was negative at both radii, no
learned/fixed/random candidate was exactly safe, and the two directions had
cosine `0.997199`. The learned field did beat all 256 random directions
(`p=1/257`), proving that the controller-pulled physical gradient is useful,
but learning its monotone row weights added no value over analytical soft-min
repulsion and could not create missing local safe support.

The conditional two-action continuation correctly did not execute. Result and
validation SHA-256 values are
`7ddaf40e25ce7a31e7687229bc86890583d87b604cd615c58cfdadb34a6ce0a2`
and `7f64d148dd3861d94e196be063ec73ec654381f64de5c9f2024bee58604ae38a`.
The exact next audit command is:
`jq '{direction_gate_pass,interpretation,model:{train_rmse_m:.model.train_rmse_m,validation_rmse_m:.model.validation_rmse_m},test:{base:.test.basis.base,radii:[.test.radii[]|{radius_action,learned_exact_gain_m,fixed_exact_gain_m,gain_over_fixed_m,matched_random_p_value,learned_safe:.learned.exact_safe}]},continuation}' /mnt/data/quanth/experiments/vlsa-distal-repulsive-force-direction-e05/repulsive-direction-20260812c/result.json`.

## Preregistered early fixed repulsion inside pi0.5 denoising

The next single-case mechanism test uses the validated analytical seven-row
soft-min repulsive direction, not the failed learned weighting. The immutable
Table-1 action ledger is replayed through step 179 and pi0.5 is queried at
step 180 with the registered policy seed. Because the released controller
executes five actions per query, action steps 182--184 are chunk slots 2--4.
Exactly those slots receive `0.05` physical action units after each of final
Euler updates 5--9, for a total per-slot budget of `0.25`.

Ordinary pi0.5, equal-budget post-hoc repulsion, and inside-denoising
repulsion share the same state, observation, RNG seed, and five-action
horizon. A 31-rollout cloned-OSC finite-difference probe supplies the fixed
seven-row L5--L7 direction. The inside-denoising arm may execute only if a
fresh exact clone proves a nonnegative five-action margin strictly better
than both comparisons. Execution must match the clone, remain free of raw
L5--L7 contact and paper CAR, and retain at least half the ordinary prefix's
end-effector progress. This is not a learned, population, or task-completion
claim. The exact next command is to commit this preregistered harness, sync
the clean commit, run live Slurm preflight, and submit
`slurm/fixed_repulsion_flow_e05.sbatch` on one H100.

H100 attempt `38803` reached `worker-2` and passed all 36 evaluation tests,
then stopped before policy startup or simulation because the OpenPI pytest
repository hook imports optional `pynvml`, which is absent from the pinned
OpenPI environment. The immutable run is apparatus failure only. The repair
replaces that unsuitable pytest invocation with allocation-side bytecode
compilation of the three modified OpenPI modules; evaluation behavior,
comparison arms, state, seed, direction, budget, geometry, and gates are
unchanged.

H100 retry `38806` passed both allocation preflights, loaded pi0.5, replayed
the immutable prefix, and stopped before candidate rollouts because the live
query-36 chunk exceeded the `0.005` raw-action tolerance. That tolerance was
calibrated only for the initial split-JIT query in job `37054`; no evidence
supports extrapolating it across 36 replans. The attempt is apparatus failure,
not a scientific result. The revised harness retains the late live-versus-
archived discrepancy as a diagnostic without claiming equivalence. The three
scientific arms remain exactly paired to each other by the same replayed
step-180 state, live observation, RNG seed, and five-action horizon.

Clean H100 job `38808` completed on `worker-2` from commit `b56854e` in 79
seconds. All three exact cloned five-action prefixes were proxy-safe. Fixed
repulsion inside final denoising improved the ordinary minimum by `2.917279
mm` (`51.560573` to `54.477851 mm`) but was `3.451833 mm` worse than the
equal injected-budget post-hoc arm (`57.929685 mm`). The final output
correction norms at slots 2--4 were only `0.0944/0.1112/0.1056`, showing that
subsequent denoiser dynamics canceled much of the requested `0.25` injected
budget. The comparative gate failed, execution correctly did not occur, and
`primary_problem_solved=false`. The stored generic “no safe support” label is
too broad; the immutable numbers instead mean inside-denoising fixed force
failed to outperform post-hoc force. Independent H100 validation is pending.

Independent H100 validator job `38809` completed on `worker-1` and reproduced
all three exact minima, both differences, the failed comparative gate, and
the non-execution verdict. It records the corrected interpretation
`fixed_repulsion_inside_flow_worse_than_posthoc`. Result and validation file
SHA-256 values are `0c4afaa2d15a61d2defd251a9eaea8f2307cfefef18cbd80a6110d635d792419`
and `2e8a0be2ac566a44e0ac17ef49c8f77484907f4aa2020e41e3ba783b4eb3cd22`.
Because the live query-36 chunk differed substantially from the archived
dangerous chunk and every compared prefix was already positive-margin, this
run does not establish avoidance of the original E05 collision.

## Preregistered archived E05 field-mixture oracle ceiling

The next decisive gate returns to the exact state before archived action 185
and requires the immutable two-action minimum to reproduce `-9.313104 mm`
before search. It compares three post-hoc arms over the six XYZ values of
archived actions 185--186: unrestricted deterministic correction directions,
nonnegative mixtures of seven cloned-OSC clearance normals, and the same
normal mixtures plus nominal end-effector progress projected tangent to the
active safety rows. Rotation and gripper are unchanged.

All arms share four exact correction/relinearization rounds, action bounds
`[-1,1]`, maximum total correction L2 `1.0`, and matched final norms
`0.25/0.50/0.75/1.0`. Every proposal is judged by a fresh two-action cloned
OSC rollout. Passing requires nonnegative seven-row clearance, no raw L5--L7
contact, no paper CAR from episode start, and at least 50% nominal end-
effector progress. The unrestricted arm tests whether any bounded repair
exists; either structured arm must pass before learning field weights is
justified. No MLP, QP, flow guidance, execution, or task-completion claim is
included. The exact next command is to commit the harness, sync the clean
commit, perform live Slurm preflight, and submit
`slurm/distal_field_mixture_oracle_e05.sbatch` on one H100.

H100 attempt `38832` reached `worker-2` and stopped in allocation preflight
before simulator construction. The task-tangent numeric test exposed that the
oracle reused a repulsive-direction normalizer intentionally restricted to
three-dimensional XYZ, while this experiment's two-action tangent has six
dimensions. The immutable attempt is apparatus failure only. The repair adds
a strict finite six-dimensional unit normalizer inside the oracle module and
changes no state, action, search, comparator, geometry, or acceptance setting.

Clean H100 job `38834` completed on `worker-2` from producer commit
`c772784c3b83e3882e5a50d7f797a97c2487c13a` in 235 allocation seconds
(`232.397` evaluator seconds). It exactly reproduced the archived dangerous
minimum `-9.313104 mm` and evaluated 5,351 exact two-action cloned-OSC
rollouts. All three arms found a nonnegative, raw-contact-free, paper-CAR-free
candidate retaining more than 82% of nominal end-effector progress:

- unrestricted: `+0.023973 mm`, progress `0.822360`, correction L2 `0.896320`;
- nonnegative normal mixture: `+0.007236 mm`, progress `0.820680`, correction
  L2 `0.896963`;
- normal plus task tangent: `+0.009566 mm`, progress `0.828249`, correction L2
  `0.891318`.

Matched one-shot nominal-state directions were still unsafe at radii
`0.25/0.50/0.75`; every arm first passed at registered radius `1.0`, where
the normal and normal-plus-tangent minima were `+0.479936/+0.792457 mm`.
Thus, the positive result establishes that iterative physical field mixtures
can express a verified-safe, task-progressing repair of the archived two-action
transition. It does not establish a small correction, execution, complete E05
recovery, population generalization, or learned online steering.

Validator attempts `38835` and `38836` stopped on receipt-only assumptions
about sorted JSON arm order and bounds on untouched rotation/gripper channels;
they produced no validation artifact and did not alter the immutable producer
result. Independent H100 validator `38838` then recomputed every exposed
candidate flag, correction/action identity, seven-row minimum, raw-contact
condition, progress condition, matched norm, and top-level gate. It validated
`structured_field_oracle_supported`. Producer and validation file SHA-256
values are `ff160be3589e76e185582fddc856e40e28ac80c2abe93ac84c16d4cd54674395`
and `70cecda1b5c1bc45587052f68bf131cfd51d9f9320404805c808b5c670746958`.
Table 1 remained read-only at its frozen SHA-256. The next scientific action
is to preregister a multi-state field-weight learner against this exact oracle,
with a positive clearance buffer and matched analytical/random baselines; do
not insert the field into pi0.5 denoising or claim closed-loop task success yet.

## Preregistered late-ramped fixed-repulsion timing ablation

The next narrow gate revisits the live step-180 pi0.5 query solely to measure
denoiser cancellation. Ordinary pi0.5 and three fixed analytical repulsion
schedules share the exact state, observation, query seed, physical direction,
guided slots 2--4, and injected per-slot XYZ budget `0.25`. The schedules are
uniform over final Euler updates 5--9, linearly ramped over only updates 8--9,
and concentrated entirely at update 9.

The registered fairness metric is the L2 norm that survives in all nine final
output XYZ coordinates, not the injected budget. Each flow result receives a
post-hoc comparator in the same physical direction, solved after clipping to
that exact surviving norm. Exact cloned-OSC L5--L7 clearance and ordinary-axis
end-effector progress are reported. The timing hypothesis passes only if both
late schedules retain more output correction than the uniform-final-five arm.
No arm executes: this live prefix was already proxy-safe and does not reproduce
the archived dangerous policy mode. The experiment is not collision-prevention,
task-completion, learning, or population evidence. The exact next command is
to commit and sync the preregistered harness, run live Slurm preflight, and
submit `slurm/fixed_repulsion_flow_e05.sbatch` with
`EXPERIMENT_CONFIG=configs/vlsa_late_ramped_repulsion_flow_e05.v1.json` on one
H100.

Clean H100 producer job `38839` completed on `worker-2` from commit
`6578b4137d906b4012ca3aedc59c7305ea1b4496` in 76 allocation seconds
(`57.294` evaluator seconds). The registered timing gate passed. With the same
unclipped injected chunk correction L2 `sqrt(3)*0.25 = 0.433013`, the correction
surviving in final output XYZ was:

- uniform updates 5--9: `0.177598` (`41.01%`), direction cosine `0.923638`;
- late-linear updates 8--9: `0.341439` (`78.85%`), cosine `0.980229`;
- final update 9 only: `0.420494` (`97.11%`), cosine `0.987790`.

Thus, concentrating repulsion near the final Euler update materially reduces
denoiser cancellation. Exact cloned-OSC minima were already positive for the
ordinary live prefix (`54.934686 mm`) and increased to `58.220835/60.217061/
60.886436 mm` for uniform/late/final timing. Ordinary-axis task progress was
`0.936215/0.908824/0.898034`.

At exactly matched surviving output norms, post-hoc minima were `57.489534/
59.897178/61.114679 mm`. The corresponding inside-flow minus post-hoc
differences were `+0.731301/+0.319882/-0.228243 mm`: the final-step arm is
effectively post-hoc and does not outperform it. No arm executed, and
`primary_problem_solved=false`, because this live prefix was already safe and
did not reproduce the archived dangerous action.

Independent H100 validator job `38840` recomputed all rollout minima,
surviving norms, exact norm matching, schedules, timing gate, and non-execution
verdict. Producer and validation file SHA-256 values are
`d6a0e681ca22db2c89cc6041bb1b70e157fb2b5163ae6dc7f2ca9eb9ca6d4d60` and
`477bbe063eed157ac5085270276396c379e4cd9a0f58b3a8a3f00dfb6ead1ae1`.
The accepted conclusion is narrow: late scheduling fixes the cancellation
mechanism, while post-hoc remains the cleanest maximum-survival baseline. It
does not yet demonstrate prevention of the archived E05 collision.

## Preregistered paired long-horizon ellipsoid-field mechanism gate

The next experiment isolates whether offline counterfactual controller
rollouts teach a useful physical avoidance direction without first adding an
MLP, QP, Poisson field, or VLA-flow modification. It replays immutable Table-1
actions through step 181, then uses the already validated task-rejoining run
`five-detour-e05-20260812b` as a fixed twenty-action continuation for steps
182--201. That continuation is authoritative only for this local mechanism
test and must reproduce its known L5 contact at action 197.

Only the XYZ components of actions 182--186 are perturbed. Every direction is
smooth, has unit L2 norm, is feasible with both signs at registered radius
`0.1`, and preserves the five-action endpoint by summing to zero over time.
Sixty-four paired `+epsilon/-epsilon` cloned-OSC rollouts evaluate the same
fixed continuation; 48 directional derivatives fit one local 15-dimensional
utility field and 16 remain untouched for sign/correlation checks. Utility is
the seven-row long-horizon ellipsoid soft minimum minus terminal EEF and action
deviation penalties.

At equal final correction norm `0.1`, the learned direction is compared with
the current short-horizon analytical ellipsoid repulsion, its negative, and 32
matched random directions. Passing requires held-out Pearson at least `0.5`,
directional sign accuracy at least `0.65`, exact nonnegative L5--L7 clearance,
zero raw protected contact, paper CAR pass, terminal EEF error at most 15 mm,
strictly higher utility than the analytical field, and add-one random p-value
at most `0.05`. No action executes and no task-completion or generalization
claim is permitted. The exact next command after a clean commit and live Slurm
preflight is `sbatch slurm/distal_counterfactual_field_e05.sbatch` on one H100.

Clean H100 producer job `38929` completed on `worker-2` from commit
`ae5d4e8cfd71bbd0f1591969e0b5af164db2b3c2` in 102 allocation seconds and
evaluated 197 deterministic cloned-OSC rollouts. The fixed continuation exactly
reproduced the later action-197 protected contact, with hard ellipsoid minimum
`-14.263205 mm` and paper CAR displacement `1.013437 mm`.

The fitted counterfactual field passed its held-out directional-prediction
gate: Pearson `0.997344`, R2 `0.993467`, sign accuracy `93.75%`, and derivative
RMSE `0.081023 mm/action`. At matched correction L2 `0.1`, its positive
direction improved the hard long-horizon minimum by `0.248367 mm`, incurred
only `0.695685 mm` terminal EEF error, passed paper CAR, and improved utility
by `0.230205 mm`. The short-horizon analytical ellipsoid field instead reduced
the hard minimum by `0.046285 mm` and failed paper CAR. The learned direction
beat all 32 matched random directions on utility (add-one `p=1/33=0.030303`)
and exceeded the analytical field by `0.288702 mm` utility.

The safety mechanism gate nevertheless failed. The learned candidate still
had hard clearance `-14.014838 mm` and raw L5 contact at action 197. No action
among all 128 paired branches or 32 matched random candidates was exactly
safe; the best paired hard minimum was still `-14.057551 mm`. Thus the local
counterfactual labels contain useful later-horizon steering information, but
the registered small endpoint-preserving five-action neighborhood has no safe
support and cannot validate collision prevention.

Independent H100 validator job `38932` recomputed pairing, correction norms,
held-out gates, exact safety, task preservation, analytical/random comparison,
and the strict NO-GO. Producer result/payload SHA-256 values are
`ec686d928214d2f7c5652e8bdec092e8861b940e78d9f0c1021ee7c39a117aa3` and
`d7cf3cb874e50370b4702293a9cab40822fcdbdd8c9a25f2510dc5fa6de4e313`;
validation file/payload SHA-256 values are
`4d544db234a8b3c4ed7f1cfaf0e57e277de55c6deec007677f0ca98f060f4a05` and
`958f9eb992dcd471cea7499672a48c8bccd11fdf07985586d52b0832afde099a`.
Do not train an MLP from this state yet. The next gate must first establish
safe support using a larger or iterative endpoint-preserving field correction,
an earlier intervention state, or a longer corrected horizon while retaining
the same fixed continuation and exact final verification.

## Preregistered decisive pure-risk/constraint/radius diagnostic

The next gate keeps the same action-182 state, five corrected XYZ actions, and
immutable twenty-action continuation through the known action-197 collision.
It first refits job `38929` without simulation using the paper-inspired pure
finite-horizon action risk `V_H=-min h`, with no task penalty or positive-part
clipping. It also audits the short-horizon analytical ellipsoid direction under
both five- and twenty-action hard risk and records active link/time witness
switches. This separates a sign defect from short-horizon myopia.

The simulation stage evaluates total correction radii `0.1/0.25/0.5/1.0`.
The learned arm estimates a fresh pure-risk direction from 32 paired
`+0.05/-0.05` cloned-OSC rollouts at every accepted point, uses inner steps
at most `0.1` with exact backtracking, and accepts a step only when a fresh
twenty-action rollout improves the exact hard margin. The analytical arm is
relinearized under the same iterative schedule. A separate derivative-free
oracle uses three generations of 32 actions per radius, while 32 matched random
directions provide a non-optimized control.

Exact five-action endpoint preservation is the primary arm. The soft-terminal
arm runs only if the primary empirical search finds no safe support. Task terms
never enter the learned risk target; terminal EEF deviation and smoothness are
used only to rank equally safe proposals. All candidates are measured against
zero and +1 mm ellipsoid buffers, raw L5--L7 contact, paper CAR, terminal EEF
error at most 15 mm, correction norm, rollout count, runtime, and active-witness
switches. Nothing executes and no MLP, QP, policy-flow, task-completion, or
generalization claim is allowed.

The interpretation is preregistered: empirical-oracle success with learned
failure identifies field estimation/optimization; soft-only support identifies
the exact endpoint constraint; failure of both empirical searches identifies
the registered five-action/fixed-tail correction family as insufficient.

Clean H100 producer job `38955` completed on `worker-2` from commit
`dd8cf17ae63ac20e414a30a5a84a469c23aceafa` in 14:24 and evaluated 2,275
deterministic cloned-OSC rollouts. Independent H100 validator `38960`
recomputed the result in one second. The nominal hard margin again reproduced
`-14.263205 mm`, with the active second L5 slab at continuation action 201 and
raw protected contact at action 197.

The pure-risk refit is almost identical to the former task-penalized field:
direction cosine `0.999837` and paired fit RMSE `0.081085 mm/action`. Therefore
the task penalty did not cause the earlier directional failure. The analytical
sign audit also found no implementation reversal: its clearance derivative is
`+10.520606 mm/action` over the first five actions but `-1.928330 mm/action`
over all twenty. Fixed repulsion is locally correct and long-horizon myopic.

Exact endpoint preservation remained unsafe for every method and radius. The
iterative pure-risk field stopped after three iterations at `-13.894031 mm`
with correction L2 `0.091695`; all 32 paired branches switched active witness
on every iteration. The analytical arm accepted no long-horizon-improving
step. At radius `1.0`, the best derivative-free candidate reached only
`-7.526865 mm` with terminal EEF error `12.689 mm`; no zero- or one-millimetre
safe candidate existed.

Relaxing endpoint equality exposed the actual tradeoff. Iterative analytical
repulsion achieved positive hard clearance `+16.023852 mm`, zero protected
contact, and paper CAR pass at correction L2 `0.996275`, but terminal EEF error
was `39.795 mm`. The radius-1 derivative-free and random candidates also had
positive margins (`+7.892343/+3.497057 mm`) and zero contact/CAR, but terminal
errors were `28.156/21.752 mm`. None passed the registered 15 mm task-preserving
gate. The learned pure-risk path remained unsafe and crossed the 15 mm task
budget before it could remove the violation.

The scientific result is therefore not “no geometric avoidance exists.” A
large unconstrained five-action correction can avoid the later collision. The
result is that the fixed five-action correction cannot simultaneously preserve
the original twenty-action task endpoint and eliminate the collision under the
registered family. Removing the task term or increasing the radius is not the
missing solution. The next defensible oracle must intervene earlier, correct a
longer action horizon with explicit task rejoining, or allow live receding VLA
replanning after a temporary nonzero endpoint displacement. Do not train the
direct field MLP from this fixed-tail state yet.

Producer result/payload SHA-256 values are
`e5931bad66d3c1782df1488b2ba6aa33dbab8211e54de26bf073dc6006ff88d4` and
`c4af050c1997e76efa9d8886a8f17c6e23ba03b064e3be8fed7a86e7ae1cef96`;
validation file/payload SHA-256 values are
`87fcae0869b442a8398d6c552f533fb2f80a5f0e3fb82f63d94f07d4b4543af7` and
`4b7de1d83e903c07b44a454ea02ec488a61021dbce023bd7617b1fc0ab9df089`.

## Receding persistent-route oracle and route-value diagnostic

The action-187 local route gate passed, but the complete receding test exposed
the missing decision variable. Clean H100 execute-five producer `39048` and
independent validator `39051` selected the smallest immediately safe action-182
route (`right`, correction norm `1.0`, internal minimum `+1.095 mm`). From its
resulting action-187 state, no registered route, smooth refinement, or 256-arm
derivative-free search achieved the `+1 mm` buffer. The run failed closed with
zero contact/CAR and no task success. Result/validation payload SHA-256 values
are `3dca01ce5f1f5d486092066f3c61261e5322b42683c61171cd42b618a6563023`
and `5741def171e0cf99fae7f6191c3ca11e72bdffaad35edf6a7d596839c7475112`.

H100 route-value array `39053` then forced the other five immediately safe
action-182 routes: `right` at norms `1.5/2.0` and `retreat` at norms
`1.0/1.5/2.0`. Every arm executed through action 302 with zero protected
contact, zero paper CAR, and minimum internal clearances
`3.137/7.464/4.010/14.575/16.780 mm`, respectively. None ever satisfied the
native goal predicate; maximum goal fraction remained zero. Independent H100
validation array `39065` accepted all five immutable artifacts. Thus immediate
clearance alone chooses between a later infeasible state and a safe but
task-abandoning state. The route selector needs a continuation/task value;
larger repulsion is not the missing component.

The earlier execute-one mechanism arm `39044` is retained separately. It was
contact/CAR-free through action 299 but task-failed after 118 policy queries;
it queried pi0.5 after every action and therefore does not match the baseline
five-action query schedule. Independent validator `39047` accepted its
artifact, but it cannot support the main execute-five claim.

## Complete compound positive control and robustness audit

To determine whether a safe task-compatible continuation exists at all, clean
H100 producer `39073` replayed the complete task-successful detour trajectory
and applied the registered smooth correction only to actions 182--186. It
completed the native task at action 205, had zero L5--L7 contact and zero paper
CAR, and achieved strictly positive minimum ellipsoid clearance
`+0.167720 mm`. Independent H100 validator `39079` reproduced every trace and
media hash. This is the first complete E05 trajectory in this branch that both
finishes the task and removes the original physical L5--L7 collision. It is an
open-loop compound oracle positive control, not a learned or receding policy,
and it remains a strict NO-GO under the preregistered `+1 mm` buffer. Producer
and validation payload SHA-256 values are
`7e702ab35fd1dee42726694e039d39de1a0cba17ef3c000b4e79b44d4fc09b85`
and `b245f9d7fcdc6b9fb90d42fd3e696c050698cebe1aa42b1d03afdca830f88001`.

The narrow success is not robust to scaling. H100 arrays `39082` and `39096`
tested correction scales `0.98/0.99/1.01/1.02/1.05/1.10/1.20/1.30` with the
identical complete continuation. Scales `0.98/0.99` still completed the task
without contact but reached only `-0.220/-0.022 mm`; every scale above `1.0`
lost task success and produced later contact around actions 232--233, with
minimum clearance between `-15.0` and `-12.1 mm`. Independent H100 arrays
`39090` and `39101` validated all eight arms. Therefore the scale-one result is
a narrow safe corridor, not evidence that a larger repulsive gain supplies a
robust margin.

## Terminal buffer authority audit

The scale-one minimum occurs at the last MuJoCo substep of task-completing
action 205. One-action terminal search `39109` evaluated 261 exact cloned-OSC
actions up to correction radius `1.0`; 200 still satisfied the goal, but none
reached `+1 mm`. All improving actions saturated at `+0.644619 mm` because the
fixed state entering action 205 was already below the buffer. The corrected
two-action search beginning at action 204 (`39115`) evaluated 235 candidates;
74 satisfied the goal, but the best task-preserving margin was only
`+0.667327 mm`, again with zero contact/CAR. Independent H100 validator `39116`
accepted the strict zero-support result; result/validation payload SHA-256
values are `162a33660cf3a4cffa4f9a9c3d1ee0e3e5079e8201aee0019a727bbf62889669`
and `2a46668c16cb50b073ea738bda32d63e7557fcd42223d6f51c0a13f93a5afae7`.

The immediate conclusion is split. Feasibility of collision-free task
completion is demonstrated by an exact open-loop oracle. Robust buffered
feedback control is not: the greedy receding selector is myopic, safety-only
routes abandon the task, and late local repair cannot restore the buffer.
Do not train the proposed policy-value MLP yet. The next oracle must select a
task-compatible route before the terminal corridor narrows, using a fixed,
reproducible continuation policy and route memory.

For apparatus provenance, complete-replay submission `39072` stopped before
simulation on a mistyped expected commit hash. Terminal-repair attempts
`39106` and `39107` stopped before candidate evaluation on replay-binding code
errors. They produced no scientific result and were superseded by clean jobs
`39073`, `39109`, and `39115` without changing the registered gates.

## Compound-scale threshold and live-feedback audit

The earlier statement that every correction scale above one immediately made
the compound unsafe was too coarse. A read-only audit of the validated scale
artifacts shows that `1.01/1.02` remained contact- and CAR-free through action
205, with internal minima `+0.370323/+0.556261 mm`, both higher than the
scale-one `+0.167720 mm`. They missed the native `On(bowl, plate)` predicate
only because the horizontal center distances were `31.145/30.454 mm`, outside
the strict `30 mm` threshold by `1.145/0.454 mm`. The stale archived
continuation then crossed zero clearance at action 206 and did not make raw
protected contact until action 232.

Paired H100 live-feedback array `39141` therefore restored those exact states
through action 205, queried frozen pi0.5 at query index 41 with matched policy
seeds, and evaluated every five-action released-AEGIS chunk through complete
cloned OSC before execution. Neither arm recovered the task. The exact
lookahead issued its first `+1 mm` warning immediately at action 206 and
failed closed before predicted protected contact/CAR: action 231 for scale
`1.01` and action 211 for scale `1.02`. No protected contact or CAR was
executed. Maximum live-versus-clone clearance error was
`4.44e-16/0.0 m`. Independent H100 validator `39143` accepted both immutable
arms; result payload SHA-256 values are
`14ff7b4545f6a1478de9e8f5a66f8d816c3094240650fca25bf2d917132bc862`
and `00af8f1bea99924029d30a2d473345e6175f1e4a4059377fcb4c61e16908ecd6`,
and validation payload SHA-256 is
`0d15633c76d1e2beaab2d4fa8b521dbbe1f766dff860e535fcc965e3e2e9ccdb`.

This supports the robust system's receding future-risk evaluator, but not yet
its learned value or task-compatible correction. Detection alone fails closed
and deadlocks. The next oracle should activate at the first buffered warning,
keep persistent route proposals separate, and rank them by both verified
five-action safety and task-continuation value. Do not wait for raw-contact
prediction and do not train the policy-value MLP from the current incomplete
backup policy.

Producer array `39139` is retained as apparatus evidence only: scale `1.02`
stopped on an over-strict `1e-12` cross-environment live-clearance assertion.
Its surviving scale-`1.01` arm also exposed a reproducibility risk: repeating
the identical simulator state and registered policy seed on separate pi0.5
server instances produced materially different action chunks. Future
policy-value comparisons must freeze returned action tensors or explicitly
model multiple policy samples; recording a seed alone is not sufficient on
this deployment stack.

## Repulsive-field generalization pilot

Commit `6523ec866db6a6d52f140ad8049cea53543ca468` preregistered three
previously unused, outcome-conditioned Table-1 collision cases and compared
paired raw AEGIS, short-horizon analytical repulsion, smooth twenty-action
counterfactual repulsion, and a bounded derivative-free search. All arms
edited only the first five XYZ commands, retained the immutable twenty-action
suffix, and were verified after every one of the 25 MuJoCo model steps inside
each OSC action. No policy server, MLP, QP, task term, or Table-1 artifact
write was used.

H100 array `39149` completed all three producers, and independent H100 job
`39152` recomputed and validated the aggregate:

- goal-II E00, L6/moka: raw `-3.149500 mm`; smooth `+7.171654 mm`, zero
  protected contact/CAR, correction L2 `0.475332`; analytical
  `+21.523158 mm` and exact search `+16.779685 mm` also passed;
- goal-II E24, L6/milk: raw `-26.018758 mm`; smooth `-18.813943 mm`, a
  `+7.204815 mm` gain but still collision; analytical and exact search also
  failed at `-23.076336/-21.707473 mm` within correction norm one;
- spatial-I E12, L5/moka: the intervention state was already
  `-264.041321 mm` under the frozen obstacle-MVEE proxy despite no initial raw
  protected contact. Smooth/analytical/exact search improved the horizon
  minimum by `16.720451/16.921166/16.031580 mm`, but none approached safety.

The aggregate is a strict generalization NO-GO: positive clearance gain
`3/3`, zero-margin collision prevention `1/3`, and +1 mm prevention `1/3`.
The failures are not interchangeable. In E24, smooth repulsion improved the
prefix but did not clear it; analytical repulsion cleared the first five
actions, yet the immutable continuation later re-entered danger. It therefore
needs more prefix authority or a different detour as well as receding
warning-state intervention. In E12, the released obstacle MVEE
makes the ellipsoid proxy deeply negative before raw contact, so proxy-valid
prevention states must be audited before testing avoidance authority. Do not
train or scale an MLP from this gate.

Summary/validation payload SHA-256 values are
`e1c3219ca5245ba153cba1c1613be5021e97767bf8177886e81fdd3e1ce18208`
and `89878a1edcbec9597ada10d82cd67268cb473d6dbd19244c0ee45c54888de217`.

Post-hoc task-validity audit invalidated E24 and spatial E12 as efficacy
denominators without changing their immutable results. In E24, both pi0.5
and AEGIS failed `put_the_bowl_on_the_plate`; the goal was never satisfied,
and at the action-154 intervention the bowl was about `0.301 m` from the AEGIS
end effector and no longer followed it. E24 therefore remains clearance-only
mechanism evidence, not a task-preserving safety failure. Spatial E12 remains
proxy-invalid because its initial ellipsoid margin was `-264 mm` without raw
protected contact.

The focused replacement selects previously unused goal-II E38. Both pi0.5 and
AEGIS completed `put_the_cream_cheese_in_the_bowl`; AEGIS retained the cream
cheese about `4 mm` from the end effector at action 108 with a closing gripper,
then made raw L5/wine-bottle contact at action 113. The new immutable protocol
reuses the exact v1 smooth-field, analytical, and derivative-free algorithms,
budgets, twenty-action horizon, and internal-substep gates. It adds only an
eligibility gate for both-arm native task success, retained task object,
closed gripper, nonnegative initial proxy clearance, and zero initial contact.
No prior artifact is replaced and no population claim is authorized.

Clean H100 producer `39162` completed that focused task-valid experiment from
commit `279bf618688776c52f9f3c7e5c87211a59dda0d1` in 7:40. Independent H100
validator `39166` accepted the immutable artifact. The eligibility gate passed:
both archived policy arms completed the native task, the cream cheese was
`3.561651 mm` from the end effector at action 108, the gripper command was
`+1.000359`, initial protected contact count was zero, and the initial proxy
clearance was `+9.816061 mm`.

Raw AEGIS reached `-52.880546 mm` and 115 protected internal contact samples
over actions 108--127. Smooth counterfactual repulsion improved the ellipsoid
minimum by `+16.792745 mm` to `-36.087801 mm`; analytical repulsion improved it
by `+19.340293 mm` to `-33.540253 mm`; and the bounded derivative-free control
improved it by `+11.898779 mm` to `-40.981767 mm`. All three corrected arms
removed raw protected MuJoCo contact in this twenty-action window and passed
CAR, but none reached even zero ellipsoid clearance. The focused result is
therefore positive physical contact-removal and clearance-direction evidence,
but a strict proxy-buffer NO-GO. It does not yet prove full-episode collision
prevention or preserved task completion: the horizon ends at action 127, while
the archived AEGIS episode completes at action 163 and has another late contact.

The corrected generalization conclusion is consequently not `1/3`. E24 is
task-invalid and E12 is proxy-invalid; they remain diagnostics only. Among the
two eligible prevention cases evaluated so far, E00 passes the full +1 mm gate,
whereas E38 removes physical contact in the registered warning horizon but
fails the ellipsoid gate. This is too small and outcome-conditioned for a rate.
The next focused experiment must extend E38 through completion with receding
warning-state intervention, while reporting hard MuJoCo contact and the
conservative ellipsoid gate separately.

Producer/validation file SHA-256 values are
`199d4fc6fdb25825517be27a8ced324501a28f9bc8e9827505ae34825f8f02b6` and
`6352108aa2369a39ac70a36a0a5b5011f91775b960c39a39b930274b50063753`;
payload SHA-256 values are
`120e036ea0859f62dbc5ef3ba2972dd034e8050380c17f1a1aa09613ceca3822` and
`c8ee5a15ec65db0ef6dbf35ebd69ea11bc0062a46a37dca0059311cdd3d243bb`.
Attempts `39160` and `39161` stopped before scientific rollout on, respectively,
a completed-ledger length assumption and an improper archived MVEE basis. The
latter was canonicalized to determinant +1 by flipping one eigenvector, which
preserves the centered ellipsoid exactly. Validator attempts `39163/39165`
were import-path/expected-hash apparatus failures and changed no result.

The next gate is frozen before execution: starting at E38 action 108, evaluate
the immutable archived AEGIS continuation over a twenty-action horizon. Trigger
the unchanged smooth field whenever its all-substep ellipsoid minimum is below
`+1 mm`, edit only the next five XYZ actions, execute at most five actions, and
recompute from the resulting measured state. Preserve a full terminal
five-action correction window. Stop at native task success, archived-ledger
exhaustion, or a physically unsafe candidate. Report three authorities without
collapsing them: protected MuJoCo contact/CAR, zero/+1 mm ellipsoid
certification, and native task completion. This is an archived-policy receding
oracle, not live VLA feedback, learning, a QP, a CBF, or population evidence.

Clean H100 producer `39181` completed this receding gate on `worker-2` in
16:06 from commit `d4bfe4afa6cd1dfa2fb93be47e16f12f6f5b62d3`; independent
H100 validator `39185` accepted the immutable artifact. Three consecutive
windows at actions `108/113/118` all triggered the ellipsoid warning and the
frozen smooth field. It executed `5/5/3` actions and reached native task
success at action 120, earlier than the archived AEGIS action 163 completion.

The physical and proxy authorities disagree. Across every executed MuJoCo
substep after intervention, protected contact count was zero, no boundary
robot contact occurred, maximum obstacle L1 displacement was only
`0.014614 mm`, and CAR passed. Thus physical collision-free task completion
passes. The conservative ellipsoid proxy nevertheless reached
`-31.577525 mm`, so both zero- and +1 mm certification gates fail. All three
field windows passed held-out direction validation, but no window reached the
ellipsoid target within correction/path norm one. The correct interpretation
is `repeated_repulsion_physical_safe_task_success_but_proxy_uncertified`.

This is the first task-valid cross-case result showing repeated physical
repulsion can remove the targeted contact while retaining native completion.
It is not live-VLA generalization: the continuation is the immutable archived
AEGIS ledger. It also shows that training or controlling against the current
ellipsoid sign alone would remain permanently active even after physical
contact has been removed. The next scientific decision is a proxy/calibration
audit on E38, followed by a matched live-VLA receding test only if the physical
and conservative authorities are explicitly separated.

Producer/validation file SHA-256 values are
`50a28bc8cbb09fbc15fefc113acfa968e139463227d2f18113e18173ce7165fa` and
`de8bb0ba08f44270f1b19aa549fd65673f44519b1e38a3ef5a04de3983bda9a6`;
payload SHA-256 values are
`51ead24d0dc6b84e4c8b6ad8b33e318179d3e19ce4a205fdd62ea5b0c2f591e4` and
`43d1356ae7ec1b938d6f2df2a6509c687fc5dd202bedcf69f26d220dfcf48d1b`.
The video SHA-256 is
`38300408eafac72b3c9100414527961df45dcf008bc63e165af6c2a69b6dfa55`.
Attempts `39177/39178` stopped respectively before rollout on a config-key
typo and at terminal completion on a five-step clone-state comparison. They
are apparatus failures only and produced no scientific result.

The matched obstacle-representation audit is now complete. H100 producer
`39224` replayed the identical successful E38 action ledger from action 108
through native completion, measuring the fixed seven L5--L7 robot ellipsoids
at the pre-action state and all 25 MuJoCo substeps of each of 13 actions. H100
validator `39225` independently accepted all 326 samples. No controller,
action, warning rule, or repulsion output changed.

The released perceived single MVEE reproduced `-31.577525 mm`. Replacing it
with a privileged single MVEE fitted to every vertex of all 21 compiled
wine-bottle collision boxes improved the report to `-20.285566 mm`, but did
not remove the false negative. In contrast, exact solid intersection of each
fixed robot ellipsoid with the union of the 21 compiled boxes found zero
overlap samples, agreeing with zero raw protected MuJoCo contact samples. The
closest exact box/ellipsoid pair retained positive dimensionless radial slack
`0.093303`; this is not claimed as metric clearance.

The perceived MVEE is itself poor: its volume is `8.139691` times the
compiled-box-vertex MVEE volume, its center is displaced `128.831335 mm`, and
it does not contain every compiled collision-box vertex. However, ground-truth
geometry inside one enclosing ellipsoid still reports overlap because the
single convex envelope contains empty space and the center-line support gap is
conservative. Therefore the E38 proxy problem is not perception alone. It is
the composition of inaccurate perception and a single-obstacle-ellipsoid
representation. A deployable follow-up should use multiple tight obstacle
primitives or a validated field, not merely refit one ground-truth MVEE.

Result/validation file SHA-256 values are
`81fe02dd3beaf1eb30d18db306ea4975c0a191226d82401f40b91a6024e95ef6` and
`51ac65dc7ae29158d7bddfe3c4f2f1e8f9dd0d53786610d1b30f8225e15a81e4`;
payload SHA-256 values are
`866db4258e52397aa26bc983fbf154ddd23173e9d0328e4c35da0c9289bf8e5f` and
`63afef1ba2a528a40a8bf149e904e0f2015cf323ed18006c6bee9dfbba0c5918`.
Jobs `39220--39223` stopped before scientific replay on runtime-call, commit,
or MVEE-tolerance apparatus errors and support no inference.

## Exact fixed-backup policy-value pilot

The next gate preserves the validated job-39384 E05 backup and adds only the
missing PNCBF evaluation object: seven exact finite-horizon suffix values at
every receding decision state, plus a registered verified hold/normal-retreat
terminal tail. Positive values mean unsafe, `h_j = 0.001 - clearance_j`. The
pilot requires zero Bellman residual, all decision-state values nonpositive,
zero protected contact, CAR pass, and a buffer-safe terminal tail. It
authorizes no model training, QP, neural-CBF, infinite-horizon,
task-completion, or generalization claim. The preregistration is
`docs/distal_pncbf_policy_value_e05_preregistration.md`.

H100 attempt `39404` passed allocation preflight and stopped before atomic
result output because the shared instrumented-rollout helper retained an old
20-action shape cap while the frozen terminal-tail protocol requires 25
actions. It produced no scientific metric. The apparatus-only repair raises
that cap to 25 without changing the terminal policy or any executed action.

H100 retry `39406` reached exact policy-value construction but stopped before
atomic output because the JSON-oriented trace parser rejected the producer's
NumPy trace object. No result or scientific metric was written. The
representation-only repair accepts objects exposing `tolist()` and changes no
trajectory, clearance, value definition, or acceptance threshold.

H100 retry `39408` reached the exact terminal-successor check but stopped
before atomic output because the strict checker raised on a boundary mismatch
instead of retaining the failed gate. No complete result was written. The
fail-recording repair preserves the `1e-12` threshold, reports the measured
error, and marks successor consistency false rather than discarding a NO-GO.

Clean H100 producer `39410` wrote the first complete exact-value result. All
22 decision states had nonpositive seven-row suffix maxima, Bellman residual
was zero, the 25-action zero-motion terminal hold remained buffer-safe at
+1.205807 mm, and the 288-action ledger retained zero protected contact/CAR.
The gate is nevertheless NO-GO: successor-boundary error was 0.255712 mm,
above `1e-12`. Validator `39411` independently confirmed that stored cloned
candidate traces differ from the executed primary-policy traces and therefore
issued no validation artifact. Exact policy values must be labeled from the
executed policy trajectory; cloned OSC remains the candidate verifier only.

H100 attempt `39419` completed the primary executed-trace episode but stopped
before atomic output because the terminal-tail restore omitted the base
environment clock (`timestep`, `cur_time`, `done`). The dynamic-state equality
gate caught this apparatus omission. The retry adds those already-defined
state variables to snapshot/restore and changes no policy or safety metric.

Clean producer `39422` passed the corrected executed-policy value gate with
22/22 safe decision states, zero Bellman residual, zero successor-boundary
error, a +1.205807 mm terminal hold, and zero protected contact/CAR. Validator
`39423` reproduced every primary clearance trace within `1e-12` but then
rejected derived records using byte-exact floating-point equality. It issued
no receipt. The validator-only repair retains exact structure/booleans and
uses the already registered `1e-12` tolerance for numeric leaves.

Final H100 producer `39422` and independent validator `39424` passed. The
fixed augmented backup generated 22 exact seven-row finite-horizon decision
values; all 22 were nonpositive, Bellman residual and successor-boundary error
were exactly zero, and the registered 25-action zero-motion hold retained
+1.205807 mm minimum clearance. Fresh replay reproduced all 288 actions with
+1.037969 mm minimum executed L5--L7 proxy clearance, zero protected contact,
and paper CAR pass. Maximum producer/validator trace and value discrepancies
were `9.853229e-16 m` and `7.632783e-16`. The frozen proposal ledger still did
not complete the native task. This is a finite-horizon policy-evaluation pass,
not task success, invariance, a neural CBF, or generalization. Producer and
validation payload SHA-256 values are
`89b8e11f81dd7c39a25c48dc78cc46fff6347292dc4a59aa8ae4d490e9ceaff7`
and `36170db254c1d495eafe2b3ab2d78fb869c670490dd1801af7098bb2d5261978`.
## Pure ledger-independent backup audit (active)

The uploaded system specification exposes one material mismatch with the
validated jobs 39422/39424. Those jobs evaluated a deterministic policy on an
augmented state containing the immutable pi0.5 proposal ledger. This is valid
finite-horizon policy evaluation, but it is not the stricter pure backup
required before collecting reusable policy-value data.

The active gate reconstructs only three representative physical/controller
states (steps 187, 222 and 288) from the immutable validated ledger. The backup
itself cannot read the VLA chunk or ledger. It evaluates hold, positive and
negative world axes, and positive and negative local normal/tangent actions at
two amplitudes, executes one action, and verifies a 25-action hold at every
internal substep. Pass requires deterministic replay, candidate-order
invariance, no source-state mutation, +1 mm seven-row support, zero protected
contact, and paper CAR at all three states. This is a focused mechanism audit,
not complete policy, learning, QP, CBF, invariance, task-completion, or
generalization evidence.

The validated clean video from job 39393 ends with the bowl visually near the
plate but not inside the native predicate. The final horizontal bowl/plate
center separation is 35.4439 mm, whereas LIBERO `On` requires less than 30 mm
and contact. Therefore the run is near-complete/task-compatible but not a
native-detector false negative.

H100 producer `39443` and independent validator `39444` passed the focused
pure-backup gate. Fresh replay reproduced every candidate metric with exactly
zero error. At steps 187/222/288 the finite family contained 25/1/12 verified
safe candidates; selected world -X, world +Z, and local +normal candidates
retained minimum 19.887934/2.320242/1.554514 mm proxy clearance over one action
plus a 25-action hold, with zero protected contacts and negligible
`2.274852e-8 mm` CAR displacement. Candidate order, repeated replay, source
state, and ledger-independence gates all passed. This validates the stricter
pure-backup mechanism at three representative states. It does not yet validate
a complete receding backup policy, broad safe set, task completion, learning,
QP, CBF, or invariance. Producer/validation payload SHA-256 values are
`c98812fc3afae7fa8a4dcc6352cef5eb57524cd79e9055738f9603d81b0a3ef3`
and `c87b62873cfc14ed46c5224f81a27b37143fad50aab2c5a9c58e71a8b993ab4f`.

## Grouped real-query action-risk coverage (validated NO-GO)

The query-boundary producer/validator retained one initially safe real
five-action query boundary from each of 15 diagnostic/train/validation episode
groups. Nominal violation coverage reached only rows 1--3; rows 0 and 4--6
remain unsupported. This authorizes candidate-plus-complete-backup coverage
measurement, not training.

H100 canary `39652` reproduced the validated E05 mechanism at action 185: 2
verified-safe candidates, 32 physical unsafe terminals, and 3 censored backup
timeouts. All 37 active witnesses were L5 row 1, and no proxy-safe candidate
had a protected MuJoCo/CAR violation.

The first grouped attempt `39657` preserved two complete artifacts, then cases
2 and 3 hit the 20-minute Slurm wall limit before atomic output. This is an
infrastructure-censored attempt with no result for those cases. The scientific
method was not shortened. Commit `553f001` changes only the allocation limit
to 60 minutes.

Uniform H100 array `39669` completed all 15 states. Independent H100 validator
`39701` checked all result schemas, source/config hashes, archived Table-1
bindings, candidate identities, stored replay hashes, seven-row traces,
contacts/CAR, and deterministic replays. All 15 artifacts and replays passed.
The final state partition is 11 usable mixed-support, one proxy-invalid, and
three with no safe candidate; no state lacked known unsafe support. The
proxy-invalid diagnostic is E10, where 20 candidates with nonpositive proxy
risk nevertheless violate protected contact or CAR. Unknown backup timeouts
remain censored: 183 candidates across 13 states, none included in the learning
manifest.

The frozen coverage snapshot contains 234 known candidates from usable states,
15 state records, and a complete-episode split: 10 train, three validation,
two diagnostic, and three label-free reserved test episodes. Its status is
`coverage_no_go_not_for_training`. Rows 0 and 3 pass every preregistered
coverage threshold. Row 1 lacks one train episode with an active witness; row
2 has no active-witness samples; rows 4 and 6 have no unsafe, near-boundary, or
active samples; row 5 has only two near-boundary samples and no unsafe/active
samples. Therefore only two of seven claimed rows are adequate.

The validated interpretation is
`grouped_query_action_risk_coverage_no_go_targeted_collection_required`.
MLP training, calibration, QP, and closed-loop execution remain forbidden.
Next collection must preserve the exact candidate/backup method while targeting
L5 rows 1--2, L6 row 1, and both L7 rows across complete train/validation
episode groups. Validation file/payload SHA-256 values are
`de33306634feaebe1696a741653026c53d60fa739dc74a6f363c10a64d728bf4`
and `e8371c866bb9083eebb7d2b2514dcc9f6b32ded6d6657c4d7b2d901a322c8a3c`.
Frozen candidate/state/split manifest SHA-256 values are
`380e7e7e3ad82961218778ec150ef962b1a42481d678147f2a360e848d5c77ae`,
`e473a8e585b0ee76c338582e90dbc3c3a44e4f7f9eb25abc20a6c5da714af1f6`,
and `ee1d3286568b178b0f7c225a8ec60f41886b724ce01cbed965c55df1d6659567`.
Validator attempts `39677` and `39698` stopped on coverage-audit numeric and
field-name apparatus bugs and wrote no accepted validation artifact; they do
not alter the final scientific result.

## Finite-set active-boundary identifiability audit

H100 producer `39725` and independent H100 validator `39726` audited all 37
already-executed actions at the 13 train/validation states without training or
new simulation. Unknown timeouts were excluded from extrema, reserved test
episodes remained untouched, and every source/result/hash and per-state row
metric reproduced exactly.

Rows 0--3 all show meaningful action variation and a robust two-sided boundary
with a globally safe negative-side action. This corrects the earlier coverage
diagnosis: rows 1 and 2 are informative despite not always being the globally
active witness. Rows 0/1/2/3 have useful-boundary support in 6/4/4/4 states.
Rows 0, 1 and 3 also have 7/1/5 independent-violation state witnesses. Row 2
has no independent-violation counterexample and is therefore only empirically
dominated on this audited set, not mathematically redundant.

Only rows 4--6 require targeted active search. Row 4 and row 6 vary at every
state but never cross the robust boundary in the finite set. Row 5 crosses in
one no-safe state but never has a globally safe negative-side action in that
same crossing state. No proxy-invalid candidate occurs in the primary
train/validation audit; E10 remains a separate diagnostic proxy failure.
Training and new collection remain unauthorized until the targeted search
determines whether rows 4--6 have deployment-supported useful boundaries.

Producer/validation file SHA-256 values are
`f137a52605955d13ab1235f62cd727564f083508efc4d60eb3d0ae8060efb772`
and `daee504c437f7c7c65c7535e8dd35f0271c3eae838172df0910081f059331b1e`;
validation payload SHA-256 is
`800991933f35a720c0455f8be644957b11c1536f7991ff2d360196f9590c7601`.
Attempts `39723` and `39724` stopped before output on Python-3.8 path API and
mistyped expected-commit apparatus errors and support no scientific inference.

## Restricted three-output L5 action-risk feasibility gate

The user narrowed the learned claim to the three L5 rows (rows 0--2). Rows
3--6 are excluded from the MLP target, but remain mandatory exact ellipsoid,
protected-contact, and paper-CAR vetoes. The abandoned row-4--6 active-search
array `39745` completed only its first two tasks; the remaining exact jobs and
their validators were canceled after exact job inspection when the scope
changed. No accepted row-4--6 search result was produced and no artifact was
deleted.

H100 dataset audit `39759` authorized this restricted prediction experiment on
the immutable grouped data: 140 training and 60 validation candidates, with
E05 retained only as a diagnostic positive control and reserved test episodes
unopened. Each learned row has two-sided boundary support in both grouped
splits. The dataset result file/payload SHA-256 values are
`42c25b44eef2896e85f545b37b387be190ab1951988f2e885e58d8d46968a511`
and `6759587ad0d867515d712fbb9d25f5b4da9cc45f3c71d6559a3b181b9701bbc7`.

H100 training job `39778` completed the fixed 86D, 36,099-parameter,
three-output MLP after 1,200 preregistered epochs. It fit the training set
closely (0.635 mm RMSE, one false-safe, 100% exact-safe recall), but failed
grouped validation badly: 34.587 mm RMSE, 54.062 mm near-boundary RMSE, and 29
L5 false-safe candidates. It recalled 6/8 exact globally safe candidates and
retained a predicted-safe/exact-safe action in all 3/3 recoverable validation
states, but zero false-safes is the mandatory first gate. Rows 3--6 physically
vetoed 30 candidates that appeared L5-safe, confirming that excluded rows
cannot be ignored at acceptance time.

Independent H100 replay `39781` reconstructed the frozen JSON weights and
recomputed every train/validation prediction with exactly zero discrepancy.
It reproduced every metric and the strict interpretation
`three_output_L5_action_risk_prediction_no_go`. The model result file/payload,
model, and validation file/payload SHA-256 values are respectively
`132fc7fe2683aad59c96aa1cfbce71f3fac2c58440a03fc8c1b86791c062dc52`,
`1c3146c5bcf28aee3347f157188f05b58f1e455bfdfccea5665debc3317a3a52`,
`3b337d6fc73b965ca8c10665178c90b540ba61a14c55bf54eb10e3e32a5043b0`,
`b46649b1be66637ec6019efa1968c18f4f017bbaa3dc694d39fd5cede728d6ba`,
and `02f3722a3a0faf4680bc7cc56eb694efc9fbf2a84678f7380669dd42b45b8c3a`.
Jobs `39765`, `39769`, `39773`, and `39774` are retained as pre-result runtime,
test-discovery, source-binding, and deterministic-cuBLAS apparatus failures.

The restricted feasibility test therefore fails at learned prediction, not at
dataset support or the physical veto. QP, calibration, reserved-test opening,
and closed-loop execution remain blocked. The strong train/validation gap
localizes the next question to grouped-state generalization/representation;
training longer cannot establish the missing safety gate.

## L5-only acceptance-rule refinement

The next feasibility claim is now narrower than the validated 39778/39781
analysis: learn only L5 rows 0--2 and combine the final candidate with the
released AEGIS EE constraint. L6/L7 no longer belong to the scoped decision
rule; they remain compulsory diagnostic traces for collision transfer. The
primary reported outcomes will be L5 success and accepted-action L6/L7 contact,
with CAR and native task completion reported separately. No whole-arm or
L5--L7 learned-safety claim is permitted.

A read-only schema audit found that the immutable grouped candidates are
released-AEGIS outputs followed by post-AEGIS structured residuals. They store
seven distal risks, protected contacts, and CAR, but no released EE-QP record
for the modified candidate. Therefore existing artifacts cannot be relabeled
as AEGIS-compatible. Any future control artifact must bind the EE-QP input,
output, status, modification norm, final executed action hash, and final-action
L5 prediction. L6/L7 link-specific contact transfer must remain separately
countable.

This refinement changes no existing result: the current three-output model
still fails its own L5 gate with 29 held-out false-safes, so no candidate is
executed and no new simulation/training is authorized. First resolve the L5
grouped-state prediction failure; only then measure the AEGIS-compatible
acceptance and cross-link-transfer rates.

## Matched L5 state-generalization diagnostic (validated NO-GO)

The attached advisor feedback refines the current root-cause hypothesis from
generic model failure to state coverage plus representation. The decisive
no-new-simulation audit now holds the MLP, labels, seed, optimizer, and action
family fixed while separating new actions at known states from the same new
actions at disjoint episode states. Nominal and constant-profile candidates
from train episodes fit the model; front-loaded candidates form both Test A
(same states) and Test B (validation episodes). Reserved tests, E05 diagnostic
training, AEGIS compatibility claims, asymmetric loss, calibration, QP, and
closed loop remain forbidden.

This audit is deliberately diagnostic. Even a Test-A-pass/Test-B-fail result
does not repair the deployment mismatch: future data must begin with the VLA
action passed through released AEGIS, sample/repair around that output, run the
final AEGIS EE consistency check, and label the exact final executable action.

H100 producer `39809` fit 73 nominal/constant-profile examples and evaluated
67 held-out front-loaded actions across the same seven states (Test A) plus 27
front-loaded actions across three disjoint validation states (Test B). Test A
had one false-safe, 2.883 mm RMSE, 2.392 mm near-boundary RMSE, 100% L5-safe
recall, and support in 7/7 recoverable states. Test B had 14 false-safes,
35.955 mm RMSE, 54.552 mm near-boundary RMSE, 75% recall, and support in 3/3
recoverable states. Every Test-B false-safe occurred at E34; row-level errors
affected rows 1 and 2. Thus both strict gates fail, but the roughly 12.5x RMSE
increase and 14x false-safe increase identify unseen-state transfer as the
dominant degradation.

Independent H100 validator `39810` reconstructed the frozen model and
recomputed every prediction/metric with zero error. Producer file/payload,
model, and validation file/payload SHA-256 values are
`473a35c21f13abf5f70cf83e69e2d5f5fbabb60951c953a1d4f6c43a6eb00152`,
`d2e77a4efb0be9c11ecebd5157c7cb1baf48c3c70f8b47f24af5b6b4ff396fb3`,
`553d24e6c106040edcc6327a481890b671575e82369aa785985c6a6b9cfdedd6`,
`86c7997d32994b9d69c93e2691dfe57b54950c047191ed95ccdb24a71b37bd2b`,
and `e5e58c8f45390f896505efc9aaa7624996053dcf63fb101237350e4c6a8c6f6e`.
No simulator ran, no reserved test label opened, and AEGIS compatibility was
not claimed. The next data collection should prioritize distinct post-AEGIS
states and relative/controller-state features rather than additional actions
at the current states. QP and closed loop remain blocked.
## AEGIS-consistent L5 action-label contract

The first H100-host canary `39819` correctly rejected the initial implementation:
the cloned EE proxy was read from a stale observation cache, producing
nondeterministic actions and state transitions. Canary `39822` restored exact
clone determinism but exposed a frame mismatch: robosuite defines EE position
from the grip site and EE orientation from the robot-model EEF body. Canary
`39825` used that exact mixed-frame definition and showed the remaining issue
was conceptual rather than numerical: released AEGIS is not idempotent because
its virtual direction changes at every QP.

The implementation now transfers the structured L5 residual from the
recomputed post-AEGIS action to the corresponding raw VLA action, then applies
the original AEGIS filter exactly once. H100-host job `39831` passed the action
contract: repeated executed actions, complete controller state, every internal
clearance sample, contacts, and CAR were exactly identical, and a zero L5
residual reproduced the recomputed released-AEGIS baseline with zero error.
Its artifact file/payload SHA-256 values are
`438ffd7eefa3e52beac53a1e2d0ac5a2a4be8e22d4e5e8026cdd036120d2a459`
and `3e877644ac58e3262c25af1cd2d8904663b0bb044d66a320510fc3bd7656194b`.
The two-candidate canary intentionally does not establish mixed support.

Full 37-candidate H100 positive-control job `39832` completed on `worker-2` at
commit `8ffe6ed1325b2df57b4a1a03017f09a4c66e6665` in 381.535 seconds. All ten
producer gates passed. The population contains two exact-safe candidates,
33 contact/CAR failures, and two censored `UNKNOWN_TIMEOUT` candidates. No
proxy-safe candidate physically collided. All 37 active witnesses are L5 row
1, so the result is nonvacuous but not row-complete. Result file/payload
SHA-256 values are
`73aa2734630b57a2768cc56a04e7b4c56e68437c927c2519b18cdaf5af9e3d94`
and `900e3b26a600246e1285c0c17c935b30b3f9ec9f1615e134df31d9819f074387`.

Independent Slurm validator `39846` on `worker-2` passed all eight contract
checks: payload integrity, original AEGIS enabled, exact zero-residual
baseline, every prefix label bound to its final AEGIS output, every backup
action passed through AEGIS, seven-row diagnostics present, mixed support,
and no proxy-safe physical collision. Validation payload SHA-256 is
`ea3eef230284f5e563534a50cf3ff0197ef0d4258e437a12ea1cde2f6d39b603`.
The failed validator launch `39840` and source-prep launch `39844` are cluster
apparatus failures only: `/home` is worker-local and Slurm's `--wrap` used
`/bin/sh`; jobs `39845`/`39846` established the reproducible worker-local
source preparation path.

Training, calibration, QP, reserved-test access, and closed loop remain
blocked. The exact next command is the AEGIS-consistent grouped H100 canary on
one train and one validation episode; it must retain failed/no-support states
and validate the final executed-action bindings before any wider array.

## AEGIS-consistent grouped L5 canary

The first grouped submission `39853` is retained as apparatus failure only:
both tasks reached `worker-0`, where a one-GPU allocation exposed multiple
H100s, and the exact-one-H100 preflight rejected them before simulation or
artifact creation. Replacement producer array `39902` therefore retained the
original `worker-0,worker-3` exclusion and ran sequentially on `worker-2`.
E19 train completed in 35:55 and E22 validation in 48:28 from clean producer
commit `04613c3ee2f5e03b3cf606b3fa089d87401f0793`.

Independent validator attempts `39903` and `39931` exposed grouped-schema-only
apparatus mismatches (`base_method_config` and grouped gate names). Validator
`39935` then exposed that a timeout must not create mixed support. No producer
artifact changed. Final validator commit
`5f31c96ff668eeee4483f44925c98c132b13551a` explicitly separates
`UNKNOWN_TIMEOUT` from known unsafe candidates; final jobs `39939` and `39940`
passed every action-contract check, and summary job `39941` completed.

Both states have genuine mixed support without dropping failures:

- E19 train: 15 exact-safe, 7 known-unsafe, 15 unknown timeouts;
- E22 validation: 4 exact-safe, 9 known-unsafe, 24 unknown timeouts.

Across 74 candidates, L5 row 0 has 19 known-safe, 16 known-unsafe, 15
near-boundary, and 35 active-witness examples across both episodes. Rows 1--6
have 35 known-safe and zero known-unsafe, near-boundary, or active-witness
examples; all 39 timeouts remain censored for every row. Thus the corrected
AEGIS action-label contract transfers across one train and one validation
episode, but the canary identifies only row 0. It does not authorize the
three-output L5 MLP yet. The next gate is broader episode-grouped collection
with the identical contract, targeted toward real controllable boundaries for
L5 rows 1 and 2; rows 3--6 remain diagnostics.

Summary file/payload SHA-256 values are
`4447be21ec3ef3f9626a7427fd1d0d8d5c5715f108b8651612687f9bccead108`
and `5376fcd112ea2c2938eb44879f296515135b817c27295d082c717abd3382cee7`.
E19/E22 validation payload SHA-256 values are
`93c769a573ec862137e99dedb932c4a314c6d8c96c02372cfbde7874def72ffa`
and `771fc58e70423b721e5a3e2c9911f952147fea2da8c6905200c48ef41d471f1c`.
Training, calibration, QP, reserved-test access, and closed loop remain
blocked.

## Adaptive post-AEGIS L5 boundary collection (preregistered)

The exhaustive 37-candidate population launch `39957` was inspected and
cancelled after three completed cases when the collection design changed; its
completed artifacts remain immutable apparatus history and no outcome was
dropped. The replacement prioritizes distinct states over repeated expensive
backup rollouts at one state.

At each of the 15 unsealed clean episode groups, the adaptive collector first
executes nine fixed normal/tangent five-action candidates through the original
AEGIS EE filter and exact OSC. Prefix screening stops after the first
high-level action that records protected contact or CAR failure. If both sides
of the L5 boundary are present, five direct action-space bisection probes refine
the bracket. Only nominal, the final bracket endpoints, and the bisection
samples—at most eight actions—receive the complete fixed-backup rollout that
defines the authoritative seven-row risk label. If no prefix bracket exists,
the state is retained with its nominal label and explicit support failure.

Every artifact records the final post-AEGIS action and EE-QP trace, q/dq,
complete simulator/controller context, EE pose, L5--obstacle relative centers
and normals, all seven proxy rows, raw contacts, CAR, and censored timeouts.
The clean manifest contains only 18 eligible episode groups: 15 development
groups and three sealed test groups. Known E38 proxy mismatch and gripper-
contact cases are not admitted merely to reach 20 groups. Collection cannot
authorize training. Sufficient grouped boundary coverage authorizes only the
matched no-training 86D/compact/complete-state feature audit.

Allocated-H100 canary `39982` and independent validator `39983` passed on
`worker-1`. The nine prefix screens found a genuine L5 row-1 bracket: the
radius-2 outward-normal proposal had `-1.193830 mm` buffered risk while nominal
had `+16.974327 mm`. Five bisection probes reduced the complete-backup set to
six unique actions. Authoritative labels contain one exact-safe and five
known-unsafe candidates, zero timeouts, zero proxy-safe physical collisions,
and one near-boundary safe action at `-0.064121 mm` risk. Every action and
backup decision passed the unchanged original AEGIS EE filter.

The producer completed in `222.088 s` (`3:45` Slurm elapsed), versus `381.535
s` (`6:21`) for the prior 37-candidate E05 positive control: a 41.8% wall-time
reduction while retaining mixed support. Result/validation payload SHA-256
values are `fcc58013d2f136760d1cabb5077778571bce54f6cb8dc720104e3c40149abd98`
and `ff0c7738e70f68f563aa1de2a3dec6dedd781ae088d92cef90bafa2bbc8ad2a2`.
This authorizes the 15-development-state adaptive array and independent
validation. It does not authorize feature fitting, MLP training, QP, test
opening, or closed-loop control.

The population protocol is refined to v2 before scale-up. Twelve coarse
proposals include nominal, paired normal/tangent axes, two temporal profiles,
three correction amplitudes, and three fixed-seed normal--tangent mixtures.
For each state the artifact reports observed negative/positive support for
each L5 row separately. A row is controllable only when the coarse set contains
a physical-veto-free action with all L5 risks below `-0.5 mm` and another
action with that row above `+0.5 mm`. Among controllable rows, the row with the
largest nominal risk becomes the registered target; five complete-prefix
bisection probes refine only that row. All three L5 targets and rows 3--6 are
still recorded by the authoritative retained candidate-plus-backup rollouts.

This v2 change prevents the hard maximum from silently choosing every training
boundary. States without a globally safe endpoint or row-positive endpoint
remain explicit no-support/unrecoverable cases. The passed v1 canary remains
immutable cost-hierarchy evidence.

H100 producer `39986` and independent validator `39987` correctly retained a
per-row support NO-GO on E05. Row 1 ranged from `+23.019862 mm` to only
`-0.009707 mm`, so it did not cross the preregistered `-0.5 mm` safe endpoint;
rows 0 and 2 stayed negative throughout. The collector therefore selected no
target row, ran no bisection, and retained only the unsafe nominal action. This
is valid negative scientific evidence, not an apparatus failure. Result and
validation payload SHA-256 values are
`faffdf6b39b6dcb4472f745b6b53b92a63db8f588a422d0a8384ba3a329d060c`
and `9a49836facb0ed44e5afc44731f9ec636e22bff817ef12529338ff731bd33169`.

The audit localized the failure to one candidate-bank substitution: v2 used a
constant radius-2 normal proposal, whereas the immutable v1 evidence found its
verified-safe endpoint with the front-loaded radius-2 normal proposal
(`-1.193830 mm`). Version 3 restores that known recovery arm while preserving
the 12-candidate budget, fixed mixtures, per-row selection, 0.5 mm thresholds,
bisection, and authoritative complete-backup labels. A v3 E05 H100 canary must
recover the registered row-1 bracket before the 15-development-state array is
launched.

H100 producer `39989` and independent validator `39990` passed that v3 gate on
`worker-1` in `3:43`. The restored front-loaded radius-2 arm supplied a globally
L5-safe row-1 endpoint at `-1.193830 mm`; row 1 also reached `+24.347685 mm` on
the positive side. Row 2 independently exhibited coarse two-sided support, but
row 1 was correctly selected because it had the largest nominal risk. Five
bisections yielded six unique authoritative labels: one exact-safe and five
known-unsafe, zero timeouts, zero proxy-safe physical collisions, and six row-1
active witnesses. All final actions and backup decisions passed the unchanged
original AEGIS EE filter. Result/validation payload SHA-256 values are
`d93c3aecfcab4a3e9b742275f8b7e069a97fd8ef28250aa95853624ceb622474`
and `0c10df85f79df2b6925b2114c60ba18dcd0e04f9551d45d8fd9d9fd8c289e218`.
The 15 unsealed development groups are now authorized under v3. Training,
feature fitting, calibration, QP, closed loop, and sealed-test access remain
blocked until grouped coverage and dataset-freeze gates complete.

Grouped H100 producer array `39993`, independent validator array `39996`, and
summary job `39997` completed all 15 unsealed groups without apparatus failure.
The immutable population contains 80 authoritative candidate-plus-backup
labels and 20 censored timeouts. State classification is five usable mixed-
support, six no-safe-candidate, three unknown-only, and one proxy-invalid. No
state, failed action, or timeout was discarded. Thirteen states had a coarse
per-row bracket; target rows were 0/1/2 in 6/5/2 states respectively.

The result is a strict coverage NO-GO. Candidate-count gates pass for all three
L5 rows on the eligible train+validation labels: rows 0/1/2 have 32/33/26 known
safe, 16/15/22 known unsafe, and 23/20/23 near-boundary candidates. The failure
is independent-state support, not raw action count. Useful train boundary-state
counts are only 2/2/1 versus the registered minimum 3; validation counts are
2/0/0 versus minimum 1. Moreover, the fit population has 25 row-0 active
witnesses but zero row-1 or row-2 active witnesses. Known train candidates are
32 versus the minimum 40. Therefore neither the generalization feature gate nor
deployment-gated MLP training is authorized, and no dataset split is frozen as
learning-ready.

Summary file/payload SHA-256 values are
`1c93d3f3e815c7f7dfa0eed824f145221ba549c6642bda491fd14f1794eaf0d7`
and `5babbf5ec8b2dda200897a35d0fafd75f30f32800eef3ae212d8cadf3655e0a4`.
The only supported next collection is additional grouped physical states that
make L5 rows 1 and 2 active and controllable, plus at least one additional
row-0 train state. Repeating more actions at these 15 states would not repair
the identifiability failure. Sealed test episodes remain unopened.

The learning decision is subsequently narrowed: this population is a GO for a
capacity/implementation diagnostic but remains a NO-GO for a generalizable
safety filter. The preregistered diagnostic fits one small three-output L5 MLP
on 26 known train actions. In each train state with at least two labels, the
largest registered candidate order is held out without consulting its target,
yielding six same-state action-interpolation tests spanning target rows 0/1/2
in counts 3/1/2. Sixteen known validation actions remain episode-grouped; only
the ten actions from the two validation states with actual row-0 boundary
support form the row-0 generalization diagnostic. Rows 1--2 receive only local
interpolation reporting. E05 is an excluded positive control, all timeouts and
the proxy-invalid state remain excluded, and the three sealed test episodes
remain unopened. The run has no deployment pass gate and cannot authorize
calibration, QP, or closed-loop execution.

Initial H100 job `40048` passed its unit preflight but stopped before model
construction because adaptive midpoint records intentionally leave the legacy
top-level correction norm null and bind the authoritative final norm inside
`residual_binding`. This is an apparatus representation failure with no fitted
weights or scientific metric. The retry reads that existing authoritative
field; it changes no sample, action, target, split, feature, model, or gate.

Retry `40054` passed the repaired data preflight and reached the first model
forward pass, but the evaluation Python's legacy PyTorch binary has no `sm_90`
kernel for H100. It stopped before an optimizer step or result artifact;
dependent validator `40055` was cancelled exactly. The runtime-only retry uses
the registered OpenPI Python whose PyTorch supports H100, without changing any
scientific setting.

OpenPI-runtime attempt `40057` then stopped in repository preflight because
module-style unittest loading resolved an unrelated installed `tests` package.
It never imported the training script or constructed a model; dependent
validator `40058` was cancelled exactly. The retry uses repository-local
unittest discovery, identical to the existing validated OpenPI training
apparatus, with no protocol change.

H100 diagnostic producer `40060` and independent replay validator `40061`
completed on `worker-1` at commit
`30bded7078ed701267185f7633839577e9ae1c2e`. The fixed 3,939-parameter
three-output L5 MLP fit 26 actions and was evaluated on six label-independent
same-state action holdouts, 16 episode-grouped validation actions, the ten
row-0 boundary actions from two supported validation states, and the excluded
six-action E05 positive control. The replay validator reproduced every
prediction exactly (`0.0 m` maximum discrepancy).

The capacity mechanism is positive but not deployable. Same-state action
holdout RMSE was `0.231658 mm` overall, with row RMSEs
`0.178035/0.198677/0.299712 mm`; row 1 still had one per-row false-safe and row
2 one false-unsafe. On the only supported grouped claim, row 0 had zero
false-safes, `90%` boundary-side accuracy, `2.750499 mm` near-boundary RMSE,
`75%` exact-safe recall, and predicted-safe support in both recoverable states.
The complete grouped validation remains a strict NO-GO: rows 1 and 2 each
produced six false-safes, and E05's exact-safe candidate was rejected. This
confirms that the compact model can interpolate local actions and has limited
row-0 transfer, while the missing row-1/row-2 state coverage prevents a
generalizable L5 safety-filter claim. Calibration, QP, closed loop, and sealed
test access remain blocked; the next data action is new episode-grouped,
controllable row-1/row-2 boundary states, not more actions or epochs at the
existing states.

Model/result/validation payload SHA-256 values are
`e7d307cb54a96523e4058d8b523a4391b0fb441b99ef862745adbdf3612310ca`,
`3ed16bcbdb445f5fda5f8a854ca5ef3fc7b2b1091ebe61b94f10ea1bbb620342`,
and `6c9691b79d1a9c43a1adccf53389c761f85f01b9e7b68d3d5d95428939028f6a`.

A frozen no-training root-cause audit is preregistered to distinguish model
capacity from state coverage without opening sealed tests. It compares train,
same-state action holdout, grouped validation, supported grouped row 0, and E05
errors; measures nearest fitted support separately for the 16D geometry
context, 35D nominal action, 51D combined state context, 35D candidate
residual, and full 86D input; and binds the existing grouped coverage counts.
The primary data-coverage diagnosis requires sub-millimetre train and
same-state action error, at least a tenfold grouped error increase, zero
row-1/row-2 validation boundary states, and zero row-1/row-2 fit active
witnesses. The audit also records—but does not claim to identify—the omitted
joint/OSC/controller-state hypothesis and the non-conservative symmetric-loss
risk. No retraining, calibration, QP, control, or test access is authorized.

H100 audit `40063` and independent recomputation `40064` completed on
`worker-1` and exactly reproduced the analysis. Train and same-state action
holdout RMSE are `0.186811 mm` and `0.231658 mm`, but complete grouped
validation RMSE is `31.405546 mm`, a `135.569x` increase. Supported grouped
row-0 near-boundary error is `16.696x` its same-state counterpart, and E05
row-1 near-boundary error is `71.677x`. This hierarchy rejects model capacity
as the primary failure and identifies insufficient grouped state plus active
boundary coverage as the primary root cause.

The distribution audit agrees: all six action holdouts share a fitted state
and have zero state-context distance, whereas grouped validation has no fitted
state, reaches `1.536` nearest-state-context RMS z and places up to 23/86 input
dimensions outside the fitted range. E05 is more extreme: state-context and
nominal-action distances are `3.489` and `3.797` RMS z, with 33/86 dimensions
outside the train range. This coincides with useful boundary-state counts
`[2,2,1]` train and `[2,0,0]` validation and active fit witnesses `[25,0,0]`.

The result is not evidence that data is the only issue. The 86D input omits
recorded q/dq, EEF pose, OSC goals/controller memory, obstacle pose, and
per-row transforms; causality requires a matched complete-input ablation after
coverage exists. Symmetric Huber also retains one row-1 false-safe on fitted
data and is not a conservative gate. Audit/validation payload SHA-256 values
are `d25bcca06d7d463887daf0e34e39bf0ac00690b120ed372492296b93302a18c1`
and `40692ccb9932d252d86a06a96f2276573ca14ab026fa3bda227f55f1eadb920a`.

## Targeted L5 boundary-state extension (2026-08-14)

Five previously unused, clean, task-successful task-2 L5-contact episode
groups were preregistered without inspecting their row outcomes. Training uses
E04/E27/E49 and validation uses E14/E39; the sealed E42 test episode and all
completed Table-1 artifacts remain untouched. H100 query-state producers
`40067` and independent validators `40068` found one reproducible, initially
safe real five-action warning boundary in every episode.

H100 adaptive producers `40077`, independent validators `40078`, and aggregate
job `40079` then applied the unchanged v3 candidate bank, per-row 0.5 mm
controllability rule, five complete-prefix bisections, authoritative
candidate-plus-fixed-backup labels, and original AEGIS EE compatibility gate.
The five state classifications are: E27 and E39 usable mixed support; E04 and
E14 no safe candidate; E49 unknown-only. All 12 timeouts remain censored
(`7` train, `5` validation); no state or failed candidate was discarded and no
proxy-safe physical collision was observed.

The registered extra row-0 train requirement passed: E27 adds one independent
useful training boundary state, and E39 adds one validation boundary state.
However, every targetable boundary in this extension was L5 row 0. Cumulative
useful boundary-state counts are now `[3,2,1]` for training and `[3,0,0]` for
validation. Consequently, the row-0 learned-row gate passes, while row-1 and
row-2 train/validation state-support gates remain strict failures. The known
candidate count also remains below the registered training minimum (`38 <
40`).

This exhausts the obvious unused clean task-2 L5-contact cohort and establishes
that repeating its collection cannot identify rows 1--2. It does not establish
that those rows are physically irrelevant. The matched feature ablation,
retraining, calibration, QP, closed loop, and sealed tests remain blocked. The
next experiment requires an explicit population decision: preregister a new
task-3 active-boundary discovery cohort, generate new clean task-3 episodes, or
narrow the first learned claim to row 0. Aggregate `summary.json` SHA-256 is
`d32eac3ffd59976809ed862ac0dfe4f13abc282c1801a00e63345bf8b913a008`.

A read-only row/contact alignment audit is now preregistered before any proxy
change. It binds the independently validated 15-state adaptive-v3 development
population and five-state targeted extension, opens no sealed test episode,
and runs no new simulation. For every unique exact screening prefix and every
authoritative candidate/selected-backup substep, it aligns raw MuJoCo L5--L7
contacts with the simultaneous seven-row ellipsoid trace. Proxy-invalid states
remain a separate diagnostic and observed timeout prefixes remain geometry
evidence without becoming complete rollout labels.

The primary geometry gate is zero physical contact samples whose corresponding
link-row minimum is positive. It also reports conservative proxy-only overlap,
the active row at every physical contact sample, and per-row contact-time
clearance distributions. Ellipsoid refitting is authorized only if a
proxy-valid physical false-safe is observed. Otherwise the correction is to
the population or learned claim, not to introduce Poisson or change geometry.

Initial H100 job `40106` passed its allocation tests and stopped before writing
an audit because two source traces encode a repeated action-boundary contact as
`substep=-1`. The combined clearance trace retains that same state as the
previous action's final substep. The apparatus repair maps this duplicate
identity to that retained sample; it changes no clearance, contact, population,
threshold, or decision rule. Dependent validator `40107` was cancelled exactly.

Final H100 producer `40111` and independent validator `40112` completed the
read-only alignment over 57,275 future substep samples from proxy-valid states;
the validator reproduced the analysis exactly. There are zero physical-contact
false-safes at zero margin and zero physical-contact false-safes at the 1 mm
certification buffer. Therefore the accepted ellipsoid geometry passes this
population audit and neither refitting nor Poisson is authorized.

Physical witness identity resolves the coverage ambiguity. The 376 L5 contact
samples activate row counts `[258,118,0]`: all 118 row-1 witnesses come from
the E05 moka state, while all 258 row-0 witnesses come from six task-2 milk
states. Row 2 is never the active contact witness, although it lies within 1 mm
at 17 contact samples. L6 has 686 physical contact samples, all on row 3, from
four task-3 states; L7 has no contact sample. Conservative proxy-only overlap is
large (13,009 L5 and 5,003 L6 samples) and is reported separately rather than
misclassified as unsafe geometry evidence.

Including the already-separated proxy-invalid state creates 68 apparent
physical false-safes; excluding it restores zero. This localizes that failure
to the invalid obstacle proxy, not the L5--L7 arm ellipsoids. The corrected data
action is therefore new, clean E05-like/task-0 row-1 episode groups. More
task-2 sampling cannot supply row 1. Row 2 remains an exact diagnostic monitor
without a learned-generalization claim until a population physically activates
it. A row-0-only model may remain a mechanism diagnostic, but it cannot solve
the primary E05 row-1 collision. Training, QP, and closed loop remain blocked.

Final audit/validation payload SHA-256 values are
`479f097fe05e1bb18139e75f3f4c3280ffafbeb27e614d766a3061d478dde4d5`
and `b87abcbe8ca62fe9893c9235b9201d548f3dac226313d1b615d6ac807be8ead5`;
file SHA-256 values are
`ba6913ddb74a2bf9246e85dcf49d739962ed06fcbe9f93ef099dcd2590106e24`
and `67a20857667d406cb87c2cd5ca5599b579ee1da39a3054a339d03544c2e6d621`.

## Task-0 moka noise-discovery population (2026-08-14)

The immutable 50-episode task-0 Table-1 audit contains exactly one clean,
task-successful, proxy-valid L5/moka contact case: E05. E10 is L6, while the
other task-0 L5/moka contacts occur only in task-failed episodes. Admitting
those failures would silently weaken the declared deployment cohort, so they
remain excluded. No Table-1 result or sealed episode is modified or opened.

ADR-0127 preregisters ten new development trajectories, one for each task-0
level-II moka initial state. The simulator state, frozen pi0.5, released AEGIS
EE correction, perception, five-action execution, and 300-action horizon stay
unchanged. Only the outcome-blind policy-noise seed changes according to
`2026081400 + 1024 * episode_index`. Each trajectory is its own group and every
failure, timeout, task failure, proxy-invalid result, L6/L7 contact, and
no-contact result is retained.

Only complete native-task-success trajectories with a valid proxy, no settled
or dynamic task/other contact, and a post-control `robot0_link5` moka contact
are eligible for the unchanged controllable-boundary collector. At least two
new distinct initial states are required before that downstream job is
authorized. This is development-only coverage discovery, not learned-filter
evidence. MLP training, calibration, QP, closed loop, Poisson/SDF replacement,
and sealed-test access remain forbidden.

H100 producer array `40129` completed all ten original-AEGIS trajectories with
successful exits. Dependent summary attempt `40130` stopped before reading any
result because its allocation omitted the repository from `PYTHONPATH`;
validator `40131` remained dependency-blocked. This is retained as apparatus
history. The environment-only repair changes no trajectory, manifest,
classification, threshold, or downstream gate and reuses the immutable
producer artifacts.

Corrected summary `40141` and independent H100 validator `40142` bind the
immutable producer commit and validate all ten outcomes. Four trajectories are
clean, task-successful, proxy-valid L5/moka failures: E00, E05, E10, and E45.
Three additional trajectories complete without L5 contact and three fail the
task; none is discarded. The 4/10 result passes the preregistered two-state
discovery gate and authorizes boundary collection, not model training.

E05 and E10 remain diagnostic because those initial states have already shaped
the method. E00 is assigned to development train and E45 to development
validation; policy-noise variants are not misrepresented as new unseen initial
states. A new four-case query-boundary audit is preregistered against the exact
discovery result hashes. It reuses the accepted seven ellipsoid rows, original
AEGIS EE action ledger, real five-action query boundaries, and unchanged
initial-safety/nominal-risk rules. Adaptive labels are allowed only after the
independent boundary audit retains an initially safe warning state. Training,
calibration, QP, closed loop, Poisson, and sealed tests remain blocked.

Discovery summary/validation payload SHA-256 values are
`49143d0f80581a526d47df36ebb381e2739b2b996432d57903cfda5fbe5de99f`
and `1a0add5cb2e15c0ba4a365ccfea2921ad50be100300253ba39580cdfaf02816b`;
file SHA-256 values are
`b97babc6a05d234113891d4dbf64cb47989ddd4a921a22770929a874e502baa0`
and `240ac17200c71389cfa8fc8de959dd052ea48ddcb59b0b4043dbbacc55db0472`.

Initial boundary array `40143` failed in the source-binding preflight before
environment construction because the discovery summary's canonical field is
named `payload_sha256`, not `result_payload_sha256`; dependent validator
`40144` is cancelled exactly. The key-name repair binds the same preregistered
digest and changes no source trajectory, boundary rule, simulation, or gate.

Retry `40148` was rejected by the clean-source gate before environment creation
because the submitted expected full hash was mistyped; validator `40149` is
cancelled exactly. Valid producer `40153` and independent validator `40154`
then completed all four trajectories. Every case has exactly one initially safe
warning query and L5 row 1 is the nominal active witness in all four. Three of
four nominal prefixes cross row 1's 1 mm boundary; L6 row 3 is additionally
violated in those same three prefixes, but it is never the active witness.

This supplies one new row-1 train state (E00) and one new row-1 validation state
(E45), plus diagnostic E05/E10 positive controls. It does not supply row 2.
The independently validated result authorizes unchanged adaptive-v3 boundary
candidate collection on these four exact warning states. The candidate runner
now accepts the already-bound policy-noise seed from the source trajectory so
the cloned rollout restores the same ledger; ordinary Table-1 callers remain
unchanged. Training and every downstream control gate remain blocked until the
adaptive labels are complete and independently validated.

Boundary validation payload SHA-256 is
`994fd0921142a38f725b744dca918a89afded410e661aa5b74564302c3b459bf`;
validation file SHA-256 is
`77b88ff82d281ca8c8d47a325fa0f575b2e761506fd7256137554edcdbd17975`.

Adaptive-v3 producer `40158` and independent H100 validator `40159` completed
all four warning states with no timeout. E00 train and E45 validation each have
two-sided row-1 support: after L5-scoped censoring E00 contributes three safe
and two unsafe complete candidates, while E45 contributes two safe and four
unsafe candidates. Diagnostic E05/E10 have no globally safe candidate in the
frozen bank and remain explicit no-safe states.

The broad all-protected summary `40168` initially classified E00 as proxy
invalid. The exact cause is one nominal-prefix `robot0_link6_collision` contact
at action 164/substep 24 while L6 rows remain positive (row-3 minimum 7.417 mm).
This is a real L6 proxy miss exposed by the new trajectory, but it is outside
the declared learned-L5 acceptance rule. It is not silently called safe: the
early-stopped candidate is censored/unknown for L5 because it lacks a complete
backup rollout, and the L6 contact is retained as a cross-link diagnostic.
Rows 0--2 have no observed proxy-safe L5 physical contact in this extension.

H100 summary `40172` applies exactly that declared L5 scope. Cumulative known
candidate counts become 43 train and 29 validation. Useful boundary-state
counts become `[3,3,1]` train and `[3,1,0]` validation for L5 rows 0/1/2.
Consequently rows 0 and 1 pass every registered sample and state-support gate,
while row 2 still fails both train and validation state support. A matched
row-0/row-1 feature audit is now scientifically supported with row 2 retained
as an exact diagnostic monitor. The requested three-output L5 MLP is not yet
authorized; training, calibration, QP, closed loop, and sealed tests remain
blocked rather than manufacturing row-2 data.

Scoped summary payload/file SHA-256 values are
`9f3a98f96ff7c9995bb04aaa1e9f2b686f7e9f99ceaf5380a4ab0de4ef588c03`
and `3b873cef172e37a1e6d24827860ad54c4b20a248168302cb205f22c93be0e342`.

The requested learned-correction feasibility test is preregistered as a strict
two-output L5 row-0/row-1 candidate selector. It binds the three immutable
summary payloads, 43 train and 29 disjoint validation labels, useful-state
counts `[3,3]` train and `[3,1]` validation, the existing compact 86D input,
and the fixed 32--32 MLP. Prediction determines only which registered candidate
to propose; exact all-seven ellipsoid/contact/CAR checks and the original AEGIS
EE projection remain vetoes. Row 2, L6, and L7 are diagnostic only and cannot
support a learned-safety claim. The passing gate requires zero selected
row-0/row-1 false-safes and safe support in every one of the four recoverable
validation states before fresh replay. QP, calibration, closed loop, and sealed
tests remain blocked.

Initial H100 producer `40173` passed allocation tests and stopped before model
construction because one immutable candidate stores a null top-level applied
correction norm while retaining the numeric norm in its validated residual
binding. Dependent validator `40174` is cancelled exactly. The loader-only
repair uses the residual value when the optional top-level alias is null; it
changes no sample, feature, target, split, model, seed, selection, or gate.

H100 producer `40175` and independent validator `40176` completed the fixed
two-output selection experiment on worker-2. The 3,906-parameter model retained the exact
43/29 train/validation populations and validator replay error is exactly zero.
Training RMSE is `0.164472 mm`, while disjoint grouped-validation RMSE is
`14.716616 mm` with row RMSEs `17.939185/10.551929 mm`; 16 of 29 validation
actions are dangerous row-0/row-1 false-safes.

The action bank is not the blocker: all four recoverable states contain two
exact-safe candidates. The MLP supports all four only in the weak sense that it
predicts something safe, but its closest predicted-safe action is exact-safe
only for E22. E37, E39, and E45 selections violate row 0/1 by
`1.630478/3.838207/9.017434 mm`; E34 is unrecoverable and the MLP still selects
its unsafe nominal candidate. All selected actions retain original AEGIS EE
compatibility. Exact-safe selection is `1/4 = 25%`, essentially the same as the
fixed-seed random-bank rate `24.5768%`.

The prediction gate fails, so no fresh selected-action OSC replay is authorized
and no unsafe action is executed merely to reproduce an already authoritative
label. The result narrows the root cause: the model learns the local training
map but the compact 86D representation/data support does not transfer to
grouped states. Candidate selection, QP, calibration, closed loop, and learned
row-2/L6/L7 claims remain NO-GO. Model/validation file SHA-256 values are
`55b422390a5b32059cecaacfb6f0bf46fc3cd24c1fd82e8574328d82fc6e02f9`
and `d5ef9314fc4581026c5299b555be34397085eb689ed58e621e60e61aaebe16fd`;
payload SHA-256 values are
`9f44de2a857507fbe54ea7448fca9d427f7d43dad862e86cad3e228486af54ec`
and `ec5b44f338e871f2a04d5f39d2381c496024f7c0c3158e94228ca99b591dbfff`.

The matched input ablation is preregistered. Frozen artifacts contain every
requested field: arm q/dq, current and OSC-goal EE pose, recorded controller
memory, nominal/candidate chunks, obstacle transform and dimensions, L5
ellipsoid transforms and relative geometry, and five original-AEGIS EE QP
records. These form an explicit 354D feature vector. The raw 648D simulator
snapshot is intentionally excluded rather than used as an episode identifier.
The 86D and 354D arms share the same 43/29 labels, grouped splits, 32--32
architecture, seed, loss, optimizer, and schedule. No new simulation, candidate,
target, QP, calibration, or control change is included.

H100 producer `40181` and independent validator `40182` completed the matched
input ablation at commit `cc996f5a0ecd93baed35a4b926ebe22adf884624`.
The compact arm exactly reproduces frozen job `40175`, and the independent
validator reproduces every prediction from both arms with `0.0 m` maximum
error. The 354D input lowers grouped-validation RMSE from `14.716616 mm` to
`11.427894 mm` and near-boundary RMSE from `10.135355 mm` to `7.220164 mm`;
false-safes fall from 16 to 13. This is a real but insufficient input benefit.

Both arms fit the same training population closely (`0.164472 mm` compact and
`0.112910 mm` complete RMSE) but fail on disjoint physical states. The complete
arm rejects every candidate in recoverable E37, retains unsafe selections in
E39/E45, and remains exact-safe in only E22: exact-safe selection is still
`1/4 = 25%`, versus the fixed-seed random-bank rate `24.5768%`. It therefore
fails zero false-safes, 4/4 support, all-seven-safe selection, and the 1 mm
near-boundary gate. Omitted inputs contribute error, but do not explain the
grouped failure; state/episode coverage or state-generalizing representation is
the dominant unresolved issue. Fresh replay, calibration, QP, closed loop, and
sealed tests remain blocked.

Result/validation file SHA-256 values are
`7c9c4f95221ff025ff9da568ad031b78ae582f91f4a9a9060711fce581072312`
and `0e6d4af9829d4221543af0425ec379b2f5356c0d92b4e834cb103eb844dfa956`;
payload SHA-256 values are
`927e3b5421257f8e311029522416c4aa2528639c0ca7faf42fec7b2bccb3414e`
and `2f97054344d31b31a290b88013e6867ee7b8f025b9308c97d896684c27a2468c`.

The remaining root-cause audit is preregistered without new simulation. It
corrects an important overstatement: compact 86D is already partly relative,
so a genuinely invariant test must rotate both geometry and Cartesian action
vectors into attached physical frames. The new 134D arm uses obstacle-frame
OSC/action quantities and per-row ellipsoid-frame geometry, retains the exact
43/29 labels and training recipe, and is compared with fixed KNN/ridge
baselines plus the frozen 354D arm. Nearest-state support and prefix-versus-
backup witness switching are reported separately. Only this audit can decide
whether representation alone plausibly repairs transfer or whether additional
independent boundary states/structured rollout targets are mandatory.

H100 producer `40186` and final independent validator `40188` completed the
relative root-cause audit; validator `40187` is retained as apparatus history
for its legacy 86D loader assertion. Prediction replay error is exactly zero.
The invariant 134D MLP improves near-boundary RMSE `7.220164 -> 4.476086 mm`,
false-safes `13 -> 6`, and exact-safe selection `1/4 -> 2/4` relative to the
complete 354D model, but overall RMSE worsens `11.427894 -> 14.417849 mm`, safe
support remains only 3/4, and every strict prediction gate fails. KNN and ridge
select no safe action in any recoverable state.

The failure is now localized. E45 is unsupported (`1.284` nearest-state RMS z,
14/64 state features outside range) and changes from nearest backup witnesses
to prefix witnesses. E39 is nearby (`0.266` RMS z, 1/64 outside) with the same
backup witness phase, yet still has millimetre-scale candidate-response error.
Thus limited independent state/action-response coverage is primary; rigid-frame
representation is useful but insufficient, and prefix/backup switching is a
localized secondary cause rather than the sole explanation. No fresh replay,
QP, calibration, closed loop, or sealed-test access is authorized.

Result/validation file SHA-256 values are
`564b49b6e474cc37e1ea0902b2afec5ec6593dfa524c1260a8f3a98045bc296c`
and `febd9e70b5bda801dcead4acd4cefef07b17b7d0044fa1c148a4b621b9741824`;
payload SHA-256 values are
`7b16bc1f8097fc411245d283317e8d7062591342612e4c244cdb38d950f56be7`
and `adfa8060b0df65cf726ecd18473e92b18c85e7d931335401dd4efb651ead745a`.

ADR-0132 preregisters the requested last audit before new simulation. Existing
traces supply exact prefix risk for every known action and backup risk only when
the fixed backup actually ran; absent backup values remain censored. Two matched
134D models predict the components independently, their hard maximum is tested
against the direct combined model, and every false-safe is attributed to its
active phase. E39's exact component slopes are compared with the nearest
response-capable training state. This will decide whether new collection must
target prefix, backup, or local action-response diversity.

H100 producer `40190` and independent validator `40191` complete ADR-0132 with
zero replay error. Prefix and backup predictors both fail grouped validation:
prefix has `14.357290 mm` RMSE, `9.369379 mm` near-boundary RMSE, and 7 false-
safes; backup has `8.752153/5.309479 mm` and 9 false-safes. Their hard maximum
has 9 false-safes and only 2/4 exact-safe selections, so mechanism switching is
not the sole failure. Every one of the direct relative model's six false-safes
is backup-dominated.

E39 nevertheless has the same safety-improving slope sign as response-capable
training state E27 for all six identifiable row/component curves. This supports
learning action-induced risk change or ranking, while rejecting the present
absolute safety gate. The next collection should span backup crossings across
new E39-like states and prefix crossings across E45-like states; the next model
should separate state baseline from action-induced change. No QP, calibration,
gradient claim, closed loop, or sealed test is authorized.

Result/validation file SHA-256 values are
`d42acbaed425530a036ecb8a03456b3e5298980e9632e64d3a74f43ad62fc245`
and `57683263c435477cca0549a2e672d18204b3b80b3603ce380d635db2eb3d03ae`;
payload SHA-256 values are
`cb82a161b138b421c6a7ea64937c9d9d9f33599289709e3de10e119f10617220`
and `c0325c701592166258ef7805cd263f727b4106b236d520f635f45932ab25df41`.

ADR-0133 preregisters the factorized data gate rather than fitting a biased
baseline model. Twelve unused initial states across task-0 moka and task-2 milk
have fixed outcome-blind policy seeds and grouped 8/4 train/validation splits;
four additional initial episodes remain unopened. Eligible trajectories flow
through unchanged real-query warning-state discovery and adaptive-v3 candidate
plus fixed-backup labeling. The collection retains every task failure, missing
boundary, timeout, physical veto, and proxy-invalid case. Training remains
blocked until independent validation demonstrates known nominal anchors,
multiple response actions, and prefix/backup boundary support in both splits.

H100 producer array `40201` and independent allocation-backed validator array
`40202` completed all twelve registered cases. Aggregate attempt `40203` is
apparatus-only failure: the CPU summary allocation invoked the repository GPU
provenance recorder and `nvidia-smi` was unavailable. The replacement changes
only the summary resource request to one H100; producer trajectories, case
classifications, validations, coverage rules, and forbidden gates are frozen.

Replacement H100 summary `40229` completes ADR-0133. Classification is eleven
`TASK_SUCCESS_NO_L5_CONTACT` and one `TASK_FAILURE`; no trajectory is eligible
for warning-boundary collection. Train and validation therefore each have zero
known nominal/four-response states and zero prefix/backup boundary states.
Only artifact completeness, sealed-episode preservation, and zero proxy-safe
physical collisions pass. Factorized training remains unauthorized.

The validated summary is
`/mnt/data/quanth/experiments/vlsa-distal-l5-factorized-boundary/l5-factorized-boundary-20260814a/summary.json`.
Its file SHA-256 is
`2ac16449e75b3f4b9ffb09ab70aed5db22aa9082d34582bc699c1dcdb11e993d`
and payload SHA-256 is
`598d9d8dcdd79f3a3e0dbcbf78573bf3b040bda0ea44fdd1d0d0d6452364e6b5`.
This establishes that passive unused-episode/noise sampling is inefficient for
the rare L5 boundary; it does not test or reject the anchored `B+Delta` model.

ADR-0134 preregisters the cheapest remaining factorization test. The frozen
authoritative data are restricted to states with an exact nominal combined-risk
anchor plus at least four response actions: 3/3 grouped train/validation states
and 14/15 non-nominal responses. A 32--32 MLP predicts only
`Delta = Q(A) - Q(A_nominal)` and is architecturally zero at zero correction.
The exact nominal anchor is supplied at evaluation, isolating action response
from baseline-risk prediction. No new simulation is launched.

Each state contains amplitudes along one registered bisection path, so the
experiment tests held-out-state risk change, improvement sign, safe/unsafe
ordering, and Best-of-N amplitude selection—not an unrestricted action-space
gradient. A pass authorizes only a small active direction pilot; a failure
stops this formulation. QP, calibration, closed loop, nominal-risk learning,
and sealed-test access remain forbidden.

H100 jobs `40237`/`40238` complete ADR-0134. Independent replay reproduces all
predictions exactly and confirms architectural `Delta(z,0)=0`. Validation
response RMSE is `18.134943 mm` versus `19.091172 mm` for the zero-change
baseline. Improvement-sign accuracy is 15/15, safe/unsafe ordering is 16/16,
and all non-tied pair ordering is 45/45, but magnitude errors produce three
false-safes. Predicted-safe support is only 1/2 recoverable states, and the
minimum-correction selected action is exact-safe in 0/2.

The best-ranked action is exact-safe in both recoverable states but is exactly
the strongest registered path candidate in both. Since the frozen bank varies
only amplitude along one chosen path, this does not beat fixed analytical
repulsion or demonstrate a transferable direction. The exact-anchor scalar
`Delta` formulation is a strict NO-GO, and the capped active perturbation pilot
is not authorized. Result/validation file SHA-256 values are
`f7c58e33a51d47df1b069a1491ac8f9e761754503d56fbc9ad6eee81e4df1131`
and `7ec80fbfa9bd14b7b28fd2078a0c7b319b110d1f712d1246553e35e67c378cb5`;
payload SHA-256 values are
`434b17096bbe500509f3c18729c71adc07bebb314c57e06ecfa5569bcb905bdc`
and `6d9a224ee8297e3439e9e259ef6edd7b2543f09112ef325115d7de5495d1a940`.

ADR-0135 preregisters a no-learning, no-simulation audit of all 12 stored v3
coarse prefix screens across the 16 proxy-valid grouped states. It compares the
strongest outward-normal radius-2 proposal against every registered mode and
separately compares six requested-radius-one normal/up/side proposals. Exact
applied norm, clipping, L5 rows 0--2, physical veto, and the stored 0.5 mm
screen gate remain explicit.

The audit decides whether analytical repulsion supplies every observed prefix
safe solution or whether independent tangential/temporal modes rescue states.
Because complete fixed-backup outcomes exist only for the selected bisection
path, this audit cannot authorize learning even if prefix mode diversity is
found. Training, active collection, QP, calibration, closed loop, and sealed
tests remain frozen.

H100 audit `40244` and independent validator `40245` complete ADR-0135 without
new simulation or labels. Across 16 states and 192 stored prefix probes,
`normal_pos_front_loaded_r2.0` is the exact lowest-risk registered proposal in
16/16. Any proposal is zero-margin safe in 14/16 states; the strongest normal
is safe in the same 14/16, and no alternative rescues either remaining state.

Within the requested-radius-one subset, normal/up/side win 13/2/1 states, but
no tangent turns an unsafe positive-normal result safe and only one tangent
improves risk by at least 1 mm. Sixty-six non-nominal proposals were clipped.
The evidence therefore rejects learned mode selection for the current bank and
supports analytical outward-normal repulsion as the prefix baseline. Complete
continuation and task compatibility remain unevaluated across modes, so no
policy-value, learned-filter, QP, closed-loop, or general safety claim follows.
Result/validation file SHA-256 values are
`8b1e04744f3266939b211cd1a896d9fb995291aaf890b43f7c3b9f68d52b8dac`
and `7f9bba4371eab1cb631a936aea7261fcdb67d402e28cf65abc305c7dbc118b26`;
payload SHA-256 values are
`3541886f642409883ca12ba2bc0ee5dfe24628a94b4a743269a282c2bd6da352`
and `38fe30211687847d3b274186df279d38ca497dcdae45bc6fb7829704feaa3ac0`.

ADR-0136 is preregistered before opening any repulsion outcome on the three
previously sealed complete episodes. It freezes one oracle-triggered controller:
exact five-action cloned-OSC warning at +1 mm, fixed closest-slab outward normal,
front-loaded unit-L2 radius 2, action clipping, unchanged released AEGIS EE QP,
execute five, and live frozen-pi0.5 replanning. Goal-task-2 E42 and goal-task-3
E42/E44 provide one historical L5 and two historical L6 collisions while raw
AEGIS completed each task. The paired full-episode gate reports L5/L6/L7
contacts, CAR, native task completion, clipping, timeout, terminal motion, and
video. A strict pass requires safe task success in all three; no tuning,
learning, QP change, deployable-warning claim, or population claim is permitted.

Producer attempts `40263` and `40271` are retained as apparatus failures with
no scientific results. They exposed byte-exact clone/execution checking and a
too-tight `1e-12` tolerance: the actual maximum discrepancies were only
`2.397738e-12` and `2.416745e-12`. Before opening any repulsion outcome, the
apparatus tolerance is amended to `1e-9` and the exact errors are added to the
result and validator. This does not change the frozen controller or gate.

Attempt `40275` then revealed non-roundoff clone divergence during an unchanged
raw-AEGIS grasp/contact transition at step 105: simulator-state error reached
`0.0133021` even though the five commands began from an exactly synchronized
snapshot. This is retained as apparatus evidence. Clone/execution state and
seven-row boundary errors are now diagnostics rather than abort conditions;
actual all-substep MuJoCo contacts, CAR, and native task completion remain the
strict scientific gate. The fixed analytical controller is unchanged.

Final H100 producer `40285` and independent validator `40286` complete the
three sealed full episodes. Fixed radius-2 outward-normal repulsion removes all
L5--L7 contacts and passes CAR in 3/3, compared with collision and CAR failure
in all three immutable raw-AEGIS controls. Task preservation fails: only
goal-task-3 E42 completes (step 111, one intervention). Goal-task-2 E42 times
out at 300 actions after ten interventions, and goal-task-3 E44 times out at
300 after one. Both remain physically active rather than deadlocked.

The originally reported safe-task NO-GO and collision-only transfer verdict are
withdrawn as controller evidence. A visual audit found dense interleaved
corruption in both the video and separately written terminal JPEG, whereas the
immutable Table 1 video for the same case is valid. The 32px probe renderer was
constructed after the 1024px main renderer in the same OSMesa process, corrupting
subsequent main camera observations supplied to live pi0.5. Contact/CAR records
remain artifacts, but the closed-loop task comparison is confounded. Summary
and validation are retained as invalidated apparatus history at
`/mnt/data/quanth/experiments/vlsa-distal-analytical-repulsion-generalization/analytical-repulsion-generalization-20260815c/`.
Their file SHA-256 values are
`88092b4f7fd0c6fdff371f6264441e1252d1ba10faf23c673f40d93571ce000a`
and `f9a490ba8599bae5ae909e15c95e30e8dff235b0e47acefcbfb2ea1a619974d9`;
payload SHA-256 values are
`56caf77d007b82caf405147c503fa176433f4ede81176462af4397f1d372b6ba`
and `9ceccc9df2f5322b164ce1f4c0cb552d4ad1afba2ef1528d48d9cd6f340ab394`.

The initial video handoff failed in-app despite matching raw artifact hashes;
the raw MP4s were 96/34/28 MB and had not passed a delivery-specific full-decode
gate. ADR-0137 adds `slurm/finalize_simulation_videos.sbatch`. H100 job `40296`
transcodes to H.264 Main/level-3.1, yuv420p, fast-start, max width 512 and CRF
30, then decodes every frame and emits a manifest. Verified portable outputs
are 4.10/0.829/1.86 MB. Future video delivery must use this gate and must not
link raw experiment MP4s directly.

Portable transcoding did not repair the visual content. Root-cause inspection
showed corruption before encoding. The evaluator now creates and disables the
small probe renderer before creating the main renderer, and every raw frame is
checked with a normalized adjacent-pixel integrity metric before video writing
or policy use. The unchanged three-case H100 experiment must be rerun; neither
the prior task verdict nor the collision-generalization claim is final.

Corrected H100 producer `40303` and independent validator `40304` reran the
identical frozen three-case gate after commit `6225456`. Raw-frame integrity
passes throughout: maximum normalized adjacent-pixel scores are `0.007512`,
`0.008730`, and `0.008682`, all well below the fail-closed threshold `0.15`.
The three terminal JPEGs and macOS QuickLook decodes of the delivered videos
are visually clean. H100 finalizer `40319` also fully decodes the bounded-width
H.264 deliverables. Their byte sizes and SHA-256 values are:

- case 0, goal-task-2 E42: `70,059`,
  `c95d0c7bb26b39cbe1f4162289c1014b92832a27b39bf78c71bc00a7ea8e5391`;
- case 1, goal-task-3 E42: `131,278`,
  `9c9bfe490a63a66550fff87bdfeb37b82e999eef19b3aaf9f2d621abf1d9a66a`;
- case 2, goal-task-3 E44: `137,515`,
  `d008ee25ffd02fb17bbf8cb5f29ecf72c7d409d3b5e97297cbd7457a2be6a38f`.

The corrected scientific conclusion remains a strict safe-task NO-GO, but the
valid case-level outcomes differ from the corrupted run. All three episodes
have zero raw L5--L7 contact and pass CAR. Goal-task-2 E42 completes after five
interventions. Goal-task-3 E42 times out after 39 interventions and goal-task-3
E44 after 31; neither completes. Thus analytical repulsion transfers as a
collision-prevention mechanism on these three sealed cases, but safe task
success is only 1/3. The remaining failure is over-intervention/task
compatibility under a conservative warning signal, not insufficient repulsive
authority or video corruption. No learned selector, QP change, deployable
warning, population, or formal-safety claim follows. Result and validation
payload SHA-256 values are
`421adf1150c74fa7f194f14c54d6a0fa3c8b037efa834ca63a1ba139be871ea5`
and `7b246dbdbb8f67a3017471289a2cd3cd1421740e292caf004e80775101aa5900`.

ADR-0139 preregisters the no-learning exact normal-magnitude curve pilot. The
authoritative label remains the seven-row worst future violation over the
exact final post-AEGIS five-action prefix plus complete fixed backup. Nine
requested radii from zero through two are evaluated at the first warning state
of goal-task-2 E42, goal-task-3 E42, and goal-task-3 E44. The apparatus records
requested/effective magnitude, clipping, prefix/backup risks, active row,
substep/phase, internal margins, contacts, CAR, terminal status, and replay
identity. Timeouts remain unknown. Only a safe-support pass can authorize a
subsequent complete adaptive-magnitude episode; MLP training, denoising
guidance, QP changes, and new generalization claims remain blocked.

H100 producers `40339`--`40341` and independent validator `40348` complete the
ADR-0139 exact-curve gate. All three diagnostic first-warning states contain a
known safe magnitude. The minimum requested/effective post-AEGIS L2 values are
`1.25/1.065836`, `0.75/0.711527`, and `1.75/1.585015` for task-2 E42,
task-3 E42, and task-3 E44 respectively; no candidate clipped. Nine of 27
complete-backup evaluations timed out and remain unknown. Task-2 E42 is
monotone on known points, while both task-3 cases have one risk reversal, so
the next complete-episode oracle must choose the first exactly verified-safe
point on the fixed grid rather than assume bisection or map millimetres of risk
to normalized action magnitude. This is a mechanism GO for adaptive magnitude
testing only. MLP training, denoising guidance, QP changes, and generalization
claims remain blocked. Summary/validation file SHA-256 values are
`c02b37766c4d8b6b8268b5f29869fd8002466b4ed2614065e8668f7c218c6c37`
and `83d4e42f06d19e0bd0e904267e289cf9cb4e2fad1866913fc689ddb1a52affc2`;
payload SHA-256 values are
`a046bc854e32f8cf297cc00efd57d74d3f67467daedbb5fabe5100df40a1f4bc`
and `9829b9fc84205077980766fdf44d026cde5f2a63ff427ca2e9e9145143bbbd46`.

ADR-0140 preregisters the cheapest complete-episode magnitude diagnostic. It
freezes each opened case's first-warning minimum exact-safe requested radius
(`1.25/0.75/1.75`) and reuses it at every later warning. All other controller,
AEGIS, replanning, measurement, rendering, and acceptance settings match the
validated radius-2 run. This tests whether excessive magnitude caused the two
task timeouts without claiming online adaptation. The strict gate is 3/3 zero
L5--L7 contact, 3/3 CAR, 3/3 native task completion, no timeout, and lower
per-case total requested correction than the validated `10/78/62` radius-2
totals. Learning, denoising, QP changes, and generalization remain blocked.

Initial array `40352` exposed a fail-closed contact-accounting bug after the
task-3 E42 episode completed: a distal contact geom can have multiple protected
ancestors, while the writer incorrectly required exactly one. The repair
selects the closest protected ancestor and preserves the contact as scientific
failure. No radius, warning, action, controller, or acceptance rule changes.
Validator `40355` is retained as a dependency-cancelled apparatus attempt.

Clean H100 array `40357` and independent validator `40360` retain a strict
first-warning-calibrated magnitude NO-GO. Task-2 E42 is contact/CAR-safe but
times out after 12 radius-1.25 interventions (total requested norm `15`, versus
`10` for radius 2). Task-3 E42 completes but collides with L6 at step 108,
after the radius-0.75 proposal was already forecast unsafe at its second
warning at step 105; CAR also fails. Task-3 E44 is contact/CAR-safe but times
out after 33 radius-1.75 interventions (total `57.75`, versus `62`). Safe task
success is `0/3`.

The first-warning magnitude is therefore not reusable across the later states
created by live VLA replanning. The next oracle must measure the risk curve at
later warning states, starting with task-3 E42 step 105, and abstain or invoke
backup if no magnitude passes. No MLP, denoising, or QP is authorized. Attempt
`40352` also produced a different task-2 continuation under the same request
seeds, so future task comparisons require within-allocation paired arms or
repeated policy samples rather than treating separate policy-server launches
as exact stochastic pairs. Summary/validation file SHA-256 values are
`e8c3e0986ddde226330c1db182a59dbbeec7fb6d5d7d170c560df5b7defd48dd`
and `f58cd571e37701fc12607cff15df224fd42df07037d6121ddaf2f7cdcbdf9c43`;
payload SHA-256 values are
`3df38b493281aac8af34d27704a8f76d0a537b6d808c62985236137094447ae8`
and `0e9c6e90487c6561dc8b09a0ec31d68502d328fe91216dcf656fd3e2bc27fa55`.

ADR-0141 preregisters the strict task-3 E42 step-105 recoverability gate. The
clean `40357` executed ledger is replayed through step 104, while the exact
query-21 raw and released-AEGIS nominal actions are reconstructed from recorded
QP contexts. Source clearances and nominal projection must reproduce within
`1e-9`. The unchanged nine normal candidates then receive authoritative
five-action-prefix plus complete-fixed-backup labels. Unknown timeouts remain
inadmissible, and the oracle selects the known exact-safe candidate with
minimum realized post-AEGIS L2 command deviation. It does not execute the
selection. This is the smallest test of whether state-dependent finite-bank
governance could have prevented the step-108 L6 collision. MLP training,
denoising, QP changes, closed loop, direction expansion, and generalization
remain blocked pending independent H100 validation.

Producer `40365` and validator `40366` are retained as apparatus history, not
scientific evidence. They reproduced the exact post-AEGIS action but fitted
the slab templates for the first time at step 105, reversing the longitudinal
identity of L5 rows 0 and 2 relative to the source episode. The opt-in repair
primes slab templates at the settled initial state before action replay, as the
source episode did. The registered candidate bank and every scientific gate
remain unchanged; a replacement allocation is required before interpretation.

Replacement H100 producer `40368` and independent validator `40369` pass the
repaired provenance gate (`4.441e-16 m` clearance replay error; exact nominal
post-AEGIS action) and retain a strict registered NO-GO. Zero of nine normal
candidates is seven-row proxy-safe, one radius-1.5 rollout is an unknown
timeout, and all known active witnesses are L6 row 3. No minimum-intervention
candidate can therefore be selected under the preregistered `1 mm` ellipsoid
rule.

This does not reject physical normal repulsion. Radius 1.75 and 2.0 both reach
a stable terminal with zero protected MuJoCo contact and negligible CAR, but
the L6 row-3 proxy remains violated by 10.263 mm and 6.346 mm. The core audit
therefore isolates the blocker to quantitative proxy certification rather
than direction or correction authority. Training the future-risk MLP on the
current seven-row target, adding a QP, or increasing magnitude is not
authorized. The next experiment must validate a physical row-3 target or an
explicit conservative mapping between the slab proxy and compiled collision
geometry. Result/validation payload SHA-256 values are
`78cce6f36970a01d44d898ba21ae3ba84cf98d9619a3053f7470b8f26e4051c5`
and `1a469cf99cdfaa2d26ed019b7df409e1c47fd46cdc9677f4cb8f437ba5cc5bc6`.

ADR-0142 preregisters the next no-learning target-validity audit. It replays
eight immutable candidate-plus-backup ledgers from opened L6 states E44 step
110 and E42 step 105, retaining three raw-contact controls, four stable
contact-free controls, and one censored timeout. The seven robot slabs remain
unchanged; only the released obstacle MVEE is compared with exact solid
intersection against the compiled contact-capable obstacle-box union at every
internal MuJoCo substep. The resulting normalized radial slack is
dimensionless, not metric clearance. Passing requires zero physical
false-safes, safe initial states, safe support in both states, and recovery of at least three stable
actions with perception-MVEE solid overlap at zero buffer. The source's 1 mm
operational buffer is logged separately. No threshold is fitted, no action is
changed, and MLP/QP/denoising/closed-loop work remains blocked.

H100 producer `40387` and independent validator `40388` complete ADR-0142.
Source replay and initial-state eligibility pass. Exact compiled-box overlap
detects all 3/3 raw-contact controls with zero physical false-safes and keeps
safe support in 2/2 states. It rescues two perception-MVEE-rejected stable
actions: E44 radius 1.25 (`-6.390 mm` MVEE, `+0.022739` dimensionless compiled
slack) and E42 radius 2.0 (`-6.346 mm`, `+0.036349`). The strict gate remains
NO-GO because E42 radius 1.75 is raw-contact-free and stable but the fixed L6
row-3 slab overlaps the exact obstacle boxes for 24 substeps (`-0.015667`
minimum normalized slack). Obstacle MVEE conservatism is real but not the only
target error; the next audit must tighten or replace the robot-side L6 proxy.
Learning, QP, denoising, and closed-loop work remain blocked. Result/validation
payload SHA-256 values are
`b80c81aacd9d954d892492a4ffc3fd0ae0d2c498ea23d6de147e458f63ff763b`
and `d49fa478d0a14d840c6e30a69fab78024f6de0c28b4423361a6c123272b7b60f`.

ADR-0143 preregisters a narrow L6-only geometry heuristic. On the eight
validated ADR-0142 controls, sweep uniform L6 semiaxis scales
`[1, .995, .99, .985, .98]` and choose the largest scale that preserves all
three raw-contact detections while accepting all four stable contact-free
rollouts and retaining safe support in both opened states. The analytical
slack transformation requires no new action or simulator rollout. A scale
below one is explicitly non-enclosing and empirical; raw MuJoCo contact stays
the final authority. Learning and control remain blocked pending H100
validation.

H100 producer `40389` and independent validator `40390` pass ADR-0143. The
largest registered L6 scale that separates all controls is `0.98`: it improves
stable contact-free recall from `3/4` to `4/4`, preserves all `3/3` raw-contact
detections, retains safe support in `2/2` opened states, and does not relabel
the timeout. The scale is frozen in
`configs/vlsa_distal_l6_empirical_proxy.v1.json` for opt-in simulation risk
labels only. It is not an enclosing bound and does not support formal safety.
Result/validation payload SHA-256 values are
`f175c28699eb8f08eb6e6dcc32be1de57b581f10e1d59f9e481943d1d849fdc0`
and `c25e5f48a85276bcd6ad5e9717b0906f8874f7b758d83d89bb8972d54734fe85`.

ADR-0144 preregisters the next core-algorithm mechanism test. Four immutable
opened warning states contribute 36 existing post-AEGIS normal candidates.
The audit replays each five-action prefix and fixed backup using exact compiled
obstacle boxes plus the frozen empirical L6 scale, then selects the safe stable
candidate with minimum realized post-AEGIS correction. The strict gate requires
all stored contact and stable controls to be classified correctly, safe support
in `4/4` states, and a selected correction strictly smaller than always using
radius 2 in every state. No action is executed and no model is trained. H100
producer and independent validation are pending.

H100 producer `40404` and independent validator `40405` pass ADR-0144. The
empirical target detects all `10/10` raw-contact controls, accepts all `16/16`
stable controls, retains safe support in `4/4` states, and leaves ten timeouts
unknown. The selected requested magnitudes are `1.25/0.75/1.25/1.75`, all
strictly below radius 2 in realized post-AEGIS action L2. Mean realized
intervention decreases from `1.789281` to `1.064623` (`40.50%`). This validates
the exact finite-bank minimum-intervention governor as a mechanism on opened
states. It does not yet validate candidate-risk learning, execution-level task
preservation, or generalization; those require grouped independent states.
Result/validation payload SHA-256 values are
`287fdf6547f28bd44e4ed2dc3bbbd247107522f12db942861fa33af0f2307bc6`
and `45179b197f9eb4f897ec2167106a7a2de22a2da214693211d9c4a0bad0c2dabc`.

ADR-0145 preregisters the grouped dataset gate for the learned candidate-risk
stage. Ten training and three validation episodes from the clean cohort each
contribute one real five-action query boundary and the unchanged nine-point
normal bank. Every candidate receives a complete prefix-plus-fixed-backup
rollout, compiled-box/empirical-L6 scalar risk, full physical/controller
context, exact post-AEGIS actions, raw contact, CAR, and timeout status. The
strict gate requires two-sided support and recoverability in all `13/13`
states with zero missed raw-contact controls and no rejected stable controls.
Training remains blocked until independent H100 validation passes; diagnostic
and test episodes remain unopened.

H100 canary `40429` is retained as an apparatus-only preflight failure. The
reused source collector rejected task-2 E19 before candidate simulation because
its old perception-MVEE initial-clearance check was still active. ADR-0145 uses
the empirical compiled target for initial eligibility. The repair bypasses only
the obsolete proxy check in the opt-in empirical collector; initial MuJoCo
contact, CAR, empirical overlap, candidates, actions, AEGIS, backup, and all
registered gates remain unchanged.

Replacement H100 canary `40432` passes the complete collection path on task-2
E19: exact replay, initially safe empirical geometry, full context, four
compiled boxes, seven robot rows, and a two-sided bank with two stable safe
candidates. The artifact is marked apparatus-only. Full 13-case producer array
`40437` and dependent validator `40438` are now submitted with at most two H100
tasks concurrently from exact commit `e373a43`.

Regularization is now the prioritized post-collection diagnostic, without
interrupting ADR-0145. ADR-0146 freezes a matched standard-versus-monotonic
ablation conditional on the grouped dataset passing. The audit uses the exact
state input and realized post-AEGIS correction projected onto the registered
normal profile; requested alpha, clipped/direction-changing candidates, and
timeouts cannot silently define monotonic training pairs. Minimum intervention
remains a constrained candidate-selection rule, while a conservative
false-safe penalty is deferred to a separate attributable experiment.

The regularization order is corrected after user review. The first matched
training diagnostic after ADR-0145 is now no weight decay versus the established
AdamW `1e-4` weight decay, with every other factor frozen. The monotonicity
audit and loss remain prepared but are secondary; they cannot precede the
ordinary weight-decay attribution test.

H100 jobs `40437/40438` complete the grouped empirical candidate-risk dataset
with 13 cases, 117 candidates, 16 raw-contact controls, 65 stable controls, and
36 censored timeouts. The gate is strict NO-GO: only 4/13 states are two-sided;
the remaining nine have no known unsafe candidate after the complete fixed
backup. Four task-3 cases also exceed the frozen old-proxy replay tolerance by
only `1.6e-8--1.75e-7 m` despite exact state hashes and CAR. Training—including
the weight-decay ablation—remains blocked. Validation file/payload SHA-256 are
`f5c9e06412d4074ae940fdcbfcf04b334b966ff0f0616dba85b8559f4ac814bd`
and `66b97b944ab8008b1e86feeca441dd4cbea85dc08eb781786b9135820f791f21`.

The requested weight-decay check is now separated from ADR-0145. It uses the
older immutable 354D two-output L5 row-0/1 future-worst-violation dataset that
produced `0.112910 mm` training RMSE and `11.427894 mm` grouped-validation
RMSE. The preregistered arms use AdamW weight decay `0` and `1e-4`; all 43/29
labels, complete physical/OSC features, grouped episode split, normalization,
32--32 SiLU network, symmetric boundary-weighted Huber loss, seed, optimizer,
learning rate, 2000 epochs, and final checkpoint are identical. The `1e-4`
arm must reproduce the frozen job-40181 validation predictions within `1e-9`.
This no-simulation diagnostic asks only whether ordinary weight decay explains
the train/unseen gap. It cannot authorize fresh replay, calibration, a QP,
closed-loop control, sealed-test access, deployment, or a CBF claim.

H100 producer `40489` and independent validator `40490` complete that matched
test with zero reproduction and independent prediction-replay error. Without
decay, train/grouped-validation RMSE is `0.112900/11.429146 mm`; with the
established `1e-4`, it is `0.112910/11.427894 mm`. The validation change is
only `-0.001252 mm` or `0.01095%`. Near-boundary RMSE is effectively unchanged
(`7.218239` versus `7.220164 mm`), and both arms retain 13 false-safes, safe
support in `3/4` recoverable states, and an exact-safe selection rate of
`1/4`. Ordinary weight decay therefore does not explain or repair the
`0.1 mm` train versus `11 mm` grouped-state gap. Prediction, replay, QP, and
control gates remain blocked. Result/validation file SHA-256 values are
`9a72b09b804b50eb498028200d13b14d75fae3e70aa8151e2333a64e02a5d6f8`
and `f4f62998cb5a2a09d20a64823ea32a517a7980d425e1dc44e0cd12df5ed4596d`;
payload SHA-256 values are
`d466cd3b211ea7b460a2b21de9b773c9eef269cad1017e72cd5fcb398150d978`
and `3620f87398a7f30b457b571a57f64c64147f3fccac01fee8ed0c5aee5b2db6fd`.

ADR-0148 now begins the requested gradual input ablation on those same frozen
labels. Its first arm is intentionally only six dimensions:
`[P_start, P_end]`, with
`P_end=P_start+0.05*sum_k(A_post_AEGIS[k,xyz])`. The `0.05 m/action-unit`
registered OSC scale is applied without subtracting a normalization mean. The
endpoint is computed from commands available before execution; actual future
EE position and every other rollout quantity are forbidden inputs. The 43/29
labels, episode split, 32--32 network, symmetric loss, seed, AdamW `1e-4`,
optimizer, schedule, and final checkpoint remain matched to the frozen 354D
comparator. If this arm is insufficient, the next single addition is only
obstacle-relative position and dimensions. No new simulation or control is
authorized.

H100 producer `40494` and independent validator `40495` complete the 6D arm
with zero prediction replay error. Endpoint-only train/grouped-validation RMSE
is `0.814920/16.068010 mm`, compared with frozen 354D
`0.112910/11.427894 mm`. Near-boundary validation RMSE worsens from
`7.220164` to `13.871146 mm`. The simpler model is more conservative—its
false-safes fall from 13 to 6—but support remains `3/4` and exact-safe
selection remains `1/4`. Thus endpoint-only does not resolve transfer and
also loses within-training information. The next staged arm adds only
obstacle-relative endpoint coordinates and obstacle semiaxes, remaining 9D.
Result/validation file SHA-256 values are
`1e46bb48d892372dde9f1d8c0cd12c4a0cad7fe355da9557e09b9452c3479c90`
and `d37fbc203809167f14410dabb1dff3df8e1e14a0cd2598addf8d7dc575bcdbf0`;
payload SHA-256 values are
`e9c31a9b314d7c0d0ed7b8e64813466f766281a30399db591ad6652f4ad5a601`
and `6c38ec9946591a6a12d71e229e79741b4ce4f0d87480e1898f9665f1fd973ec0`.

ADR-0149 freezes the second staged arm. It uses only nine inputs:
`[P_start-c_obstacle, P_end-c_obstacle, obstacle_semiaxes]`. The same command
endpoint, labels, episode groups, model, loss, regularization, seed, optimizer,
schedule, and checkpoint are retained. Obstacle rotation, q/dq, controller
state, link-row poses/normals, intermediate waypoints, action rotation/gripper,
and future execution state remain excluded. If this group is insufficient,
the next single addition will be only the three current L5 clearances.

H100 producer `40496` and independent validator `40497` complete the 9D arm
with zero replay error. Train/grouped-validation RMSE is
`0.866652/7.305681 mm`; near-boundary validation RMSE is `6.150915 mm`.
Compared with 6D, validation RMSE improves by `54.53%`, false-safes remain 6,
and support rises from `3/4` to `4/4`. It also outperforms the frozen 354D
model's `11.427894 mm` validation RMSE and 13 false-safes. Nevertheless,
minimum-correction exact-safe selection remains only `1/4`; prediction/control
gates stay blocked. The next staged arm appends only three current L5
clearances, for 12 total dimensions. Result/validation file SHA-256 values are
`ad558908dfb3bb33a0eae9109dbf9f6f521350ddca2f4c0d40c8f1e3a978fc31`
and `877de338c0a60d28c1bc6f9bd43a37650ebb1eed62a0d96ea877bca1dac13008`;
payload SHA-256 values are
`ddad0193514daec83f074b9db047bc70734f7ebbe46b3cad8ef6e63122307f8a`
and `551b2c3537419a074fb7b1b30c8dac2f491fdca0b1537cb5fed5dfc0733ae5a7`.

ADR-0150 freezes the third staged arm. It appends only the three current L5
row clearances to the validated 9D obstacle-relative endpoint input, producing
12 dimensions. No other physical, controller, action-path, or future-state
quantity is added. Data, split, architecture, symmetric loss, AdamW `1e-4`,
seed, optimizer, schedule, and checkpoint remain identical. This directly
tests whether the missing state-dependent risk offset is explained by current
L5 proximity.

H100 producer `40498` and independent validator `40499` complete the 12D arm
with zero replay error. Adding the current clearances lowers training RMSE to
`0.621411 mm` but catastrophically raises grouped-validation RMSE to
`21.860198 mm` and near-boundary error to `14.856301 mm`, versus 9D
`7.305681/6.150915 mm`. False-safes fall from 6 to 3 and exact-safe selection
rises from `1/4` to `2/4`, but this is again concentrated conservatism: support
falls from `4/4` to `3/4`. The staged experiment therefore stops. The 9D
obstacle-relative endpoint model remains the best predictive representation,
but its six false-safes and `1/4` minimum-correction selection keep control
blocked. Audit train/validation support of the three clearance features before
testing any additional input. Result/validation file SHA-256 values are
`4028c007ef977dffa5b76083149005aa7d0cddac01536d3cde7a193e8ee0fdc1`
and `43037e29048d7515e189bf90e14c1e1edd776481d1207bcbc6d72fca74977971`;
payload SHA-256 values are
`3d24979b3f54590e194a0f36aac34d821aa8890f1afecfc775ccbb3956e9d5ee`
and `defe40572bd7f339d5c501f653debf59cb967ab0fce3999cfa4935b70c62e67f`.

ADR-0151 preregisters the requested no-training distribution audit before any
new input or model change. The frozen 9D and 12D predictions, labels, and
episode splits are bound by file and payload hashes. Because the three current
L5 clearances are constant across every candidate from one simulator state,
the audit counts each physical state once rather than treating 43/29 action
labels as independent clearance observations. It reports per-row training
ranges, state-balanced and candidate-weighted moments, validation range and
nearest-state z distances, and per-state 9D-to-12D changes in error,
false-safes, and safe support. This can establish an association between
clearance shift and model degradation, not a causal root cause. New
simulation, retraining, feature addition, calibration, QP, closed loop, and
sealed-test access remain forbidden pending allocated-H100 reproduction.

H100 audit `40500` and independent validator `40501` reproduce the frozen
artifacts exactly. The 43/29 candidates collapse to only 10/6 distinct
physical states, of which four validation states are recoverable. Three of six
validation states have at least one clearance outside the distinct-training-
state range. The decisive support loss is E45: its L5 row-1 clearance is
`7.792309 mm`, below the entire `22.185141--74.254913 mm` training range and
at `-2.39844` state-balanced standard deviations. There the 12D model loses
all predicted-safe support and its per-state RMSE rises from `9.271008` to
`47.698618 mm` (`+38.427610 mm`). No other validation state loses support.

The association is specific, not universal. Across only six states Pearson
distance/error correlation is `0.8049`, dominated by E45, while Spearman is
`-0.0857`; two other out-of-range states improve or change negligibly. Thus
the audit supports clearance extrapolation as the cause of the catastrophic
E45 failure, but does not establish clearance shift as the root of every
unseen-state error. Candidate-weighted training means also differ from
state-balanced means by `+10.477/-1.710/-12.213 mm` for rows 0/1/2, exposing
unequal state duplication in ordinary normalization. Retain the 9D model;
do not add inputs or train. If clearances are revisited, first add independent
low-row-1-clearance training states and use state-balanced sampling or
normalization. Result/validation file SHA-256 values are
`5607de7fcc395718f1a257746932469de4ac9edc3badd4122c01d7e4963cdd99`
and `24986f31b52155c6d0eec543659b1a54f03ddd61f7aecb7ded4c93546e46ba1e`;
payload SHA-256 values are
`923c7dc1458ceb4e17b7ec9f1cb9293c9ded55b23fa7e4deacc90f857315db5a`
and `aa4ddd904266aff15db73ca7ef65f1569b523e53aba291dae08020ee415e7c8a`.

ADR-0152 begins the recommended task-generalization sequence with a mandatory
availability audit rather than immediately generating or training on more
actions. The existing 18-case clean manifest covers only Goal Level-II tasks
0, 2, and 3, so it cannot support four-fold task holdout with both obstacle
levels excluded. The new no-simulation gate scans all 400 immutable released-
AEGIS Goal-suite episodes (four tasks, two levels, 50 episodes each), validates
result/contact hashes, and classifies clean task-successful L5 cases using the
same proxy-valid, no-initial-contact, no-dynamic-contact rules. For each fold,
one logical task and both its levels are excluded from training. A fold is
available only when each held-out level has at least one clean L5 case and
training contains clean cases from at least two other tasks. Only after all
four folds pass may two-sided boundary collection begin. New simulation,
training, candidate selection, calibration, QP, closed loop, and generalization
claims remain forbidden pending H100 audit and independent replay.

H100 audit `40505` and independent validator `40506` reproduce all 400 cases
and reject the proposed four-fold population before any new rollout or
training. Exactly 12 clean task-successful L5 cases exist: task 0 Level II has
2, task 2 Level II has 8, and task 3 Level II has 2. Every Level-I task has
zero, and task 1 has zero at either level. Consequently all four held-out-task
folds fail the preregistered requirement of one clean test case per held-out
level. The remaining classifications are 268 task-success/no-CAR, 88 task
failures, 13 proxy-invalid/incomplete, 9 non-L5 distal contacts, 5 dynamic
out-of-scope contacts, and 5 CAR failures without L5--L7 contact.

This is a deployment-population identifiability failure, not an MLP result.
Do not retrain the 9D model and do not balance the absent cohorts by duplicating
actions. The four-fold recommendation requires either a prospectively expanded
task/obstacle population that genuinely produces clean L5 boundaries in all
held-out task/level cohorts, or a narrowed Level-II mechanism claim over the
supported tasks. Merely changing policy noise to hunt rare collisions in
Level I would alter the trajectory distribution and still would not guarantee
task-compatible two-sided boundary support, so it is not launched silently.
Result/validation file SHA-256 values are
`07b3c5f2a6e19aa7ea1c8e4bf75e2f85ff3dd0b0015daec8e0a915f6cf8b1585`
and `18d2c5f660c11fb8e0642d16e4a10a4016f5b9a7738960fa095a5d79d18d61b7`;
payload SHA-256 values are
`f343348e281b6db7f60f7dc3a7fb1f00700e54a64c98420aa6f119d66d2700fc`
and `cf8cd4b52d6db5983ab93b7d861a548aea37f5041e5de68f3af42158f8488e06`.

## Full-population EE/distal primitive availability audit (preregistered, 2026-08-15)

The L5-only four-task audit is not the correct population gate for the proposed
shared constraint-conditioned model.  The released AEGIS proxy covers a large
end-effector volume, while the physical contact authority records the exact
palm and finger collision geoms.  Before fitting a tighter proxy or collecting
candidate actions, the new audit scans all 1,600 immutable AEGIS Table-1
episodes and keeps palm, finger-1, finger-2, L5, L6, and L7 contacts separate.

The audit requires complete task-successful, proxy-valid, initially safe,
non-dynamic-confounded episodes for a clean contact control.  Paper CAR is
reported but is not required because a physical contact can be real without
moving the obstacle beyond the CAR threshold.  Every unmapped robot geom is
reported.  A primitive is ready only for a subsequent geometry audit when it
has at least three clean contact episodes across two task/level groups and
three matched clean contact-free controls.  This gate runs no simulation,
fits no primitive, collects no boundary labels, and authorizes no training or
control.  Its next question is strictly whether the larger EE population
contains independent physical controls for tighter palm/finger geometry.

H100 producer `40509` and independent validator `40510` reproduce all 1,600
classifications with zero mismatch. Clean task-successful contact support is
3 palm episodes across two task/level groups, 20 L5 episodes across five
groups, and 16 L6 episodes across three groups; these pass the preregistered
availability threshold with 40/113/56 matched contact-free controls. The
registered finger-1/2 base geoms provide only 2/0 clean episodes, and L7 has
none. Training and boundary collection remain blocked.

The fail-closed unmapped ledger exposed two additional contact-capable geoms:
`gripper0_finger1_pad_collision` has 296 contact events and
`gripper0_finger2_pad_collision` has 151. They were not counted as negative
controls. Before fitting EE geometry, a v2 taxonomy audit now preregisters
each base and pad geom as a separate primitive. This preserves tighter
geometry and avoids recreating a single oversized finger ellipsoid. Result
and validation payload SHA-256 values are
`a861c47e5a854fb6b278f2659aa55627a7c16689e8819f9bbbd7c9eb39af594e`
and `5ff6d96eb21c64d83d72d08bdd33bb3288fa113eb60c677a31cc8df11e4973cc`.

Corrected H100 producer `40513` and independent validator `40514` reproduce
the exact-geom v2 taxonomy over all 1,600 cases with zero mismatch and zero
unmapped robot contacts. The base/pad split confirms that raw EE contact
frequency is not equivalent to clean learning support. Palm remains minimally
supported at 3 clean episodes / 2 task-level groups / 40 matched controls.
Finger-1 base is 2/1/14; finger-1 pad, finger-2 base, and finger-2 pad have
zero clean task-successful contacts despite total contact-episode counts of
5, 44, and 15 for the latter three groups. L5 remains 20/5/113, L6
16/3/56, and L7 has no clean support.

The supported first geometry pilot is therefore palm + existing L5/L6, not
all EE or whole-arm constraints. Finger and L7 remain explicit analytic and
physical-contact diagnostics until new independent boundaries exist. Palm
still needs a tighter-primitive physical false-safe/false-unsafe audit before
candidate-plus-backup boundary collection; training stays blocked. Result and
validation payload SHA-256 values are
`a8f41767e98d7c03076f626ce1471d7377883bfed5cb3c36871edd1f6fc12466`
and `b62a81428d5e110e77a7fd1fb5a835048bb28c9c3df72c9fa151016c874cce88`.

## Tight compiled-palm primitive gate (preregistered, 2026-08-15)

ADR-0155 implements the next allowed geometry-only experiment. It fits one
certified minimum-volume enclosing ellipsoid to the compiled vertices of
`gripper0_hand_collision`; the three palm-contact outcomes and 40 matched
contact-free controls are never used to choose its center, rotation, or
semiaxes. The immutable released-AEGIS action ledgers are replayed from their
paired simulator states, and tight-palm and released-EE support gaps are
measured at every one of the 25 internal MuJoCo substeps per action. Raw palm
contact remains the physical authority.

The strict gate requires exact replay of all 43 episodes, reproduction of all
three palm-contact episodes and all 40 internally contact-free controls, a
verified compiled-vertex enclosure certificate, and zero tight-primitive
physical false-safe samples. Control false-unsafes and the released-proxy
comparison are reported but do not tune the fit. Only a passing independent
replay authorizes grouped two-sided candidate-plus-fixed-backup boundary
collection for palm, L5, and L6. MLP training, QP, denoising, closed loop,
finger/L7 learning, deployment, and formal safety claims remain blocked.

H100 canary `40531` reproduces the positive E09 replay and contact-free E01
control exactly. The palm fit is only `0.411428` of the released EE-proxy
volume. E01 changes from a released-proxy minimum of `-36.861 mm` to a tight-
palm minimum of `+5.673 mm`, with zero raw palm contact. In E09, all 28 raw
palm-contact samples have contact points inside the certified tight primitive,
but the tight-palm/static-obstacle-MVEE support gap is positive at every raw
contact sample (the first is `+25.427 mm`); even the released EE proxy reports
`+9.257 mm` there. Thus the registered end-to-end static-obstacle gate is an
immediate NO-GO, while the robot-side palm fit passes its enclosure evidence.
The full 43-case static-proxy replay is not launched because its zero-false-
safe gate is already impossible.

ADR-0156 preregisters the isolated replacement: keep the tight palm fixed and
represent the moving active obstacle as a union of one certified enclosing
ellipsoid per contact-capable compiled collision geom. Mesh bounds use the
same outcome-independent compiled-vertex MVEE; standard primitives use their
registered exact or Loewner bounds. This is privileged simulation geometry,
not deployable perception. The identical 3/40 cohort, archived actions,
internal-substep sampling, raw contacts, and strict zero-false-safe gate stay
fixed. A passing canary is required before the full independent replay.

H100 compiled-union canary `40539` catches all 28 E09 palm-contact samples
with zero false-safes, but it is unusably conservative on E01: the physically
contact-free control has a `-21.824 mm` minimum and 768 proxy-overlap samples.
The registered union encloses 21 wine-bottle or 11 mug geoms independently;
overlap of those enclosing volumes is not equivalent to physical contact.
Do not shrink or threshold this union after observing the control.

ADR-0157 therefore tests only the stale-pose hypothesis. The released
obstacle MVEE shape is frozen at the settled state and expressed once in the
active obstacle root-body frame. At every substep, its pose is updated by the
current exact MuJoCo root-body transform; no shape, scale, buffer, or threshold
is fitted. The E09/E01 canary must have zero false-safes and accept E01 more
often than the unchanged released robot/static-obstacle comparator before the
full cohort is allowed.

H100 pose-tracked canary `40541` rejects ADR-0157. Replay is exact and E01
remains correctly accepted at `+5.673 mm`, but E09 still has 28/28 physical
false-safe contact samples; its first-contact primary gap is unchanged at
`+25.427 mm`. Thus neither stale obstacle pose nor the palm enclosure causes
the false-safe. The released obstacle point-cloud MVEE does not cover the
compiled wine-bottle collision surface involved in the palm contact. The full
3/40 replay, boundary collection, and training are stopped.

The palm evidence itself remains positive but narrower: its compiled-vertex
certificate passes, all 28 contact witnesses lie inside it, and it removes the
released EE false-unsafe on E01. To preserve a quantitative future-violation
target, the next geometry backend must provide a validated distance to actual
compiled obstacle geometry (for example FCL/GJK). The simpler alternative is
to change the finite-candidate risk target to raw future MuJoCo contact
probability; that would be a distinct preregistered method, not a repair of
this continuous ellipsoid-margin gate.

## Exact compiled-box palm/L5/L6 gate (preregistered, 2026-08-15)

ADR-0158 removes the failed obstacle representation rather than fitting a new
one. The registered active-set box solver computes exact dimensionless radial
slack between each certified robot ellipsoid and every live compiled obstacle
box. Inspection of the paired E09/E01 compiled canaries found 21/11
contact-capable obstacle geoms, all boxes. Any non-box encountered later is a
fail-closed unsupported case, not an approximation.

The first H100 canary replays E09 (observed palm, L5, and L6 contacts) and E01
(matched contact-free control) at every internal substep using the certified
tight palm and frozen certified L5/L6 slab rows. It requires exact replay,
valid robot enclosure certificates, zero represented-geometry false-safes for
all three groups, and positive control clearance. No obstacle MVEE or compiled-
geom ellipsoid union is used. Training and control remain blocked pending this
canary and the subsequent full independent geometry gate.

H100 canary array `40548` completes exactly and passes. E09 contains 28 palm,
169 L5, and 2 L6 contact samples with zero exact-target false-safes; minimum
dimensionless radial slacks are `-0.056608`, `-0.322074`, and `-0.037365`.
Contact-free E01 remains positive for all three groups at `+0.753981`,
`+0.047423`, and `+0.217761`. Both cases pass all robot primitive certificates
and exact state/action replay. A batched Numba implementation of the identical
27-active-set arithmetic reproduces the scalar solver exactly and reduces the
paired replay to about 1.7 minutes per episode. The unified final gate now
contains all 31 unique clean palm/L5/L6 contact episodes and three shared
matched contact-free controls; it changes coverage, not the target or fit.

Optimization attempt `40556` failed before simulation on a Numba import
opcode, and `40558` failed the exact-commit preflight after a control-plane
commit-string typo. Correct replacement `40560` passes the batched-versus-
scalar exact-arithmetic unit test and reproduces the canary minima and contact
counts exactly in `1:45/1:40`. Final producer/replay/verifier jobs
`40562/40563/40564` are running from clean commit `2c84f96`, capped at two
H100s. The first completed population case, spatial-I task-3 E03, reproduces
83 L5 and 193 L6 contact samples with zero physical false-safes.

Population producer `40562` exposed an apparatus-only compatibility defect in
three archived cases (array indices 22, 26, and 30): a diagnostic released-
perception MVEE orientation was an orthogonal reflection, while the generic
`Ellipsoid` record requires a proper rotation. The exact compiled-box target
does not use that obstacle MVEE. The repair canonicalizes only this diagnostic
orientation to its nearest proper basis, records the original determinant and
orthogonality error, and leaves the exact palm/L5/L6 target unchanged. Local
regression tests cover reflected and invalid bases. Because artifacts are
commit-bound, the full 34-case producer/replay/verifier will be restarted from
one clean repaired commit rather than mixing the completed old-commit cases
with replacements.

Replacement producer/replay/verifier jobs `40606/40607/40608` are bound to
clean repair commit `be26b3d` and a new immutable `20260815b` artifact root.
Producer `40606` started two H100 tasks concurrently; `40607` and `40608`
remain correctly dependency-blocked. The unchanged scientific gate still
requires all 34 producer cases, all 34 independent replays, and aggregate
verification before boundary collection is authorized.

Replacement chain `40606/40607/40608` completes and verifier `40608` passes.
All 34 producer artifacts and all 34 independently replayed artifacts are
present; all replay and robot-primitive certificates pass. Across 31 contact
episodes, raw-contact sample counts are palm `249`, L5 `3641`, and L6 `1539`,
with zero represented-geometry physical false-safes in every group. All three
matched contact-free controls retain positive exact represented clearance for
all three groups. Validation payload SHA-256 is
`cef3825f63b55e39db11c8a5d0618b6caf93c6d9c84c577b0ce80ac2ee637ca6`
and file SHA-256 is
`ffab8da0984b90b86edbb955de4ab35c66da9b159b71fb4abff12a52d29c6bda`.
The exact compiled-box target is now frozen and palm/L5/L6 boundary collection
is authorized. Training, QP, denoising, and closed-loop control remain blocked
until grouped two-sided boundary coverage is independently validated.

## Exact palm/L5/L6 boundary apparatus canary (preregistered, 2026-08-15)

The next phase is frozen to three independent development query states: palm
E09 at step 20, L5 E19 at step 25, and L6 E09 at step 110. At each state, nine
front-loaded outward-normal magnitudes are generated from the target group's
closest certified robot primitive and exact compiled obstacle box. Every
proposal passes through the original released AEGIS EE-QP exactly once, then
the resulting five actions and unchanged complete fixed backup are replayed.
The new learned correction QP, MLP training, denoising guidance, selected-
action execution, closed loop, and sealed tests remain disabled.

Authoritative labels are the internal-substep minimum exact dimensionless
radial slack and raw MuJoCo contact/CAR evidence, with prefix and backup phases
retained and timeouts censored. An independent H100 replay must reproduce all
27 candidates, all primitive certificates, actions, states, exact traces,
contacts, and CAR. The apparatus may authorize only a broader episode-grouped
collection, and only if each palm/L5/L6 bank contains known safe and unsafe
target outcomes with zero represented-geometry physical false-safes. It cannot
authorize training or control.

H100 producer `40706`, independent replay `40707`, and verifier `40708`
complete all 27 candidates with exact scientific reproduction and zero
represented-geometry physical false-safes. The common bank is nevertheless
NO-GO for scaling or training. L5 E19 has useful two-sided support (2 safe,
4 unsafe, 3 timeouts; 4 near-boundary). Palm E09 is one-sided unsafe (0 safe,
9 unsafe; 3 near-boundary). L6 E09 is one-sided safe and far from its target
boundary (9 safe, 0 unsafe; 0 near-boundary).

The L6 legacy released-proxy diagnostic differs by `1.564e-7 m` from its old
`1e-9 m` replay tolerance despite an exact state hash; this makes the strict
apparatus flag false but cannot explain the one-sided exact target. Validation
status is `apparatus_no_go`; grouped collection, MLP training, QP, and closed
loop remain unauthorized. Validation file/payload SHA-256 values are
`8d3f3e3fee6cb41236c256f181694e07fcb6ea5223670846f20d14d5035d698f`
and `614389369ac78bc7f223b7eda6270d920278d6ec92e7fe43527941bad59ee96b`.

## Direct no-QP L5 boundary canary (preregistered, 2026-08-15)

To isolate the core learned-candidate hypothesis, the next canary removes all
safety QPs from the proposed candidate execution. The frozen VLA supplies the
nominal five actions; nine target-relative outward-normal magnitudes are
clipped only to the registered action bounds and passed directly to the
unchanged OSC. Released AEGIS remains an archived baseline comparator and is
not applied to these candidates. No learned QP, gradient correction, MLP,
denoising guidance, or selected action execution participates.

Five initially valid development states span five independent task/level
groups and target only L5. Each retains the exact candidate-plus-fixed-backup
future-risk target, compiled-box geometry, raw palm/L5/L6 contacts, CAR,
timeouts, and complete internal-substep traces. The apparatus question is only
whether direct Cartesian candidates produce known safe and unsafe L5 outcomes
in every state. A legacy released-proxy replay discrepancy is diagnostic only;
exact source-state hashes plus independent exact scientific reproduction remain
mandatory. Even a pass authorizes only broader no-QP L5 data collection, not
training or control.

Initial producer array `40714` is retained as apparatus history. Case index 3
(L5 E19) failed before simulation because its new selection-manifest file hash
was copied incorrectly; the immutable archived payload hash and every
scientific setting were correct. The repair binds the already validated exact
Table-1 file SHA-256 and requires a new clean commit/root rather than mixing
old-commit successful cases with the replacement.

Clean replacement producer `40721`, independent replay `40722`, and verifier
`40723` complete all five cases and 45 candidates with exact scientific
reproduction and zero represented-geometry physical false-safes. The frozen
normal-only bank is nevertheless strict NO-GO: only E03 and E19 have two-sided
L5 support; E05 and E14 are one-sided safe; E02 starts with negative exact
target slack and then yields three known unsafe plus six unknown timeout
outcomes. Counts by case are E03 `7/2/0`, E02 `0/3/6`, E05 `9/0/0`, E19
`2/4/3`, and E14 `8/0/1` for safe/unsafe/unknown. Training and control remain
blocked. Validation file/payload SHA-256 values are
`2e2d76cc128fc6b137b6fbabf7082e58c55a46d5b889cc2b33cc95c426307e31`
and `7fe647edb06f4648d498b5a88d2c6d1301f571425d25511d80f7bc460dce6a94`.

The result rejects normal magnitude as the sole data-excitation coordinate;
it does not reject the exact worst-future-violation target. The next gate must
choose initially positive states whose nominal exact risk is near zero and use
a small generic symmetric Cartesian action basis. It remains a no-learning
support gate. Only independently reproduced, episode-grouped two-sided support
can authorize training a compact constraint-conditioned future-risk model.

## Generic Cartesian L5 future-risk support canary (preregistered, 2026-08-15)

The next support canary freezes the exact compiled-box candidate-plus-complete-
continuation target and changes only state/action excitation. Five explicit
real query boundaries are E03/60, E02/215, E05/185, E19/25, and E14/115. Each
bank contains nominal plus uniform five-action translation corrections along
world `+/-x`, `+/-y`, and `+/-z` at L2 radii `0.5` and `1.5`, for 13 candidates.
The basis does not use obstacle normals, rollout outcomes, learned directions,
or task-specific route labels.

Released AEGIS EE-QP and every learned QP remain disabled in the proposed arm;
actions are clipped only to the registered bounds and sent to the unchanged
OSC. Training, candidate selection/execution, gradients, calibration,
denoising, closed loop, and sealed tests remain forbidden. The canary must
independently reproduce all 65 outcomes, start positive/contact-free in every
state, have zero represented-geometry physical false-safes, and expose known
safe and unsafe L5 actions in every state. Passing authorizes only broader
episode-grouped collection for a matched 9D relative-endpoint risk model.

Initial producer `40732_4` rejected E14 before candidate simulation because
the preregistered step 120 requires archived actions through 124, while the
successful episode ledger ends at action 121. Cases 0--3 completed but remain
apparatus history. The correction binds E14 to its preceding complete real
five-action query at step 115 and requires a full clean-commit restart; no
candidate, target, controller, continuation, timeout, or gate setting changes.

Clean replacement producer `40739`, independent replay `40740`, and verifier
`40741` complete all 65 candidates with exact scientific reproduction, exact
state hashes, certified robot primitives, and zero represented-geometry
physical false-safes. The generic basis is nevertheless strict NO-GO. Per-case
safe/unsafe/unknown/near-boundary counts are E03 `0/13/0/0`, E02 `0/8/5/0`,
E05 `0/11/2/0`, E19 `0/11/2/11`, and E14 `6/6/1/5`. Only E14 has two-sided
support. E02 remains initially outside the exact target safe set with slack
`-0.090868`; the other initial slacks are positive.

The exact target and replay apparatus remain valid, but uniform signed world-
axis corrections do not expose safe actions in four of five states. Broader
collection, 9D MLP training, candidate correction, QP, denoising, and closed
loop remain unauthorized. Validation file/payload SHA-256 values are
`8c147a39625d5a2827bf84fa39a2d251d8fd11faef72328e265dacd08634c620`
and `bf20e5b48ecd951f5f0325c9fddf9eb2a04dbf2f0e0d1f3e905fce649d7bd9fd`.

## Generic L5 9D capacity diagnostic (preregistered, 2026-08-15)

At the user's request, use the failed support population only for a narrowly
scoped capacity diagnostic. This does not override the support NO-GO and
cannot authorize correction or control. The model uses the previously best
9D representation: current EE position and the commanded five-action endpoint
relative to the recorded obstacle center, plus recorded obstacle semiaxes. It
predicts the three exact compiled-geometry L5 primitive future violations;
their maximum is the reported worst-future violation.

E03, E05, E19, and E14 are prevention states and are evaluated by four-fold
leave-one-state-out prediction. E02 is initially outside the target safe set
and is excluded from prevention training and claims; a model fit on all four
prevention states reports E02 only as a recovery diagnostic. Unknown timeouts
are masked, candidates remain grouped by state, and loss and normalization
weight each training state equally. Architecture and training match the prior
9D diagnostic: 32--32 SiLU MLP, boundary-weighted symmetric Huber, AdamW,
weight decay `1e-4`, seed `20260814`, and a fixed final epoch.

The decisive outputs are unseen-state row/global error, false-safes, safe
support, within-state improvement-direction accuracy, and safe/unsafe
ordering. This diagnostic may show whether local action response is learnable;
it cannot validate threshold transfer because only E14 has two-sided support.
New simulation, calibration, correction, QP, denoising guidance, closed loop,
and sealed test access remain forbidden. The next step is one H100 producer
and one dependent independent H100 retraining validator from a clean commit.

Initial producer `40764` is retained as apparatus history. Allocation tests
passed, then extraction stopped before training because the preregistered row
indices assumed the six-entry aggregate trace layout. Candidate artifacts use
the seven-entry link-row layout (`L5[0:3]`, `L6[3:5]`, `L7[5:7]`). Correct
only this schema binding and restart both jobs from a clean commit; data,
targets, splits, model, and scientific interpretation remain unchanged.

Corrected producer `40766` and independent retraining validator `40767`
complete exactly. The held-out result already rejects threshold transfer: 28
false-safes, 6 false-unsafes, zero recall of the six exact-safe E14 actions,
and 0/4 exact-safe selections. Within E14 the risk rank remains strong
(`0.8811`) and all 36 safe/unsafe pairs are ordered correctly, indicating
useful local ordering but a badly transferred state offset. Before finalizing
the root-cause classification, add stored per-fold training metrics to prove
whether the same fixed model fits its training states. This is reporting-only;
all data, outputs, loss, seed, and gates remain unchanged.

Final reporting producer `40768` and independent validator `40769` complete on
H100 with zero frozen-prediction and zero independent-retraining error. The
strict result is NO-GO for unseen-state future-risk prediction. Across 47
known prevention candidates, leave-one-state-out global RMSE is `0.482336`
dimensionless, near-boundary RMSE is `0.403215`, false-safes are `28`, all six
exact-safe actions are rejected, and no held-out state yields an exact-safe
selected action. The three row RMSE values are `0.613482`, `0.510484`, and
`1.011665`.

This is not an optimization-capacity failure on familiar states. Per-fold
training global RMSE ranges only `0.005874--0.016806`, training rank
correlation is `0.9789--0.9977`, and training improvement-direction accuracy
is `0.875--0.939`. On held-out E05 and E14, rank correlation remains
`0.9273/0.8811` and improvement direction is `0.9/1.0`; nevertheless the
absolute offset is wrong, making all E05 candidates falsely safe and all six
safe E14 candidates falsely unsafe. E03 transfer also loses direction
(`0.4167`) and rank (`0.2198`).

Therefore the 9D endpoint representation learns local action ordering but is
not sufficient, with four independent prevention states, to identify the
absolute controller-conditioned L5 boundary. The evidence supports a combined
root cause: too few independent state-level boundaries and missing direct L5
configuration/relative-geometry context in the 9D EE-only state descriptor.
Do not add a conservative buffer: it would worsen the already zero safe recall.
Do not enable correction. The next useful gate is a no-training alias/support
audit followed by additional independent two-sided states; only then compare
9D against the smallest direct-L5 relative-geometry addition.

Final result/validation file SHA-256 values are
`874c49a7d5e98cde2762b1b6ab02f8e76231421160f1d55adaafddf2a0d6a2c3`
and `59d0ea5568a629931dceda1df24699ea849db0a9d4f24bd9b31e3f6bd1690b2b`;
payload SHA-256 values are
`a64b1c8f2c6371fbf762b00dc64a744a63291cc6891c77d0dffc50d19703597d`
and `edd95278c9e23918aa8544ac924d540090b8944ab0a35644f5581cc33f7a4dbd`.

## Generic L5 9D offset/alias audit (preregistered, 2026-08-15)

Before new simulation or training, audit the frozen `40768/40769` predictions
to separate an incorrect state-dependent nominal-risk offset from an
incorrect action-response model. For each held-out state, construct the
privileged diagnostic

`Q_anchor(A) = Q_exact(A_nominal) + Q_hat(A) - Q_hat(A_nominal)`.

Compare anchored and original RMSE, false-safes, safe recall, and exact-safe
selection. Also compare pairwise state distances in the 9D endpoint input,
direct L5 primitive pose/shape relative to the obstacle, and `q/qdot` against
nominal-risk gaps and exact candidate-response disagreement. With only four
states, pairwise correlations are descriptive support/alias evidence, not a
generalization claim.

The preregistered offset-dominant rule requires anchored RMSE at most half of
the original, fewer false-safes, and improved safe recall. A pass authorizes
only targeted collection of independent two-sided states followed by a
matched 9D versus minimal-direct-L5 input ablation. A failure means the
action-response representation or coverage also fails. The exact anchor is
forbidden online; correction, calibration, QP, denoising, closed loop, and
sealed-test access remain blocked. One H100 audit plus a dependent independent
validator are the next allocation-backed jobs.

H100 audit `40772` and independent validator `40773` reproduce the audit
exactly and pass the offset-dominant diagnostic. Oracle nominal anchoring cuts
global RMSE from `0.482336` to `0.107334` dimensionless (ratio `0.222529`),
near-boundary RMSE from `0.403215` to `0.092133`, false-safes from `28` to
`8`, and false-unsafes from `6` to `0`; safe recall rises from `0/6` to
`6/6`. This proves that incorrect state-conditioned nominal-risk offset is
the largest measured component of the frozen 9D failure.

Anchoring is not a learned solution. Eight unsafe candidates remain falsely
safe, only the one recoverable held-out state yields an exact-safe selection,
and E03 retains near-random anchored rank (`0.04396`). Pairwise distances and
correlations cover only four states and are descriptive; they do not isolate
one missing feature. The action response therefore still needs independent
coverage. Proceed to targeted, grouped, two-sided exact-L5 boundary collection
and only then run the matched 9D versus minimal direct-L5-context ablation.
Calibration, correction, QP, denoising, closed loop, and sealed tests remain
blocked. Result/validation file SHA-256 values are
`99d7f27da2733041ce85ff4b5e6d32cb0e8c1e94d0d13dd5223208257d4c801d`
and `52774507eff7ec7127f13a08545287243f4b6af434fa5f4fd4bc00d2f220ce7e`;
payload SHA-256 values are
`c49624c96765ac6d37ce54520337a254439df9c0d2d675c6c84920826005fea8`
and `0f599c0d160f54b6dc9d4699e3a9540123eb2f1b51a0599b7e0d3ab49d006557`.

## Earlier-query L5 offset-coverage canary (preregistered, 2026-08-15)

Test whether intervention timing, rather than more actions at the four old
states, supplies the missing independent boundaries. Freeze five new
development episodes from five task/level groups with observed L5 contact and
select real VLA query boundaries `15--18` actions before first contact. Each
state receives the unchanged 13-candidate nominal plus symmetric world-axis
bank at radii `0.5/1.5`, direct no-QP Cartesian execution through the unchanged
OSC, and the exact compiled-box three-row L5 candidate-plus-complete-backup
target. Original source replay, initially positive exact L5 slack, contacts,
CAR, and timeout censoring remain mandatory.

This is a targeted coverage canary, not training. Require apparatus validity,
zero represented-geometry physical false-safes, and two-sided exact L5 support
in at least `3/5` independent task/level groups. All five two-sided states
would pass the stricter same-bank gate; `3/5` only authorizes a broader grouped
collection using the same earlier-query rule. MLP training, feature comparison,
candidate selection/execution, calibration, QP, denoising, closed loop, sealed
tests, deployment, and formal safety claims remain forbidden.

Initial producer array `40776` is retained as apparatus history. Three early
tasks completed, one task rejected before candidate simulation because its
archived diagnostic obstacle-MVEE rotation was orthogonal within tolerance but
not strict enough for the geometry constructor, and the final task was still
running when the failure was diagnosed. Canonicalize the archived MVEE basis
to its nearest proper rotation before constructing the released-proxy
comparator, as already validated in the compiled-geometry audit. The exact
compiled-box target does not use this orientation. Require a clean-commit full
restart; no state, action, candidate, target, or gate changes.

The incomplete producer outputs are diagnostic only: E02 at step 210 is
already exact-target unsafe, while E10 step 140 and E22 step 15 are one-sided
safe. This suggests the useful boundary lies between registered query times,
but no coverage verdict is allowed until the clean independent replay finishes.

## L5 query-time localization canary (preregistered, 2026-08-15)

Conditioned on exact independent completion of ADR-0164, change only the
development query boundary using its one-sided classification: move initially
unsafe E02 from step `210` to `200`; move safe-only E10/E22/E21 from
`140/15/90` to `150/25/100`; retain two-sided E00 at step `60`. Preserve the
same five episodes, 13 generic candidates, direct no-QP OSC execution, exact
compiled-box L5 candidate-plus-backup targets, physical checks, and timeout
censoring.

Require all five initial states to be exact-target positive/contact-free, zero
physical false-safes, exact independent replay, and at least `3/5` two-sided
states. Passing closes the action-controllability/data-identifiability gate and
authorizes only a larger grouped development collection with prospectively
sealed episodes. Failure rejects fixed-query localization for this candidate
family and requires either a within-query adaptive boundary search or a study
redesign. Training, correction, QP, denoising, closed loop, and test access
remain blocked.

## PNCBF-inspired action-boundary trajectory-value canary (preregistered, 2026-08-15)

Review of the official PNCBF implementation isolates a missing data-design
property rather than a new collision target: it samples many initial and
random interior control-time states from complete fixed-policy trajectories.
The current L5 diagnostic instead has many actions at few query snapshots.

Conditioned on ADR-0165, rerun its identical states and candidates with one
additive capture mode. Record complete robot/controller state, exact compiled-
box geometry, action, phase, and hash before every controller action. Internal
MuJoCo substeps remain label authority only. Compute the exact reverse suffix
value `V_j(z_t) = max_{s >= t} -h_j(z_s)` at action boundaries.

Only known `backup` and `terminal_hold` boundaries are eligible value samples.
Prefix interior states are excluded because their state does not identify the
remaining open-loop chunk. Timeouts stay censored. Exact Bellman residual must
be zero, contexts complete, replay exact, and ADR-0165 must still pass `3/5`
two-sided states.

After an apparatus pass, compare the same Q-only model with a matched shared
Q-plus-V auxiliary model, batching half query Q records and half random
fixed-policy interior V records with episode/state balancing. This directly
tests whether trajectory-wide state coverage reduces the dominant unseen-state
risk-offset error. Correction and control remain blocked.

H100 producer `40822`, independent replay `40823`, and validator `40824`
complete exactly. The PNCBF-inspired labeling apparatus passes: all 65
candidates reproduce, 936 controller action-boundary contexts are complete,
361 known backup/terminal-hold states are eligible value samples, and the
maximum exact Bellman residual is zero. No represented-geometry physical
false-safe occurs.

The population gate remains NO-GO. E10, E22, and E00 are two-sided, but E02
starts outside the exact L5 safe set (`-0.086143` normalized radial slack) and
E21 is safe-only. Therefore the preregistered all-initially-safe apparatus and
targeted-coverage gate fail despite `3/5` two-sided cases. Preserve E02 as
recovery and E21 as safe-region auxiliary data. The action-boundary value
dataset is valid implementation evidence, but matched Q-plus-V training is not
authorized until prospectively selected independent initially-safe boundary
episodes provide grouped train/validation/test support. Validation file SHA-256
is `3789e2e5bc199fc1013ac41f41d85ba4579bff20f3adc79f518ea07ee7b30cae`.

## Matched direct-L5 Q versus Q-plus-V diagnostic (preregistered, 2026-08-16)

The user authorizes one diagnostic exception on the immutable ADR-0166
artifacts. It asks a single causal question: does exact interior fixed-policy
value supervision reduce the dominant unseen-state future-risk offset?

Freeze E10/E22/E00 as leave-one-two-sided-group-out folds, add E21 only to each
training fold as safe-region auxiliary data, and retain E02 only as an
initially-unsafe recovery diagnostic. Unknown timeouts remain censored. The
matched arms have identical state encoder, Q and V heads, optimizer, `1e-4`
weight decay, seed, schedule, normalization, and exact query labels. The only
changed factor is value-loss weight: zero for Q-only and one for Q-plus-V.
Normalization is fit from query training groups only in both arms; Q and V
losses are separately state-balanced.

The 15D direct-L5 state feature contains the current closest exact L5 primitive
center relative to the compiled-obstacle AABB, primitive shape, obstacle size,
all three current L5 slacks, and active-row identity. The action feature is the
metric XYZ endpoint displacement of the complete five-action chunk. This is a
development diagnostic, not a final representation claim.

Require Q-plus-V leave-one-group-out RMSE at most 75% of Q-only, strictly fewer
false-safes, and no loss of safe-support states. Even a pass cannot authorize
action selection or control because there are only three two-sided episode
groups and no untouched two-sided test population. H100 producer plus
independent deterministic retraining validation are the next jobs.

Producer `40852` ran on H100 but stopped in allocation preflight before any
training. `init.sh` used `/usr/bin/python3`, causing 20 pre-existing NumPy
imports to fail even though the registered AEGIS Python is used by the actual
training command. Validator `40853` was canceled after its dependency became
impossible. The apparatus-only repair binds the full gate to the registered
AEGIS Python path and leaves the scientific protocol unchanged.

Replacement `40854` ran 522 repository tests in the registered environment;
the sole remaining error was the unrelated five-action-detour test's optional
`cvxpy` import. No training executed, and validator `40855` was canceled. The
final apparatus repair uses the same scoped byte-compilation and
experiment-specific unit preflight as the earlier validated training jobs.
Scientific inputs, targets, arms, and decision rules remain frozen.

Scoped producer `40857` then stopped before training because its direct module
test name assumed `tests` was a package; validator `40858` was canceled. Bind
both jobs to `unittest discover` with the same single test filename, matching
the repository's existing Slurm scripts. This is test-loader apparatus only.

Producer `40859` passed the scoped allocation tests but called the repository
git-identity helper with an outdated signature and stopped before loading any
artifact or training. Validator `40860` was canceled. The one-line repair
passes the already registered expected commit to that helper; no scientific
setting changes.

Clean H100 producer `40861` and validator `40862` complete with zero frozen
prediction and zero independent-retraining discrepancy. The matched result is
strict NO-GO for the trajectory-value auxiliary:

- Q-only: RMSE `0.226469`, seven false-safes, zero false-unsafes, safe recall
  `21/21`, support `3/3`.
- Q-plus-V: RMSE `0.291339`, seven false-safes, zero false-unsafes, safe recall
  `21/21`, support `3/3`.
- Q-plus-V increases RMSE by `28.64%` and shifts mean signed error from
  `-0.075153` to `-0.211084`, making unseen predictions more optimistic.
- Familiar-group Q RMSE remains `0.00406--0.00904` across both arms, confirming
  capacity to fit the available training groups.

Therefore, adding 361 correlated backup/hold value labels does not create the
missing independent state-level information. It worsens the offset without
changing threshold errors. Do not tune the V loss, add correction, or enable a
QP. The next valid data action is prospective independent initially-safe,
two-sided query-state collection with untouched episode groups; only after
that should a prediction model be tested again. Result/validation file hashes
are `f6c8807e16e70f8069f888688052c4ff00cc3f638d4d33ef515b957de8a2a7fa`
and `44a684098c109394f4968fc3047dbfec6fd5e614fff9027e4353f91693af1590`.

ADR-0168 preregisters the prospective Q-only data gate. Ten previously unused
L5-contact episodes are frozen into six train, two validation, and two test
episode groups before any candidate outcomes are generated. Each episode uses
the deterministic real-query boundary
`floor((first_L5_contact_step-5)/5)*5`; outcome-driven query retiming and split
movement are forbidden. The exact compiled-box L5 target, 13 symmetric direct
Cartesian candidates, unchanged OSC, complete fixed continuation, contact/CAR
checks, and timeout censoring remain identical to ADR-0164--0166, while V
supervision, candidate correction, and QP are removed. Q-only prediction may
run only if all ten frozen states are initially positive/contact-free and the
split has at least 4/2/2 independently two-sided train/validation/test states
with zero represented-geometry physical false-safes. Validation/test episodes
remain excluded from normalization and fitting; test is used once after the
Q-only model and checkpoint rule are frozen.

## Prospective L5 Q-only population gate (validated, 2026-08-16)

H100 producer array `40889`, independent replay array `40900`, and validator
`40902` completed from immutable commit
`e9a81257a6b9e8853b59a87b873ba4ad9927b595`. All ten frozen states are
initially exact-target safe and contact-free, all 130 candidates are retained,
the exact compiled-box target has zero represented-geometry physical
false-safes, and independent replay reproduces every scientific result with
zero mismatched cases. The collection apparatus therefore passes.

The prospective population is nevertheless a strict scientific NO-GO for
Q-only training. Two-sided state coverage is `3/6` train, `1/2` validation,
and `1/2` test versus the preregistered `4/2/2` minimum. Known
safe/unsafe/unknown candidate counts are `32/7/39`, `14/3/9`, and `5/12/9`
for train, validation, and test respectively. The remaining five states are
safe-only under the registered bank, so their action samples cannot identify
the safe/unsafe threshold at those states.

Do not train the Q-only MLP, tune V, add correction, enable a QP, denoise, or
run closed loop from this population. Preserve every producer and replay JSON
as an immutable progressive shard. The next valid data action is a separately
preregistered prospective extension that adds independent initially-safe
states and/or revises the generic candidate bank while keeping existing
validation/test outcomes out of fitting and normalization. Validation
file/payload SHA-256 values are
`f8339258611bb1c89678beb56f46fa9f1e800c816b944f4329640eb9014e29ed` and
`55dc45efc1a44bd6b54eb86ce64ada5b72bdaeedcba399455be291f4bb6132c0`.

## Prospective Q-only diagnostic exception (preregistered, 2026-08-16)

The user authorizes one training diagnostic without relaxing ADR-0168's
population NO-GO. Train the previously best 9D relative-endpoint three-output
L5 MLP once on all known candidates from the six frozen training groups.
Timeouts remain censored, training states are equally weighted, and input and
target normalization use training states only. Architecture, `1e-4` weight
decay, seed, optimizer, loss, and 2,000-epoch final checkpoint are frozen from
ADR-0162; validation cannot select a checkpoint or hyperparameter. Evaluate
the two frozen test groups exactly once only after the final model is fixed.

This run asks whether the prospective natural-state population produces any
unseen-state transfer signal. Report train/validation/test row and global
RMSE, near-boundary RMSE, false-safes, safe recall, state support, ordering,
and improvement direction. A diagnostic pass requires zero validation and
test false-safes, at least 90% safe recall, support in both states of each
split, and near-boundary RMSE at most 0.1 dimensionless. Regardless of result,
the failed `4/2/2` population gate means correction, calibration, QP,
denoising, closed loop, deployment, and a CBF claim remain forbidden.

H100 producer `40920` and independent retraining validator `40921` complete
exactly from commit `9325907685b5fbd338bd67bd53f5f5e0dd95784b` with zero frozen
prediction and zero independent-retraining discrepancy. The diagnostic is
strict NO-GO:

- Training: global/near-boundary RMSE `0.022422/0.016691`, zero false-safes,
  safe recall `1.0`, support `6/6`, rank `0.98246`.
- Validation: `0.037133/0.065182`, zero false-safes, safe recall `1.0`,
  support `2/2`, rank `0.83455`.
- Test: `1.680067/1.750582`, 12 false-safes, one false-unsafe, safe recall
  `0.8`, rank `-0.67892`, and improvement-direction accuracy `0.25`.

The sole two-sided validation state E37 belongs to the Goal-II task-2 family
that supplies all three two-sided training states E24/E29/E32. The sole
two-sided test state E15 belongs to Spatial-I task-3; there the model predicts
all 13 candidates safe although only one is safe, selects unsafe nominal, and
ranks the exact safe `y_pos_r1.50` response incorrectly. Thus an episode-level
split alone did not create task-level boundary-mechanism generalization. The
9D endpoint representation also omits direct L5/controller configuration that
can distinguish these responses.

Do not tune this model on E15, reuse test for checkpoint selection, or add a
conservative buffer that would hide its wrong ranking. The next defensible
gate requires two-sided training boundaries across multiple task/level groups
and a matched 9D versus minimal direct-L5-context representation comparison
with new untouched validation/test groups. Correction, QP, denoising, and
closed loop remain blocked. Result/validation file SHA-256 values are
`ede647acee5d14d657b4a17f7b1b794a880c521ad9c4ab0250678881ef1e16fc` and
`44f2231477e0324df1261dd153a3c5b431a30602d9dc339eb45f55f7a80cf9d7`;
payload SHA-256 values are
`cdea244eb9d56be01b9ecad94adfcae9392519db46ce20a48323c173df9264f0` and
`2b82bf1d0cd4ecbd67adbd8eb12286b14236704154deb8f73cd30d20529b4851`.

## Matched L5 context root-cause ablation (preregistered, 2026-08-16)

ADR-0170 freezes every ADR-0169 sample, target, split, model width, loss,
optimizer, seed, schedule, weight decay, and train-only normalization. It
compares the reproduced 9D relative-endpoint input against one causal 33D
input that adds only current exact three-row L5 geometry and OSC position
error. Stale released-proxy normals and all future executed states are
excluded.

Because E15 has already been opened, this is a root-cause ablation rather
than new generalization evidence. A material representation effect requires
at least 25% lower test RMSE, fewer test false-safes, better test improvement
direction, and no support loss. The original prediction gate remains zero
false-safes, at least 90% safe recall, 2/2 support, and near-boundary RMSE at
most 0.1 on validation and test. Correction, QP, calibration, denoising,
closed loop, deployment, and CBF claims remain blocked regardless of result.

H100 producer `40926` and independent validator `40927` completed ADR-0170
from immutable commit `9619110dc9d30e1d03da00b9b221d355b9a0e4ab` with zero
frozen-prediction and zero independent-retraining discrepancy. The exact 9D
baseline is reproduced. Adding causal current L5/OSC context lowers opened-test
global RMSE `1.680067 -> 0.936041`, lowers near-boundary RMSE
`1.750582 -> 1.059791`, and improves direction accuracy `0.25 -> 0.5833`.

The verdict is nevertheless strict prediction NO-GO: false-safes stay at
`12`, all of them in E15; all 13 E15 actions are predicted safe although only
`y_pos_r1.50` is exactly safe, and the inferred minimum-intervention action is
still unsafe nominal. Validation remains zero-false-safe but its RMSE worsens
`0.037133 -> 0.073641`. All held-out samples have at least one coordinate
outside the training range, and 25.85% of 33D test feature elements are
outside it. Direct context is useful but cannot substitute for independent
two-sided task-family boundary data. New prospective Spatial-I-like and other
task-level boundaries with untouched validation/test groups are required
before another prediction gate; correction and QP remain blocked.

Result/validation file SHA-256 values are
`10976f191c1471199e8882637ba85e2364d899980415fd774865bcf40f776479`
and `44a051c9e115c0bbec10645f8bc26b47af084a17b509a688837af0501502acb0`;
payload SHA-256 values are
`ea8ff0f63e219dc301c98b2e8fbfec9186348bb12dd3174aad034c50515f6e06`
and `e84bd4c1d0b65b489b3922cdcd8712c33c6f69a649367c526eede6340462d9e7`.

## Spatial-I task-3 progressive boundary extension (preregistered, 2026-08-16)

ADR-0171 adds seven immutable episode shards without replacing the existing
prospective dataset. Previously opened E00, E03, and E15 are training-only.
Candidate-outcome-untouched E01/E04 are frozen as validation and E02/E13 as
test before the new bank runs. All are Spatial-I task 3; the archived active
obstacle binding is preserved per episode rather than forced to the training
episodes' wine-bottle obstacle. The three contact episodes use the last complete
five-action query boundary strictly before archived L5 contact; untouched
holdouts use the common preregistered action-60 boundary.

The exact compiled-box target, direct unchanged OSC execution, complete fixed
continuation, 13 symmetric Cartesian candidates, timeout censoring, and
contact/CAR authority remain unchanged. All seven states must be initially
safe and independently reproduced, with two-sided action support in `3/3`
train, `2/2` validation, and `2/2` test and zero physical false-safes. Only
that data gate authorizes matched 9D/33D Q-only retraining. QP, correction,
calibration, denoising, and closed loop remain blocked.

Immutable H100 producer array `40930` and independent replay array `40931`
were submitted from clean detached commit
`7c98b333611f39d76f1409953b84fc85c76cf7ae`; one task from each array is
running on worker-2 with `%1` array concurrency. The first full source-bundle
transfer was truncated and its v1 clone is retained as apparatus history;
the experiment uses the verified incremental bundle and clean v2 source.
Validator submission is pending only because the live QOS submit-count limit
was reached; it must be added with `afterok:40930:40931` when capacity opens.

Replacement arrays `40947`/`40948` preserved the three original training
shards and completed E01, E04, and E13 independently. E02 alone was rejected
before candidate simulation because the reused E05 evaluator required the
archived VLA episode to have native task success. That precondition is absent
from ADR-0171: this is a state/action future-risk dataset, and the registered
gate already requires an initially safe, contact-free, reproducible query
state. The six completed shards remain immutable. The repair makes archived
task success configurable, freezes it to `false` only for this progressive
data gate, and retains the exact state, action, continuation, OSC, geometry,
contact, CAR, timeout, and split requirements. Historical per-case source
commits are bound explicitly so progressive shards are validated rather than
overwritten.

The PNCBF-aligned learning implementation is prepared but not yet authorized
to run. It treats OSC, robot dynamics, contacts, and the fixed continuation as
the plant and extracts exact controller-boundary targets

\[
V_j^\pi(z_t)=\max_{s\ge t}c_j(z_s),\qquad
Q_j^\pi(z,A)=\max\{c_j\text{ during the prefix},V_j^\pi(z^+)\}.
\]

The query target is the first exact Bellman record, not a joint trajectory or
one-step OSC prediction. Only backup/terminal-hold controller boundaries are
eligible state-value samples; prefix states, internal substeps, and timeouts
are not promoted to independent training states. The constraint-conditioned
trainer compares matched Q-only and shared Q-plus-V arms with train-only
normalization and candidate-level false-safe/support gates. It cannot run
until an independently replayed source contains trajectory contexts and
adequate two-sided episode coverage. CPU MuJoCo remains label authority; no
GPU simulator output is admitted without a separate exact equivalence canary.

Replacement E02 producer `40960`, independent replay `40961`, and validator
`40962` completed from immutable commit
`2c0995b5a6f2eb0cfa3dad99b624e07da351d63f`. All seven progressive shards
replay exactly and retain zero represented-geometry physical false-safes, but
ADR-0171 is a strict data NO-GO. Two training states begin inside the exact L5
violation set (E00 `-0.255325`, E15 `-0.145699`), and no split contains a
two-sided state. Train candidates are `0 safe / 39 unsafe`; validation has
`4 safe / 0 unsafe / 22 unknown`; test has `26 safe / 0 unsafe`. E03 is
initially safe but all 13 candidates are unsafe; E01 is entirely timeout;
E04 is safe/timeout; E02 and E13 are safe-only. Consequently the frozen bank
does not identify an L5 decision boundary in this task family, and matched
9D/33D Q-only training is not authorized. Preserve every shard and failure;
the next scientific change must target state timing or candidate excitation,
not model tuning. Validation file/payload SHA-256 values are
`cc16e7dcaa791b47eb3a35930f83bd381daaa32a18c6c5c9c2262564aeaadc0d`
and `f6c15c2a6161473c77caddbcbb9743d30dcbf78d7906e4c55e3b7165cddc391d`.

## Spatial-I task-3 timing-localization pilot (preregistered, 2026-08-16)

ADR-0174 isolates timing before changing the candidate family. It reuses only
the three opened development episodes E00/E03/E15 and shifts each registered
query exactly one complete five-action chunk earlier: `70->65`, `60->55`, and
`70->65`. The exact compiled-box target, 13 symmetric world-axis candidates,
direct no-QP Cartesian execution, unchanged OSC, complete fixed continuation,
timeout censoring, and physical contact/CAR audit are unchanged. No validation
or test episode is accessed.

This single-factor gate asks whether ADR-0171 failed because intervention was
one VLA query too late. A mechanism pass requires all three new query states to
be initially exact-L5 safe, exact independent replay, zero represented-geometry
physical false-safes, and safe/unsafe candidate support in `3/3`. A pass freezes
this warning-offset rule before prospective episode-grouped collection. A
failure authorizes only a separate broader candidate-excitation pilot; it does
not authorize MLP training, Q/V tuning, correction, QP, denoising, closed loop,
GPU labels, or a safety/CBF claim.

Immutable H100 producer array `40977`, independent replay array `40978`, and
dependent validator `40979` were submitted from clean detached commit
`9f60bda64759dc873c95c76cce827dc590c406b6`. Each array is capped at one GPU,
so at most two H100 equivalents run concurrently. Artifacts are written to
`/mnt/data/quanth/experiments/vlsa-distal-spatial-t3-timing-localization/spatial-t3-timing-localization-20260816a`.
The VinUni login node is used only for Slurm and read-only inspection.

Arrays `40977`/`40978` rejected all cases before candidate simulation because
the development manifest added timing suffixes to immutable archived case IDs;
validator `40979` therefore became dependency-impossible. The exact logs show
`archived query-risk case differs`. Preserve the failed root as apparatus
history. The repair restores original case IDs only and requires a new clean
commit/root; query steps, candidate bank, labels, OSC, continuation, and gates
remain frozen.

Clean replacement producer `40984`, independent replay `40985`, and validator
`40986` run only the repaired manifest from immutable commit
`aa411cb560bc3f43ec53e95a0edfb6337f166d81` and new artifact root
`/mnt/data/quanth/experiments/vlsa-distal-spatial-t3-timing-localization/spatial-t3-timing-localization-20260816b`.

Replacement jobs `40984`/`40985` and validator `40986` completed exactly.
Moving one VLA query earlier fixed initial-state validity in all `3/3` cases
and produced two-sided boundaries in E03 (`3 safe / 10 unsafe`) and E15
(`1 safe / 12 unsafe`). E00 remained one-sided with `0 safe / 12 unsafe / 1
unknown`, despite positive initial slack `0.305562`. E03/E15 initial slacks were
`1.193103`/`0.796033`. Independent replay is exact, all 39 candidates are
retained, and represented-geometry physical false-safes remain zero.

The strict `3/3` gate therefore fails at `2/3`. Timing was a real cause—it
removed both initially unsafe states and recovered two boundaries—but timing
alone is insufficient for E00. Preserve these validated labels. The next
single-factor experiment may broaden spatial/temporal candidate excitation only
at opened E00 step 65; Q-only training and all control remain blocked.
Validation file/payload SHA-256 values are
`5aea93705ac2fa44a8df79536ddb9e8428cd196bf00f9d3a1785319641d11212`
and `c41f697cb6458065b849401b2294123a82cbae8191c556370d425b8acf7f5ed8`.

## Spatial-I E00 candidate-excitation pilot (preregistered, 2026-08-16)

ADR-0175 holds opened E00 at the validated initially-safe step 65 and changes
only candidate excitation. The former 13 symmetric world-axis candidates are
replaced by nominal plus all 26 normalized directions in the local
L5--obstacle frame spanned by outward normal, upward tangent, and side tangent.
Every correction uses the registered front-loaded five-action profile with
requested L2 norm 2.0. Rotation/gripper, direct no-QP Cartesian execution,
unchanged OSC, exact compiled-box candidate-plus-complete-continuation target,
timeouts, contacts, and CAR remain frozen.

The mechanism gate asks only whether the registered family contains at least
one exact-safe and one unsafe E00 candidate with positive initial L5 slack,
exact independent replay, and zero represented-geometry physical false-safes.
It uses no validation/test episode and cannot authorize training directly. A
pass freezes the timing and candidate procedure for prospective grouped
collection; a failure shows that this five-action action region still lacks
safe support and requires revisiting intervention horizon or timing before
learning. Q/V training, correction, QP, denoising, closed loop, and GPU labels
remain blocked.

Initial producer/replay jobs `40995`/`40996` passed all allocation tests and
generated the candidate rollouts, then both stopped before exact aggregation
because the reused compiled-box auditor requires the legacy receipt alias
`requested_alpha`, while the new grid emitted only the equivalent
`requested_correction_l2_action`. No scientific artifact was accepted and
validator `40997` is dependency-impossible. The apparatus-only repair adds the
alias (`0` for nominal, `2.0` for every grid candidate) without changing any
action, state, target, rollout, or gate. The retry must use a new clean commit
and artifact root.

Clean replacement producer `40999`, independent replay `41000`, and validator
`41001` completed from immutable commit
`1676717c7bd7e8bacd41a8c6ee596b67e2aa8b52`. ADR-0175 passes its targeted
mechanism gate. E00 step 65 remains initially exact-L5 safe with slack
`0.305562`; among 27 candidates, 25 are known, four are exact-safe, 21 are
unsafe, two are unknown timeouts, and four lie near the registered boundary.
Independent replay and state hashes are exact, physical false-safes are zero,
and the apparatus authorizes same-bank grouped collection. Nineteen of 26
non-nominal proposals clip, so requested and effective correction norms must
remain separate inputs/records.

This is candidate-support evidence, not training authorization. Freeze the
step-65 timing and local normal/tangent grid before prospectively splitting new
episode groups. Validation file/payload SHA-256 values are
`4421a5097e7b65cca2c1da1e4983096c83566849bd5da4a946d0ee794cc18c96`
and `20d71b81f2d7335c7974d26e0a69f15766e09cf444d35b825e8a144183438f68`.

### 2026-08-16 — freeze the whole-body superset artifact before population collection

The validated ADR-0175 artifact already stores exact tight-palm future-risk
traces and initial EE/joint/OSC context, but it does not store an explicit
released-EE future-risk head or EE/palm poses at every internal MuJoCo
substep. Those missing quantities cannot be recovered exactly from the saved
L5 scalar traces alone.

ADR-0176 therefore changes only the artifact schema before the expensive
prospective population is launched. The registered E00 step, 27 candidates,
direct clipped Cartesian execution, unchanged OSC, fixed continuation,
timeouts, contacts, and CAR remain fixed. Every candidate now records separate
future violations for `end_effector`, `palm`, `L5`, `L6`, and `L7`; EE and palm
poses at every internal substep; and complete joint/controller/exact-geometry
contexts at action boundaries. The released EE proxy remains a diagnostic
AEGIS representation, while raw MuJoCo palm/L5--L7 contacts and CAR remain the
physical authorities.

Existing validated shards remain immutable L5/palm evidence and are not
overwritten or falsely relabeled as EE-complete. Run one independent E00
allocation-backed schema canary, then use the passing superset contract for all
new grouped shards so later EE, palm, L5, L6, L7, or policy-value ablations do
not require repeating those rollouts. Training and control remain blocked.

Initial superset producer/replay `41008`/`41009` are apparatus history only.
Both passed all scoped tests and stopped before candidate simulation because
the modern MuJoCo `MjData` exposes body rotation through indexed `xmat`, not
the legacy `get_body_xmat` convenience method. Validator `41010` is
dependency-impossible. Replace only that API access with the equivalent
site-body `xmat`/`xquat` arrays, keep every scientific setting fixed, and use a
new immutable commit and artifact root.

Replacement producer/replay `41011`/`41012` are also apparatus history only.
Both passed all scoped tests and stopped before candidate simulation because
the whole-body auditor imported `_eef_site_id` from
`main.evaluate_safelibero_aegis`, although the established helper is owned by
`main.multilink_ellipsoid.shadow`. Validator `41013` is dependency-impossible.
The repair changes only that import binding and adds a structural regression;
state, candidates, execution, geometry, labels, and all gates remain frozen.

Clean replacement producer `41016`, independent replay `41017`, and validator
`41018` complete from immutable commit
`6a17b99599428494b1ac654117fed24d9f5ca6f4`. ADR-0176 passes its additive
schema gate. The canary preserves E00 step 65 and all 27 registered candidates,
including `4 safe / 21 unsafe / 2 unknown` and four near-boundary candidates.
Producer/replay scientific content is exact, the initial L5 slack remains
`0.305562`, and represented-geometry physical false-safes remain zero.

The artifact now records all five named future-risk groups
`end_effector/palm/L5/L6/L7`, nine initial robot primitives, complete internal
substep EE/palm poses, and complete action-boundary joint/OSC/exact-geometry
contexts. It contains 299 action boundaries and 144 eligible known
backup/terminal-hold value states with zero Bellman residual. Freeze this
schema for prospective grouped collection. This is apparatus authorization,
not MLP training or control authorization. Validation file/payload SHA-256
values are `1d194577aad3cd07ee28961ab90361b7f1d151d5fbf8ef23137915469a808b18`
and `52cb2d67775788e07e92d30422b3fa494eaa100b58a5d67452f9ab506f9eea96`.

## Whole-body prospective Q-only coverage cohort (preregistered, 2026-08-16)

ADR-0177 freezes the passing ADR-0176 superset schema and begins the minimum
independent population needed for the first Q-only prediction gate. Eight
candidate-outcome-untouched archived episode groups are split before outcomes
as `4 train / 2 validation / 2 test`. They span Spatial-I and Goal-II tasks,
milk, moka-pot, and wine-bottle obstacles, and L5- and L6-conditioned warning
states. Each state receives the unchanged nominal-plus-26 local
normal/tangent grid at requested L2 `2.0`, direct clipped Cartesian execution,
unchanged OSC, and the complete fixed continuation.

Every shard stores separate end-effector, palm, L5, L6, and L7 future risks,
all internal-substep physical evidence, and complete action-boundary context.
Candidate outcomes may not retime states or move episodes across splits.
Timeouts remain unknown. Training is authorized only if all eight states are
initially target-safe, independent replay is exact, represented-geometry
physical false-safes are zero, and two-sided target support reaches exactly
`4/2/2`. One-sided and failed cases remain in the audit and trigger targeted
progressive extension rather than model tuning. QP, correction, calibration,
denoising, and closed-loop use remain blocked.

Producer array `41021`, independent replay array `41023`, and validator
`41024` completed all eight ADR-0177 states. The apparatus and artifact gates
pass: all eight states are initially target-safe, producer/replay scientific
content and state hashes agree exactly, represented-geometry physical
false-safes are zero, the whole-body superset and action-boundary context are
complete, and the maximum Bellman residual is zero. All 216 candidates are
retained: 166 have known outcomes (`95 safe / 71 unsafe`) and 50 are unknown
timeouts.

The strict training-coverage gate does not pass. Target-group two-sided
coverage is `2/4 train, 2/2 validation, 2/2 test`, rather than the registered
`4/2/2`. Training E00 is unsafe-only among known candidates (`0 safe / 10
unsafe / 17 unknown`) and training E35 is safe-only (`22 safe / 0 unsafe / 5
unknown`); both remain immutable evidence. This is a boundary-coverage
shortfall, not a replay, geometry, or label failure. Preserve every shard,
derive per-row/per-group and global-safe support from the stored superset
traces without recollection, freeze a development-selected 13-candidate bank,
and add prospectively registered independent states toward the 24-state
checkpoint before Q-only training. Validation file/payload SHA-256 values are
`d4d1d73625b6f95fda47de157d1d27583ecbfcec57a4a010be26fea0174b95c6`
and `82a260e138140f8d371c95f5d947141d13fa6970660bca4e9e717375d50d1c2d`.

## Whole-body per-constraint support audit (preregistered, 2026-08-16)

ADR-0178 derives the missing per-row, per-group, and global-support accounting
from the immutable ADR-0177 producer/replay artifacts. It performs no new
simulation and does not alter or reinterpret an UNKNOWN timeout. For every
known candidate it verifies the stored group future risk against the minimum
of the complete nine-row trace, then reports safe, unsafe, near-boundary,
active-witness, two-sided-state, state, and episode counts for the released EE
proxy, palm, L5, L6, L7, and all nine rows.

The audit reports two distinct global gates. Represented-global support
includes all five stored risk groups. Physical-global support excludes the
diagnostic EE proxy and requires nonpositive palm/L5/L6/L7 risks, no stored
group or raw protected contact, CAR pass, and no physical veto. Independent
producer/replay equality and the immutable ADR-0177 validation hashes are
mandatory. The result may guide development-only candidate-bank and cohort
design, but it cannot authorize training, correction, QP, denoising, or
closed-loop execution.

CPU allocation job `41063` completed ADR-0178 in 30 seconds from immutable
commit `57ac6127fbab1431a69f4bea9f17d6fde67a7519`. It verifies exact
producer/replay equality for all eight states and all 216 candidates, retains
166 known outcomes and 50 UNKNOWN timeouts, and reproduces every stored group
risk from the complete nine-row traces.

The audit shows that the earlier target-group result understated the useful
L5/global evidence but also confirms that the current population is not a
whole-body training set. Strict two-sided group-state counts for
`end_effector/palm/L5/L6/L7` are respectively `1/0/2/0/0` in train,
`0/0/2/1/0` in validation, and `1/0/2/0/0` in test. Palm and L7 are entirely
safe-only with zero near-boundary and zero active-witness candidates; L6 has a
single validation boundary and no training boundary. The diagnostic EE proxy
has limited train/test boundaries but E35 remains proxy-unsafe for every known
candidate despite being physically safe for palm/L5/L6/L7.

Physical-global safe support exists in `3/4 train, 2/2 validation, 2/2 test`
states and physical-global two-sided support in `2/4, 2/2, 2/2` states.
Represented-global safe support, which additionally includes the diagnostic EE
proxy, is only `2/4, 2/2, 2/2`. Therefore the next prospective cohort must
target missing EE, palm, L6, and L7 mechanisms; unsupported heads remain
diagnostic. Audit file/payload SHA-256 values are
`4c9cf8859e6e74d0a3f8724d1cd8ea391388a325e82d2024aea77cf4c5020be7`
and `1eafe2a49289028f97ca5e861cd1d26412cb565ccfd83005becb97f9456cb220`.

## Coverage-preserving 13-candidate bank (preregistered, 2026-08-16)

ADR-0179 reduces the opened 27-candidate grid without using validation or test
outcomes for selection. The bank must contain nominal plus six exact
opposite-direction pairs. Across all four ADR-0177 training states it must
preserve every safe, unsafe, and two-sided group/row/global support property
observed in the full bank, including physical- and represented-global safe
support. Infeasible reductions are reported rather than forced.

Among feasible banks, the deterministic training-only objective first
maximizes known outcomes, then near-boundary samples on supported constraints,
direction covariance determinant, and the effective/requested correction
ratio. Existing validation/test states are evaluated only after the bank is
frozen and remain diagnostic; their labels cannot change the selection. This
is a data-collection efficiency decision, not prediction or control evidence.

CPU allocation job `41064` completes ADR-0179 from immutable commit
`9574f755eaa58b4c6a1b96eae5ee86d6894dc6d9`. Of 1,716 possible choices of
six opposite pairs, 1,171 preserve every registered development support
property. The selected nominal-plus-12 bank retains all training group
two-sided counts (`L5=2`, diagnostic EE `=1`), represented-global safe support
in two states, and physical-global safe support in three states while reducing
known development labels from 74 to 40. It also preserves the opened
validation/test L5, L6, EE, and global support exactly in the diagnostic
post-selection check.

The frozen candidate names are recorded in the selection artifact; selection
uses no validation/test label. Its score is 40 known labels, 19 supported
near-boundary incidences, direction determinant `5.5`, and mean
effective/requested ratio `0.873340494819`. Use this bank unchanged for newly
sealed episode groups. Selection file/payload SHA-256 values are
`ee286ac855d9ed35f6067058b8374e9048099e967e44c2dbd03452e02a9895b9`
and `c376f46fec9e1bcdcf2958aa0a2a90100638e510a6aa7967344d63d04b9739cf`.

## Additive 16-state whole-body extension (preregistered, 2026-08-16)

ADR-0180 freezes 16 additional candidate-outcome-untouched archived episode
groups as `10 train / 2 validation / 4 sealed test`, producing a combined
24-state checkpoint of `16/4/4` when joined with immutable ADR-0177. State
timing is derived only from the earliest raw represented-physical contact in
the archived ledger, never from counterfactual candidate outcomes. Every
state is one real five-action query before that contact and must begin safe
for palm/L5/L6/L7, not merely for its target group.

The cohort targets the supported missing mechanisms exposed by ADR-0178:
three palm, nine L6, and four L5 warning episodes. L7 remains recorded in the
whole-body superset but diagnostic because the 1,600-episode availability
audit found no clean natural L7-contact population; it is not filled with
artificial far-safe samples. The frozen ADR-0179 nominal-plus-12 bank is used
unchanged. This halves per-state rollout cost while preserving all opened
development support properties.

The extension records separate end-effector, palm, L5, L6, and L7 risks,
complete nine-row and controller contexts, raw contacts, CAR, and UNKNOWN
timeouts under direct no-QP Cartesian execution and unchanged OSC. It cannot
authorize training by itself. Only a new combined 24-state audit may authorize
supported per-constraint Q-only heads after exact producer/replay equality,
zero represented-geometry physical false-safes, at least `4/2/2` two-sided
support for every claimed group, and global safe support in every recoverable
validation/test state. One-sided, initially unsafe, timeout, contact/CAR, and
failed cases remain evidence and are never replaced.

Immutable commit `9d6ee2888bc8c287ca0924b06ad951d959e70cc2`
launches H100 producer array `41066`, independent replay array `41067`, and
dependent validator `41068` from clean remote source
`/home/quanth/working_space/vlsa-aegis-wholebody-extension-9d6ee28-v3`.
The artifact root is
`/mnt/data/quanth/experiments/vlsa-distal-whole-body-extension/whole-body-extension-20260816a`.
Both arrays are capped at one live task each, so the workflow uses at most two
H100 equivalents, 16 CPUs, and 128 GB memory. Preserve completed per-case
JSONs as progressive shards and rerun only exact missing/schema-invalid
indices after inspection.

Initial arrays `41066`/`41067` failed indices 0--1 before candidate rollout
and index 2 was canceled; validator `41068` was canceled. All scoped
allocation tests passed. The frozen-subset wrapper passed the registered
13-count into the underlying full-grid generator, whose internal invariant
correctly requires 27 entries before filtering. Preserve the first root as
apparatus history. The repair generates the unchanged 27 definitions under
their native receipt, then selects the already frozen 13 names in order and
asserts a final count of 13. It changes no state, action, split, target, OSC,
continuation, or gate and is covered by an allocation-capable regression.

Replacement arrays `41075`/`41076` and canceled validator `41077` are also
apparatus history with no candidate rollout. The new allocation regression
correctly exposed that JSON generation had serialized the requested norm as
integer `2`, so the native generator emitted receipt suffix `r2` while the
immutable selected names from ADR-0179 use `r2.0`. Restore the exact registered
floating receipt `2.0`; the numerical magnitude and every candidate action are
unchanged. Require the allocation regression to pass before simulation.

Paired case-0 H100 canaries `41108`/`41109` pass in 8:27/8:25 from
commit `e52b79348507a66f7f87b8f502423c44999ea975`. Both write schema-valid
E09 shards with the exact frozen 13 names, all known outcomes, complete
end-effector/palm/L5/L6/L7 superset records, and equal scientific views.
Preserve case 0 and run only missing indices 1--15 as producer `41110` and
independent replay `41111`, each at `%1`, with full validator `41112` after
both arrays. The progressive root remains `whole-body-extension-20260816c`.

Producer `41110`, replay `41111`, and validator `41112` complete successfully.
All 16 producer/replay scientific views agree, all 208 candidates are
retained, the whole-body/controller superset is complete, Bellman residual is
zero, and represented-geometry physical false-safes remain zero. There are 29
UNKNOWN timeouts. The target-group result is intentionally retained as a
scientific NO-GO: two frozen training states are initially unsafe, while
target two-sided support is only `4/10` train, `1/2` validation, and `1/4`
test. These outcomes are data, not apparatus failures, and no state is dropped
or replaced.

ADR-0181 preregisters the required read-only combined audit over immutable
ADR-0177 and ADR-0180 roots. It independently binds each cohort config,
artifact commit, validation file/payload hash, producer/replay pair, and then
recomputes the common per-group/per-row/global support summary across all 24
states. Palm, L5, and L6 are the preregistered physical claim heads;
end-effector and L7 remain diagnostic. Training requires every prevention
state to be initially safe, zero physical false-safes, exact replay/context,
split counts `16/4/4`, per-claimed-group two-sided support `4/2/2`, and global
physical-safe candidates in every initially-safe validation/test state. This
audit performs no fitting and cannot silently discard unsafe or timeout data.

## Train/validation-only whole-body Q diagnostic (preregistered, 2026-08-17)

At the user's explicit request, ADR-0182 permits one capacity/state-transfer
diagnostic before deployment coverage authorization. This does not weaken the
coverage gate and does not access any test artifact. It reads only immutable
ADR-0177/ADR-0180 producer cases assigned to train or validation, verifies the
bound validation hashes and independent replay, retains UNKNOWN timeouts as
censored, and excludes the two initially unsafe recovery states from ordinary
prevention fitting and metrics while keeping their identities in the source
record.

Compare three fixed-final-epoch Q-only arms: the established relative endpoint
9D L5 predictor, the causal direct-L5/OSC 33D predictor, and a shared
constraint-conditioned 135D predictor for palm/L5/L6 rows. The shared feature
contains row-local normalized obstacle geometry, semiaxes and initial slack,
`q/qdot`, OSC position error, exact effective and nominal five-action chunks,
their residual, and a row identity. Every arm uses the same two-layer SiLU
MLP, train-only state/constraint-balanced normalization, symmetric Huber loss,
AdamW weight decay `1e-4`, fixed seed, and 2,000 epochs. Validation labels do
not choose a checkpoint or hyperparameter. Report RMSE, boundary error,
false-safes, safe recall, ordering, and virtual safe support. Test, correction,
QP, calibration, denoising, and closed-loop execution remain blocked.

Initial diagnostic trainer `41251` passes structural tests and fails before
sample construction or fitting because the loader incorrectly requires the
legacy `source_replay_exact` proxy flag. ADR-0180 explicitly classifies that
flag as diagnostic-only: independent producer/replay scientific views are
equal and state hashes are exact. Remove only this redundant loader check;
continue requiring immutable payloads, independent validation, state hashes,
primitive certificates, initial physical safety, and timeout censoring. The
failure produces no result/checkpoint and changes no scientific factor.

Clean source worktree `41255`, H100 trainer `41256`, and independent H100
retrainer/validator `41257` complete from immutable repair commit
`9b03e046e08ed0949a005ab4a77b8f1541d688d1`. The validator reports exact model
and prediction reproduction, no test-artifact access, and no authorization for
training claims, correction, QP, or closed loop. The diagnostic fits 12
eligible train states / 173 known candidates and evaluates four validation
states / 63 known candidates; shared row supervision contains 1,038 train and
378 validation samples.

The diagnostic confirms capacity but rejects state-transfer readiness. The
33D L5/OSC arm is the best L5 value predictor on validation at global RMSE
`0.257113` and near-boundary RMSE `0.195521`, but still has nine false-safes
and its least-modifying selection is exact-safe in only `3/4` states. The 9D
arm has validation RMSE `0.373440`, seven false-safes, and `3/4` selected-safe
states. The shared palm/L5/L6 model fits train RMSE `0.025278`, then degrades to
validation RMSE `0.345089`, 14 global false-safes, safe recall `0.780488`, and
`3/4` selected-safe states. Its validation group RMSE/false-safe counts are
palm `0.513024/0`, L5 `0.366173/5`, and L6 `0.356442/14`; palm is safe-only in
validation, so its zero false-safes do not test a boundary. This is positive
implementation evidence but scientific NO-GO for unseen-state safe selection.

The completed ADR-0181 read-only audit contains 24 root episode groups, 424
candidates, 353 known outcomes, and 71 UNKNOWN timeouts. The frozen split is
actually `14/4/6`, not the planned `16/4/4`; E21 and E02 are initially unsafe
recovery cases. Strict per-group two-sided train/validation/test state counts
are L5 `5/3/4`, L6 `1/2/0`, palm `2/0/0`, diagnostic EE `6/1/3`, and diagnostic
L7 `0/0/0`. Row support localizes further: row 2 has `0/0/0`, row 3 `3/2/3`,
row 4 `3/2/3`, row 5 `1/2/0`, and row 6 `0/0/0`. Global physical-safe support
exists in every validation/test state and physical false-safes remain zero.
The data apparatus and represented geometry therefore pass; the decisive
blocker is per-constraint warning-state/boundary coverage.

## Targeted per-constraint warning-time extension (next protocol)

Preserve every existing safe, unsafe, recovery, one-sided, failure, and timeout
shard. On opened development episodes, scan earlier/later query boundaries only
to localize one frozen warning snapshot per root episode. The selected snapshot
must be initially safe across all represented physical groups and should make
the frozen bank straddle the target constraint boundary. New validation/test
episode identities and target constraints must be frozen before their candidate
outcomes. Count EE, palm, L5, L6, and L7 independently, require physical-global
safe support separately, and keep timeouts UNKNOWN. Prioritize palm and L6
train/test deficits; L5 already passes its preregistered coverage gate. The EE
proxy and L7 remain diagnostic until their geometry/support gates change.

The next step is intentionally narrower than the earlier warning-localizer
draft. First rerun the immutable combined audit with episode-group
classifications for every constraint: initially safe, safe-only, unsafe-only,
two-sided, and timeout-limited. Then freeze only additive palm and L6 root
episodes into train/validation/test before candidate labels. L5 already has
`5/3/4` two-sided support and will not trigger generic recollection; L7 and the
released EE proxy remain diagnostic. The existing 13-bank, exact
candidate-plus-continuation target, unchanged OSC, UNKNOWN timeout semantics,
and complete whole-body superset remain unchanged.

New direct/no-QP data must explicitly bind its nominal to the raw pi0.5
translational five-action chunk. The evaluator now exposes this as an opt-in
nominal source while preserving the historical post-AEGIS default for every
immutable artifact and legacy caller. This prevents “QP disabled” from being
mistaken for “raw pi0.5 nominal” without invalidating prior evidence.

CPU audit `41287` completes the new episode classification from immutable
commit `dac85fd215baacd9a5b60388442a40797b77ed44`. It reads the same 24 root
groups / 424 candidates, retains 353 known outcomes and all 71 UNKNOWN
timeouts, and leaves the scientific gate NO-GO. Exact two-sided episode counts
remain L5 `5/3/4`, L6 `1/2/0`, palm `2/0/0`, diagnostic EE `6/1/3`, and
diagnostic L7 `0/0/0` over train/validation/test. The new classifications show:

- palm: train `13` initially safe (`11` safe-only, `2` two-sided, `7`
  timeout-limited), validation `4/4` safe-only, test `6/6` safe-only;
- L6: train `14` initially safe (`13` safe-only, `1` two-sided, `7`
  timeout-limited), validation `2` safe-only / `2` two-sided, test `6/6`
  safe-only;
- L5 already passes and must not trigger additive collection.

The apparatus, exact replay/state hashes, complete contexts, zero physical
false-safes, and global holdout safe support all remain passing. The immutable
audit artifact SHA-256 is
`74a311dd632c822ba8004a7aeadb99306c813624e9b598735fc91641e2c5dea4`.
The next frozen cohort therefore targets only the missing palm `+2/+2/+2`
and L6 `+3/+0/+2` two-sided root-episode deficits, with extra prospective
groups allowed for one-sided yield but never selected after labels.

## Natural-pi0.5 palm/L6 source audit (in progress, 2026-08-17)

ADR-0185 implements a separate read-only source audit over all 1,600 immutable
natural `pi05_translational` Table-1 arms. It excludes every case already named
by a committed counterfactual manifest, requires the raw action ledger to have
zero correction and no QP, maps raw compiled contact geoms, and registers at
most one warning boundary per root episode. Candidate outcomes are not read.

The natural pi0.5 result is the simulator-state and nominal-action authority.
Its paired AEGIS result supplies only the frozen obstacle perception used by
the existing backup policy and diagnostic released-EE proxy; it is not an
action or state source. The exact palm/L5/L6/L7 labels remain compiled-box
targets. After the allocation-backed audit, eligible palm/L6 identities and
strict splits will be committed before any 13-bank rollout is launched.

The user's final gate is explicit: Q-only training/evaluation proceeds even if
the new combined `4/2/2` coverage gate fails, but such a result is diagnostic.
Finite-bank action correction remains forbidden unless both coverage and
held-out prediction pass.

Immutable source-audit commit `457cbd15eda2e501521e14c9a448dc7cd815a9ff`
is pushed. Source-prep job `41291` created the correct clean worktree but its
wrapper failed only because the submitted expected commit was abbreviated;
retain it as apparatus history. Independent CPU audits `41292` and `41293`
now scan the 1,600 natural pi0.5 results into separate immutable outputs.

The collector apparatus is extended only by an opt-in perception binding for
natural pi0.5: replay state and nominal actions remain from the pi0.5 result,
while the paired AEGIS perception is explicitly provenance-bound as backup and
diagnostic-EE geometry only. Legacy callers remain unchanged. An independent
source-audit validator compares allocation-independent scientific views before
any cohort manifest is frozen.

Initial source scans `41292`/`41293` complete the expensive 1,600-case read,
then reject before writing results because the reused shadow allocation receipt
requires an H100 even for a read-only CPU audit. Dependent validator `41296`
is consequently dependency-impossible. Preserve these jobs as post-scan
apparatus history. The repair records a strict Slurm CPU receipt and changes no
source case, eligibility rule, exclusion, warning time, or scientific field.

The next apparatus commit also prepares a deterministic cohort freezer and a
compact inherited exact-group configuration. The freezer requires validated
independent source views, assigns complete task-level groups to exactly one
split, selects unique root episodes without reading candidate outcomes, and
fails rather than silently replacing a missing palm/L6 source.

Clean CPU producer `41298`, independent replay `41299`, and validator `41300`
complete ADR-0185 from immutable commit
`dba5935125dfa1c1f2cd2e92b6cda94f7c09c076`. Both audits independently read
all 1,600 natural pi0.5 roots and agree exactly. The audit retains 232 eligible
outcome-untouched sources: 228 palm roots across 15 task-level groups and five
L6 roots across two task-level groups (one root contains both targets). It
accessed no counterfactual candidate outcome. The five L6 sources are the
measured availability ceiling in this immutable population, so the additive
cohort freezes all of them rather than inventing generic states.

The prospective additive cohort is frozen before candidate outcomes at 14
unique roots: palm `3/3/3` and L6 `4/0/1` over train/validation/test. Within
this cohort every task-level group belongs to exactly one split. This targets
only the measured palm/L6 deficits, uses one raw-contact-derived warning time
per root, and deliberately permits a final under-covered result: if the
available L6 test root is not two-sided, the Q-only run still proceeds as an
explicit diagnostic and correction remains blocked.

The final Q-only prediction protocol is also frozen before these outcomes. It
retains the 9D, 33D, and shared 135D per-constraint arms, symmetric loss,
AdamW weight decay `1e-4`, train-only normalization, fixed 2,000-epoch model,
UNKNOWN censoring, and one-time validation/test evaluation. Correction can be
authorized only by the conjunction of the external coverage gate and the
held-out prediction gate; under-coverage never becomes a safety claim.

Initial freezer `41303` is pre-simulation apparatus history. It revealed that
the one natural root eligible for both palm and L6 was consumed by the abundant
palm pool, leaving only three of four registered L6-train identities. No
candidate outcome was opened and no simulation ran. The deterministic repair
reserves multi-target roots for the later scarce target whenever exclusive
palm roots already satisfy the requested count; all source records, splits,
counts, warning rules, and scientific settings remain unchanged.

Clean freezer `41305` passes from commit
`de857ca91be3e089f83e2f57f08249c0f349c55b`. The committed manifest SHA-256
is `3f4182a08b2eca8b87b1b906bdf99c90a3752b8e925d739f3e74026f2e5be500`.
It contains exactly 14 unique roots with the frozen `7/3/4` split, palm
`3/3/3`, L6 `4/0/1`, one warning state per root, task-level-group isolation
inside the new cohort, raw pi0.5 nominal actions, and no candidate-outcome
access. The targeted collection configuration inherits the validated
whole-body superset and 13-bank unchanged.

The final combined-audit implementation now distinguishes raw two-sided states
from prevention two-sided states. A root counts toward the `4/2/2` gate only
when its snapshot is initially safe across palm/L5/L6/L7 and its known
candidates straddle the named constraint. Initially unsafe roots remain fully
reported as recovery diagnostics but do not enter prevention coverage or
training. This fixes the prior coarse all-roots initial-safety veto without
dropping any evidence.

The first targeted producer/replay chain `41307/41308/41309` exposed an
apparatus-only raw-pi0.5 geometry binding defect before any final case shard was
written. The source curves correctly used raw pi0.5 state/actions and stored the
paired AEGIS result only as fixed diagnostic geometry, but the compiled-target
replay attempted to read MVEE fields from the raw pi0.5 ledger. The repair now
resolves and hash-validates the registered geometry-only binding, verifies that
it cannot supply state/actions, and records that provenance in the exact case.
The frozen 14 roots, `7/3/4` split, 13-bank, no-QP execution, OSC, continuation,
and timeout policy are unchanged. Failed tasks in the initial chain remain
apparatus history; no candidate outcomes from them are used.

Queued repair chain `41318/41321/41322/41323/41324` is also apparatus history:
its source-prep binding expanded short commit `80f45ef` to the wrong full hash,
so it cannot create its source tree or reach simulation. The verified immutable
repair commit is `80f45ef8156b4b5d9cf8be6d8ddd2fad2cdadeb1`. Clean source-prep/retry-prep
jobs are `41327/41328`; the dependent replacement producer, replay, and
validator are `41329/41330/41331`.

A hash-checked final-gate binder is now prepared. After targeted validation it
derives the actual combined split arithmetic (`14/4/6` existing plus `7/3/4`
targeted = `21/7/10`), binds the third validation file and payload, and writes
the v2 coverage config. After the coverage audit it binds that exact audit to
the frozen Q-only protocol. This avoids manual hash transcription and does not
alter coverage, model, loss, split, or correction criteria.

The post-collection workflow is dependency-scheduled rather than left for a
manual handoff. Final source-prep `41336` binds commit
`578278a5a253bacfcb00faf3d184299baf6d9e20`; coverage binder/audit jobs are
`41339/41340`; Q-only binder/trainer/independent retrainer are
`41341/41342/41343`. These jobs run only after clean validator `41331` passes.
Training therefore still runs when coverage is scientifically under-supported,
but the result is labeled diagnostic and cannot authorize correction.

The preceding replacement chains through `41378` are retained as pre-simulation
source-transfer apparatus history. They failed on shell portability, incorrect
or unreachable commit bindings, or a partial clone missing required Git blobs;
none produced a final scientific case shard. Robust source-prep job `41389`
fetched the named remote branch without blob filtering, verified immutable
collection commit `80f45ef8156b4b5d9cf8be6d8ddd2fad2cdadeb1`, and created clean source
`/home/quanth/working_space/vlsa-aegis-palm-l6-80f45ef-v15`. Exact archival job
`41390` moved only the 28 failed runtime directories under the existing artifact
root and preserved every completed artifact.

Clean paired collection is now producer array `41391`, independent replay array
`41392`, and dependent validator `41393` at
`/mnt/data/quanth/experiments/vlsa-distal-pi05-palm-l6-extension/pi05-palm-l6-extension-20260817a`.
The final automatically chained workflow is source-prep `41396`, coverage binder
`41399`, combined audit `41400`, Q binding `41401`, fixed Q-only trainer `41402`,
and independent retrainer/validator `41403`. It binds final audit/training commit
`578278a5a253bacfcb00faf3d184299baf6d9e20` and still executes the Q-only test
when coverage is under-supported.

The first completed paired root, `vlsa-t1-goal-i-t1-e19`, is a valid retained
recovery/one-sided diagnostic rather than prevention evidence: its registered
palm snapshot has initial normalized radial slack `-0.250159`, 12 known
palm-unsafe candidates, zero palm-safe candidates, and one UNKNOWN timeout. The
second paired root, `vlsa-t1-long-i-t2-e40`, supplies the intended prevention
boundary: it is initially safe across palm/L5/L6/L7, has palm `10 safe / 3
unsafe / 0 unknown`, and has ten globally physical-safe candidates. Both use
the frozen 13-bank, raw pi0.5 nominal chunk, disabled released EE-QP, unchanged
OSC, and full fixed continuation. Independent full-population validation remains
the authority before either root is counted in the final coverage gate.

## 2026-08-17: Targeted palm/L6 collection and diagnostic Q-only verdict

The final targeted palm/L6 collection is complete. Producer/replay arrays
`41391/41392`, retained-rejection repair jobs `41455/41456`, and independent
validator `41459` preserve all fourteen frozen root episodes under
`/mnt/data/quanth/experiments/vlsa-distal-pi05-palm-l6-extension/pi05-palm-l6-extension-20260817a`.
Twelve roots produced complete 13-candidate whole-body shards and two roots were
retained as preregistered scientific rejections because their restored warning
snapshots already failed CAR before candidate execution: validation palm
`vlsa-t1-goal-ii-t1-e36` and test L6 `vlsa-t1-spatial-i-t3-e24`. Nothing was
replaced. The successful roots contain 156 candidate rollouts: 94 known safe,
42 known unsafe, and 20 UNKNOWN timeouts. Independent producer/replay scientific
views agree exactly, and represented-geometry physical false-safes are zero.
Validation file SHA-256 is
`9c4f69c3a342f575c6b69e126653266d242f6d511860d7f2226be01a39721caa`.

The combined 36-root audit completed in job `41461` at
`/mnt/data/quanth/experiments/vlsa-distal-whole-body-final-gate/whole-body-final-gate-20260817a/combined_coverage_audit.json`.
It retains all 580 candidate slots, including 489 known outcomes, 91 UNKNOWN
timeouts, three initially-unsafe recovery cases, and both CAR rejections.
Prevention two-sided train/validation/test support is L5 `6/3/4`, palm `5/2/3`,
L6 `3/2/0`, diagnostic EE `8/3/4`, and diagnostic L7 `1/0/0`. Global safe
support passes in every recoverable held-out state and physical false-safes
remain zero. The coverage gate is nevertheless scientific NO-GO because L6
does not reach `4/2/2`; the two rejected roots also keep the full-cohort
apparatus-completeness flag false rather than being silently discarded. Audit
file SHA-256 is
`c36aa3ebde2abcfddc18c7deb69dd1b986c7a8097d32618dd29fd6107b1e5de6`.

As explicitly authorized, H100 job `41463` trained the fixed diagnostic Q-only
arms despite under-coverage, and independent H100 job `41464` reproduced every
model and prediction exactly. Eligible initially-safe known data comprise
18/6/9 train/validation/test states and 232/89/132 candidates; UNKNOWN timeouts
and recovery roots are excluded from fitting, and normalization is train-only.
The shared 135D constraint-conditioned model fits train global RMSE `0.028646`
but transfers poorly: validation/test global RMSE is `0.520684/0.590234`,
near-boundary RMSE is `0.569030/0.575875`, and rank Spearman is
`0.114709/0.336843`. It provides predicted-safe support in all 15 held-out
states but selects unsafe actions in two validation and four test states. The
matched 9D and 33D arms also fail held-out safety; their test RMSE values are
`0.738850` and `0.612150`, respectively. Result and independent-validation file
SHA-256 values are
`90a85e05f4d3927054073860bc76121cd28363c7bbd19a0509b3fc0b15368e25`
and `0bb302cd044dfe5bc5f2257b591851f5bd40d679c4ad8b3f2817c1fe45b732f7`.

Both required correction gates fail: per-constraint coverage is false and
held-out prediction is false. Therefore no candidate correction, QP,
denoising, or closed-loop action-selection run was enabled. The completed
workflow validates the collection/training apparatus and the local predictive
signal, but rejects the current model as an unseen-state whole-body safety
filter. The next data priority, if continued, is prospectively frozen,
initially-safe L6 boundary support in training and especially untouched test
task groups; palm and L5 no longer need generic expansion for this gate.

## 2026-08-17: Unseen per-constraint transfer follow-up

ADR-0188 freezes the next experiment around the existing model and data
contracts. A new allocation-backed audit is prepared to join the six shared
model false-safe selections to their exact whole-body candidate artifacts. The
preliminary exact join attributes the claimed physical witness to L6 once, L5
three times, and palm twice. This makes the scientific test more precise: new
L6 boundaries test the missing-L6 hypothesis, while unchanged L5/palm results
measure whether whole-body transfer remains unsolved.

No existing episode will be recollected. The validated natural-pi0.5 audit of
all 1,600 immutable Table-1 roots has only five eligible L6 roots across two
task-level groups, and all five are already frozen in the current evidence.
Consequently the intended additive `6/2/4` L6 cohort requires a genuinely new
prospective raw-pi0.5 source population and new task/obstacle groups. Candidate
outcomes remain unopened until complete episode-group splits are frozen.

The collection-speed change is isolated behind an opt-in equivalence gate.
Sequential execution remains the reference. One opened training state must
reproduce exactly under four isolated simulator processes before the parallel
path can be used for new producer/replay collection. The frozen 13-bank, no-QP
execution, unchanged OSC, complete continuation, whole-body targets, contact/CAR
authority, and UNKNOWN timeout handling do not change.

Initial CPU audit job `41469` is apparatus history. Its scoped unit tests pass,
but direct script execution omits the repository root from Python's module
search path and fails before opening any immutable result. Invoke the identical
auditor as a Python module; no data binding, expected selection, or attribution
rule changes.

Replacement CPU audit `41470` is also apparatus history. It opens the immutable
Q result and exact cases, then rejects because the frozen state-selection summary
does not serialize the selected candidate's predicted scalar. That scalar is not
needed for the registered attribution table. Remove only the unsupported display
field; selected identity, exact palm/L5/L6 risks, witnesses, and contacts remain
unchanged.

The opt-in process executor and equality validator are now locally prepared.
The canary uses opened L6 training root `vlsa-t1-goal-ii-t3-e31` (frozen targeted
cohort index 9), comparing the unchanged sequential collector with four external
candidate processes. New collection remains blocked until its allocation-backed
equality result passes every registered action, state, risk, contact/CAR, timeout,
terminal-state, and order check.

Allocation-backed audit `41472` completes the exact six-selection join. The
validation false-safe is L6 at Goal-II task-0 E39. The remaining validation
false-safe is palm at Long-II task-2 E13. The four test false-safes are palm at
Long-II task-3 E46 and L5 at Spatial-I task-3 E12/E15/E42. Exact physical
witness counts are therefore L6 `1`, palm `2`, and L5 `3`; the diagnostic EE
proxy also overlaps for E46 but is not promoted to physical authority. This
confirms that missing L6 support is a real isolated deficit while preventing an
incorrect claim that L6 coverage explains all six failures. Audit result file
and payload SHA-256 values are
`d7895193e7d6617767c5f496cdab5d2bdf08c4f224b312e07c253d4cc48fa34e`
and `defb68c5402ee0d3e4b4706e893ecdb7830353219bf5a6e9beae0060fb519450`.

The prospective raw-pi0.5 discovery population is now frozen before any of its
new outcomes. It contains 28 query-noise roots in three non-overlapping
task-level groups: Object-I task-2 `12` train, Goal-II task-0 `8` validation,
and Spatial-II task-2 `8` sealed test. These are source-discovery roots, not
candidate states: all outcomes are retained and only roots with an initially
contact-free, complete five-action boundary before a raw L6 contact may enter
the intended additive `6/2/4` counterfactual cohort. Manifest SHA-256 is
`eb393918bf96e8a00f08acb956818f3f111e28353624cbfc4378e320c2b9c0c5`.
The frozen 13-bank is still unopened for this population; source discovery
cannot train a model or authorize correction.

Sequential job `41473`, four-process job `41474`, and CPU validator `41475`
complete the candidate-worker equivalence canary. All requested/proposed/
effective chunks, restored and terminal state hashes, per-constraint risks,
contacts/CAR, timeout classes, and canonical candidate order agree exactly.
The optional parallel executor is therefore scientifically equivalent on the
opened state. Its measured speedup is only `0.9689x`, however, so it is not a
performance win and will not be used broadly merely because it passed. The
sequential collector remains the default for the new L6 cohort. Validation file
and payload SHA-256 values are
`62e2095cfc3c97c3a3c5cc5125aa7d1c7a9171b00af6b0954e9f59a07c836e64`
and `da6f73dc6725da7d5e7e0b8fc8b7209c0b6c0d61403de6d5b246ea94394f0984`.

## 2026-08-17: Frozen-MLP action-selection audit

At the user's deadline-driven pivot, no prospective L6 source-discovery job was
launched. The frozen manifest and draft apparatus remain provenance only; no
new outcome or candidate label was collected. Work moved to the existing frozen
shared 135D MLP, its serialized checkpoint, and the already validated candidate
artifacts.

CPU audit `41483` is the authoritative all-candidate action-selection result.
It causally rebuilds shared-model features for every candidate, including the
`UNKNOWN_TIMEOUT` candidates censored from training, and replays the serialized
MLP with maximum known-row discrepancy `2.251e-6`. Thus no timeout identity or
future label is used to rank candidates.

Direct model-only correction remains a NO-GO. Minimum predicted global risk
selects exact-safe actions in `5/6` validation and `7/9` test recoverable
states, leaving one and two false-safe selections. Zero-threshold
minimum-intervention selection leaves two validation and four test false-safes;
the validation residual margin abstains in `5/6` validation and `7/9` test
states and both selected test actions are unsafe. The exact minimum-intervention
oracle has safe support in `6/6` validation and `9/9` test states, proving the
candidate bank—not intervention authority—is sufficient at these snapshots.

The useful result is ranking under mandatory exact verification. The first
exact-safe candidate appears within ranks `4`/`5` for every validation/test
state, with mean checks `1.50`/`1.67`. Top-1 support is `5/6` and `7/9`; top-3
is `5/6` and `8/9`; top-5 is `6/6` and `9/9`. No held-out timeout candidate
precedes the first exact-safe proposal. This authorizes only a research pilot
in which the frozen MLP orders the bank and fresh cloned-rollout verification
rejects unsafe or unknown proposals before execution. It does not authorize
model-only correction, QP, denoising, deployment, or a certified barrier claim.
Result file/payload SHA-256 values are
`67d6560e17ed8e412d563bed525c9c2d7ceef3c55207dbcabcbffbde52be04b2`
and `07455877d212e017820d2c0e9ac4b0cf6622a92fee52d2fc42c377aa15d9a17c`.

ADR-0190 now freezes the first execution test around exactly the three
minimum-risk top-one false-safe states: Goal-II E39 (L6), Long-II E46 (palm),
and Spatial-I E42 (L5). Their shared-MLP top-five names are copied unchanged
from CPU rank-prefix audit `41510`; candidate outcomes do not alter their order.
Each candidate is freshly executed from the identical saved state through
clipping, unchanged OSC, and complete continuation. Unsafe and timeout results
are verifier rejections, never primary commands. The first candidate passing
palm/L5/L6/L7 risk, raw-contact, CAR, and safe-terminal checks is the selected
five-action chunk. Producer and independent replay must agree exactly. This is
an exact-verified correction mechanism test, not late denoising or closed loop.

The user stopped ADR-0190 after the first case because the initial apparatus
re-executed the complete 27-bank once per ranked candidate. Jobs `41514/41515`
were canceled only after both atomically completed their first fresh bank; no
later-case simulation was launched. Apparatus-only commit `a227d75` replaces
the redundant loop with one complete-bank execution followed by the unchanged
logical top-five prefix scan. CPU finalizers `41565/41566` bind the preserved
fresh banks and independently reproduce E39 exactly (scientific-view SHA-256
`ac8dae59ac9cd8bf0664c9222fd213ec3d0fbc467216da3953f78724fb6c4e9c`).
The MLP rank-1 candidate is freshly unsafe from positive L6 violation
`0.0145145`; ranks 2--3 are also unsafe and have raw protected contacts. Rank
4, `grid_z0_p1_m1_front_loaded_r2.0`, is known safe with palm/L5/L6/L7 future
violations `-0.466478/-0.098516/-0.041459/-0.445327`, zero group/raw contacts,
and no physical veto. Thus exact top-five verification avoids the E39
false-safe choice in `1/1` executed cases. This is encouraging mechanism
evidence only: the preregistered three-case gate is incomplete, and E46/E42,
closed loop, model-only correction, QP, and denoising remain untested. Producer
and replay file SHA-256 values are
`797c339a4fc3f9e0892bdfcbed42dbe81696bbe1c23476404d1ba82533d83d4f` and
`bdb9d90178930acd5b023e3e76dac4391fd551e23a52b4af7c3ffb122a624408`.

## 2026-08-17: Compact inference-only Q diagnostic

At the user's inference-only pivot, ADR-0191 stops new simulator collection and
the unfinished top-five execution pilot. A compact 7D shared per-row feature is
preregistered using only current exact radial slack and the nominal/effective
five-action chunk projected into the current primitive--obstacle outward
coordinate. It consumes no future state and scores the complete finite bank in
one MLP pass.

The experiment reuses the immutable 580 stored candidate slots, censors UNKNOWN
labels from fitting/error metrics while still scoring them at inference, and
keeps the established train/validation/test identities. Its primary learned
aggregate is diagnostic EE/palm/L5; L6 is retained as a separate physical audit
so any collision caused by the narrower scope remains visible. Local contract,
feature, selector, syntax, and JSON checks pass; the numerical feature test is
deferred to the registered allocation because the desktop runtime lacks NumPy.
No simulation, collection, QP, denoising, or fresh exact verifier is scheduled.

H100 fit `41587` and independent retrain `41589` finish in under one minute each
and reproduce model SHA-256
`d7c36e5a70aec7f0680d610093679af28a29b0c0a24597cd1f73dfa0268c76a8`
exactly. Dataset use is unchanged: 18/6/9 eligible train/validation/test states,
232/89/132 known candidates for error metrics, and 58/17/13 UNKNOWN candidates
scored but never treated as safe. New simulator rollout count is zero.

The compact global RMSE is `0.129139/0.161441/0.361087` on
train/validation/test, versus `0.028646/0.520684/0.590234` for the shared 135D
model. Compact near-boundary RMSE is `0.109961/0.161456/0.316331`, and rank
Spearman is `0.823066/0.704266/0.673485`. Minimum predicted primary risk selects
all-physical-safe candidates in `6/6` validation and `7/9` test states with no
UNKNOWN or L6-failing selection. The remaining unsafe test selections are E15
L5 row 3 and E42 L5 rows 3--4. Thus the compact representation substantially
improves value transfer and validation action selection, but it does not remove
test false-safes and cannot authorize model-only safety execution.

Result/validation file SHA-256 values are
`cb0fcf2e37761eb867fc222ea557d68812bdf192f9013f2adc93cfa790c8abb7` and
`65ccce6018bd272a322ca93f18fe996298a4471a1c57f66cf2654e8908ae83e4`;
payload SHA-256 values are
`18cbafbdcd8f19b99a3f734bdf6d1e4f96bbbef241e923e65c29d33a5ec76191`
and `ab772de0a347ca798aa929b947710f1dc31b6206920cf987a2a63c1d190edaea`.

## 2026-08-17: Compact inference-only paired execution pilot

ADR-0192 freezes the first action-execution test of the compact 7D selector.
It does not recollect a candidate bank or use an exact rollout verifier before
execution.  On each registered warning state it executes only two branches from
the identical restored snapshot: raw nominal pi0.5 and the already frozen
minimum-predicted-risk candidate.  Released AEGIS EE-QP, learned QP, gradients,
and denoising remain disabled; the unchanged OSC and complete registered
continuation define the future-risk outcome.

The four opened cases intentionally expose both sides of the known result:
Goal-II E39 and Long-II E46 are the former L6/palm shared-model failures that
the compact selector ranks to physically safe candidates, while Spatial-I
task-3 E15 and E42 are the two remaining compact L5 failures.  All four nominal
branches are physically unsafe in the immutable evidence.  Therefore this is a
minimal paired mechanism reproduction, not a new generalization test.  It can
show whether direct compact inference improves collision outcomes from `0/4`
without hiding the known failures, but it cannot establish collision-free
safety or full-episode task preservation.

The implementation adds an opt-in two-name subset to the existing collector;
the default 13/27-candidate paths are unchanged.  Focused local contract tests,
Python compilation, shell syntax, and `git diff --check` pass.  The desktop full
`./init.sh` still reaches all tests but fails only where the desktop Python lacks
NumPy; allocation-backed focused tests remain required before interpretation.

Producer/replay arrays `41624/41625` validate E39 exactly, but indices 1--2
reject before candidate simulation because the opt-in subset path regenerated
the 27-grid after correctly reconstructing a frozen 13-bank.  Both replicas
raise the same `active boundary search candidate count differs` exception; no
case JSON is written for either failed index.  This is an apparatus branch bug,
not a scientific outcome.  The repair keeps the reconstructed frozen 13 rows
and selects the registered two names from them.  Existing valid shards remain
immutable, and the validator explicitly accepts the preregistered and repair
commits so only missing indices need rerun.

The first missing-index retry exits at the Slurm preflight because the retained
failed attempts already own the default per-case runtime directories.  H100
diagnostic `41653` independently confirms one H100, the exact repair commit,
and a clean source.  Preserve the failed runtime directories and use an
opt-in retry suffix; final case JSON paths and all scientific settings remain
unchanged.

Final missing-index producer `41655`, replay `41657`, and validator `41659`
complete ADR-0192.  All four nominal branches are physically unsafe; the
frozen compact selections are physically safe for E39 and E46 and L5-unsafe
for E15 and E42.  Thus selected safety improves from `0/4` to `2/4`, with two
avoided collisions, two persistent L5 violations, zero UNKNOWN selections,
and exact producer/replay equality.  Every registered mechanism gate passes,
but model-only safety fails.  Correction safety, closed loop, and untouched
generalization remain unauthorized.  Validation file/payload SHA-256 values
are `c03598f26cc1fa762b82d24e6ac9b2fbc34c2adcbb5b2a4eff81e58d188ce0c5`
and `95230f86c0cce70b0c453747cf3a90fd37d8650b6981f7b848089b5a58725489`.

## 2026-08-17: Frozen compact-selector inference ablation

ADR-0193 preregisters a zero-simulation selector audit over the immutable 580
candidate slots.  It does not train a new network.  It binds the compact shared
7D model from ADR-0191 and the direct-L5/OSC 33D specialist from the fixed
Q-only gate, replays both serialized checkpoints, and compares five inference
rules: compact minimum risk, zero-margin safe-or-abstain, validation-conservative
per-group safe-or-abstain, validation-conservative L5-only gating, and compact
EE/palm plus max(7D,33D) L5 gating.

For every conservative rule, the margin is the maximum observed positive
residual `actual_group - predicted_group` over known validation candidates only.
UNKNOWN timeouts do not fit a margin and are never counted as safe.  The test
split is already opened and is evaluated only after margins and a deterministic
validation-only arm choice are frozen; it therefore remains a post-hoc
diagnostic.  Every selection reports EE, palm, L5, and L6 audit outcomes,
abstention, correction magnitude, and the E15/E42 L5 failures explicitly.

No simulator step, candidate recollection, retraining, QP, gradient correction,
denoising, or closed loop is permitted.  This audit can determine whether the
remaining failures are an inference-rule problem that merits a fresh
prospective paper test.  It cannot authorize correction safety or a CBF claim
from the already opened population.

Final H100 inference producer `41695`, independent replay `41696`, and validator
`41701` complete ADR-0193 with zero simulator rollouts and zero retraining.
Both frozen checkpoints reproduce their known predictions with zero recorded
error, and the complete scientific views agree exactly.  UNKNOWN candidates
remain scored but censored from margin fitting and never counted as safe.

The validation maximum-optimism margins are large: EE `0.471567`, palm
`1.398977`, compact L5 `1.204603`, and fused compact/33D L5 `0.682427` in the
dimensionless radial-slack target.  The validation-only arm choice therefore
remains compact minimum risk: it selects `6/6` physically safe validation
actions with no abstention or UNKNOWN.  On the already opened test it remains
`7/9` physically safe; E15 and E42 remain L5-unsafe.

Conservative gating removes selected physical collisions on the opened test,
but collapses support.  Full per-group margins select `1/9` and abstain `8/9`;
compact L5-only margins select `2/9` and abstain `7/9`; compact plus the 33D L5
specialist selects `3/9` and abstains `6/9`.  All selected candidates in those
three opened-test arms are physically safe, but none accepts E15 or E42.  The
zero-margin safe-or-abstain rule is not conservative: it selects `3` known-safe,
`2` physically unsafe, and `2` UNKNOWN outcomes while abstaining twice.

The decisive inference is that the compact model is useful for candidate
ranking but is not calibrated as a transferable safety threshold.  Maximum
validation residual margins convert it into a high-precision abstention gate,
not a useful standalone correction filter.  The 33D specialist improves L5
selective coverage slightly but does not solve the EE/palm validation failures
and is retained only as an ablation.  For a paper experiment, freeze compact
minimum-risk ranking and claim empirical collision-rate reduction, not
collision-free safety; alternatively, a safe-or-abstain system must define and
test a real abstention backup/requery path.

Several failed jobs are retained as apparatus history: `41671` used `/bin/sh`
for a Bash-only source-prep wrapper; `41672` timed out during a full clone;
`41677/41678/41682/41684/41686` established that CPU checkpoint arithmetic did
not reproduce H100 outputs; `41688/41689/41691/41692` exposed test-discovery
shadowing in the Anaconda environment; and `41697/41699` exposed shorthand
state IDs and the H100-only allocation receipt.  None produced a scientific
artifact or changed the registered arms, margins, data, models, or outcomes.

Producer/replay/validation file SHA-256 values are
`2ab162cb0fc5b8c698bec51c401c2df83ec1ce09037d0e839e0a7600487cdf79`,
`d7883a54bafffec14f4e010f1c331b24328df2c9150fa72e67496c757e267c91`,
and `2829fee5427e69512a2765a7ca5849725445aae7d49fbd82772e22701ca43a69`;
the validation payload SHA-256 is
`21adfb6de379a8fbddce437d1f5ae6ca7281eb7fa5f1dedd1e1dc8a64bb50ca4`.

## 2026-08-17: Terminalize late-flow branches before risk scoring

ADR-0194 resolves the paper's terminal-versus-denoising ambiguity without
retraining the critic or recollecting candidate labels.  The new path is
strictly opt-in.  At Euler step eight of the frozen ten-step pi0.5 sampler it
creates the frozen 13-branch bank in model coordinates using scale-only
displacement normalization, completes every branch through the same final two
velocity-field updates to `t=0`, then decodes each complete terminal action
chunk through the ordinary output transform.  The frozen ADR-0191 compact 7D
critic is outside the sampler and may receive only the clipped effective first
five terminal actions.  It never receives a raw denoising latent.

Branch zero has exactly zero residual and must reproduce an ordinary pi0.5
query under the same real SafeLIBERO observation and RNG seed.  The first gate
is therefore an allocation-backed paired producer/replay sampler canary at the
immutable E05 step-180 query.  It performs one ordinary query and one batched
13-branch terminal query per replica, but executes no candidate branch in
MuJoCo, trains no model, and reads no new safety label.  Required gates are
branch-zero parity, finite terminal chunks, nonzero branch diversity, exact
candidate names/count, no risk scoring inside the sampler, and exact
independent scientific replay.

Only if this sampler canary passes may the already frozen compact critic score
the executable terminal bank.  The next paired pilot will execute at most one
selected five-action chunk per arm and compare ordinary pi0.5, the established
post-hoc compact selector, and late-flow terminalized compact selection from
the identical state/observation/noise/controller horizon.  Released EE-QP,
learned QP, exact rollout verification at inference, calibration, retraining,
and CBF claims remain disabled.

Initial paired H100 producer `41709`, replay `41710`, and validator `41711`
complete with exact scientific replay but correctly return NO-GO.  All 13
terminal branches are finite and diverse, names/count are exact, and the
sampler performs no risk scoring, but the zero-residual first row of a batched
13-branch JAX execution differs from the exact ordinary compiled sampler by
`0.008584950775146427` output action units.  Do not weaken the `1e-6` parity
gate.  Preserve these artifacts as apparatus evidence.  The repair evaluates
branch zero through the exact ordinary sampler and batches only the twelve
nonzero branches; it also retains the raw batched-zero model drift in every
result.  This changes neither nonzero branch generation nor any controller,
critic, candidate, state, seed, or scientific acceptance rule.

Clean repaired producer `41716`, replay `41717`, and validator `41718` pass
every ADR-0194 sampler gate with exact independent scientific equality.  The
ordinary and authoritative terminal branch-zero output are identical (maximum
absolute difference `0.0`), all 13 terminal chunks are finite, maximum terminal
branch diversity is `0.8216188234314321`, candidate names/count are exact, and
the sampler neither scores risk nor selects a branch.  The unused raw batched
zero path still differs by `0.0103302001953125` model units and is retained as
diagnostic evidence.  Producer/replay/validation file SHA-256 values are
`6279b3739a5986e2fb094ebc972c3b5b1c0278e3469345fa1eff2ed2175d28c6`,
`317208b91fc7135405befeda74a62e1b3c78833f0884efd6309cd784935f637d`,
and `0e97f798ec2c1d1138c06a5c102b85b80bd5e468603104ed1c739311d95bb181`;
the validation payload SHA-256 is
`65d8d293d7a8b5b50d313e5cf7cfb9b2165700c278e26034fe2cb2a7b1b22d2b`.
This authorizes binding the frozen compact scorer to terminal chunks, not a
safety, collision-reduction, or control claim.

The ADR-0194 selected-action mechanism pilot is now implemented and locally
preregistered.  It binds the immutable sampler validation, immutable E05
step-180 state, frozen 13 candidate names, and immutable ADR-0191 compact 7D
checkpoint.  The scorer clips terminal XYZ actions, constructs causal 7D
features only from current geometry and terminal first-five action projections,
and selects minimum predicted EE/palm/L5 risk with correction norm only as a
tie-break.  The sampler itself remains critic-free.

The pilot executes only ordinary pi0.5, the post-hoc compact-selected terminal
chunk, and the late-flow-terminalized compact-selected chunk.  It performs no
13-way candidate future search, no inference-time exact verifier, no QP, and no
training.  Producer/replay each retain three scientific selected-arm outcomes;
the pre-existing source evaluator's two nominal determinism checks and exact
per-arm replay are explicitly registered as apparatus checks rather than hidden
scientific candidates.  Raw collision and represented-geometry/terminal safety
are reported separately.  Local focused tests pass `29` tests with one NumPy-
dependent test skipped; the full desktop gate ran `592` tests but the desktop
Python lacks NumPy, so allocation preflight must run the focused and collector
tests in the registered scientific environment before simulation.

Initial selected-action producer `41728` and replay `41729` passed allocation
preflights and restored E05, but neither reached selected-action simulation:
both allocations were placed on worker-2 with policy port 8028.  The second
server failed to bind and the first client's websocket closed.  Retry `41731`
correctly refused the immutable failed replay directory; validators
`41730/41732` are dependency-impossible.  These are apparatus-only history.
Use a new immutable run root and a per-job port derived from `SLURM_JOB_ID`;
keep all scientific bindings and gates unchanged.

Port-isolated producer `41734` completes in `00:09:34`.  Ordinary, post-hoc
compact selection, and late-flow-terminalized compact selection are all known,
raw-contact-free, CAR-free, and represented-geometry safe.  Post-hoc and late
flow both select `grid_p1_m1_z0_front_loaded_r2.0`; their minimum L5 slacks are
`0.0128093` and `0.0646268`, respectively.  Nominal is also safe with L5 slack
`0.0854056`, so E05 tests executable terminalization but not collision-rate
improvement.

Regenerated-policy replay `41738` independently reaches the same selected
names and three safe qualitative outcomes in `00:09:33`, but ordinary and
terminal action hashes differ slightly from `41734` despite the exact same
dynamic-state hash and RNG seed.  Preserve both as evidence and reject an exact
scientific-pair claim.  The replay repair consumes the immutable `41734`
producer action bytes and independently executes only those three arms on a
CPU Slurm allocation.  Its full scientific view must byte-match the producer;
there is no second VLA query, no server, and no new selection.

Exact-action CPU replay `41742` failed in four seconds before environment
creation because the inherited evaluator requires an explicit
`QUERY_RISK_CPU_CANARY=1` allocation receipt for CPU execution on an H100 host.
Retain the failed root.  Add only that receipt to the CPU replay Slurm script
and use a new immutable run ID; scientific actions and outcomes remain bound to
producer `41734`.

## 2026-08-17: End-to-end task preservation after terminal selection

The E05 selected-action artifact does not contain `task_success` or
`goal_satisfied`; `SAFE_TERMINAL` records only the fixed safety continuation's
clearance and stable-hold condition. The active ADR-0194 gate is therefore
extended with one minimal opened task-preservation diagnostic rather than
another candidate-data collection.

The corrected opt-in protocol binds the immutable `41734` late-flow selected
action bytes and tests our method only. It replays the same archived prefix
through step 179, executes the registered late-flow-terminalized five-action
chunk at steps 180--184, and then reobserves and reapplies the same frozen
late-flow terminalization plus compact terminal-action selector every five
actions through native task success, physical failure, or the registered
300-action limit. No nominal/post-hoc arm, released EE-QP, learned QP, exact
inference-time rollout verifier, model update, or label collection is present.

Task success, raw active-obstacle robot contacts, CAR, and timeout are separate
outcomes. A lightweight read-only monitor records contacts at all 25 MuJoCo
substeps per action, reports palm/L5/L6/L7 separately, retains unmapped robot
geoms as `other_robot`, and records paper CAR. Producer videos are emitted for
the method arm. A CPU exact-action replay must reproduce the producer's
complete scientific view before the result is accepted.

Local focused validation passes 42 tests with four optional NumPy/Pillow skips;
the new task-success contract itself passes all six tests. The full `init.sh`
gate runs 592 tests and reports only the 20 already known desktop-Python NumPy
import errors, with 67 skips. Allocation preflight will rerun focused tests in
the registered scientific runtime before execution. The next action is commit,
remote source preparation, live Slurm preflight, and one H100 producer followed
by CPU exact-action replay and validation.

Producer `41792`, exact-action CPU replay `41799`, and validator `41801`
complete the corrected our-method-only experiment at immutable commit
`a3a4213087edbc0792231326fac8e0b54eb9cef5`. Exact scientific-view replay
passes. The method executes the validated late-flow-selected chunk at steps
180--184 and performs one new terminalized 13-branch inference/compact scoring
query at step 185. It selects `grid_z0_p1_p1_front_loaded_r2.0`; importantly,
the minimum predicted EE/palm/L5 violation is already positive
(`0.3797674506`), so the frozen minimum-risk fallback has no predicted-safe
candidate.

The episode reaches native task success at step 189, but the same step contains
raw `robot0_link5_collision` contact with `moka_pot_obstacle_1_g13` beginning at
internal substep 7 (19 contact samples, penetration `-0.0001266457 m`). Paper
CAR still passes with maximum active-obstacle L1 displacement
`0.0005931699 m`; there is no timeout. Therefore native task success is `1/1`
but collision-free task success is `0/1`. This is a scientific failure of the
current minimum-risk-when-all-unsafe control rule, not a predictor miss: the
critic warned that even its best available candidate was unsafe.

Validation file/payload SHA-256 values are
`5f4ac4aa8d86d1755bbe925c7d425fc35c26a80f132361dc610de3b46f871ee4`
and `dd7b3d1369558bb58ffe9de468d29c0f43960bbdde09b7948194cfac17eba523`.
No further simulator run is authorized by this result. The next method decision
is to define a fail-closed response or increase inference-time candidate
authority when every candidate has predicted risk above zero; silently
executing the minimum-risk unsafe candidate is rejected.

## 2026-08-17: True full-episode model inference

The user authorizes one narrower E05 run that fixes both limitations of the
previous task pilot. The method is active from policy query zero, with no
archived action prefix, and it is invoked again after every executed five-action
chunk. Each query makes one batched pi0.5 terminal-branch request, scores the 13
terminal executable chunks with the unchanged frozen compact 7D critic, and
executes the least-modifying candidate whose predicted maximum EE/palm/L5 risk
is non-positive. If no candidate passes, the episode abstains immediately; it
must not execute a predicted-unsafe minimum-risk fallback.

This remains one opened-case mechanism test. No AEGIS QP, learned QP,
simulator lookahead, retraining, new label collection, calibration, or
comparison arm is present. The producer stops at native task success, first raw
robot contact, first paper-CAR violation, all-unsafe abstention, or the frozen
suite timeout. An independent CPU run must replay the producer's exact action
ledger and reproduce its complete scientific view. Local contract, evaluator,
sampler-regression, shell, and JSON checks pass 46 focused tests with four
optional desktop NumPy/Pillow skips. Allocation-backed execution is next.

Producer `41806`, CPU exact-action replay `41808`, and validator `41810`
complete ADR-0195 from immutable commit
`24de03d0b65c884f2a07fef9aaaa11805d6e8235`. Exact scientific-view replay
passes. The model is invoked at policy queries zero, one, and two. At the first
two queries, all 13 terminal candidates are predicted safe and the
least-modifying choice is nominal (predicted primary risks `-0.4642331` and
`-0.2098709`). After ten executed actions, query two has zero predicted-safe
candidates; its least predicted primary risk is `+0.0647544`. The method
therefore abstains without executing an eleventh action.

No raw robot contact occurs in any group, paper CAR passes with maximum active-
obstacle L1 displacement `2.2748740e-11 m`, and the run neither times out nor
uses a QP. Native task success is false because abstention occurs before any
goal atom is satisfied. Thus this true inference loop fixes the prior unsafe-
fallback failure and avoids collision, but it does not preserve task completion
on E05. The limiting component is now candidate/intervention authority at the
first all-unsafe query, not failure to call the model repeatedly. Validation
file/payload SHA-256 values are
`68a00ee01eb2498b557aa58141cf331193634b6b79f7b32eb2d53bc097d0480f`
for the producer result file and
`3648eedd0d5d8444182d46914486a57edcab323788b68b6a22748681b55105a6`
for the validation payload. The producer video is copied locally to
`/Users/quanth238/Downloads/vlsa-e05-true-full-episode-20260817.mp4` with
SHA-256 `d791b106ec77fb7997d4215a1c97766cd8498bce58a8e7e11ff2945eeedf3af8`.

## 2026-08-17: Learned future-risk SQP feasibility diagnostic

ADR-0196 freezes one inference-only diagnostic at the exact E05 query-2 state
where ADR-0195 abstained. Replay the immutable first ten executed actions,
obtain one terminal pi0.5 chunk with the registered query seed, and compare two
trust-region sequential QPs over its 15 five-step XYZ correction variables:
EE row 0 only, and the complete EE/palm/L5 rows 0--4. Both arms use the same
frozen compact 7D worst-future-violation predictor, zero risk margin, identical
action bounds, six nonlinear relinearizations, central finite differences, and
line search.

This is a feasibility test, not control. No QP output or candidate future is
executed, no simulator lookahead supplies a constraint, and there is no model
training, new label, released AEGIS EE-QP, or safety claim. A CPU replica must
reconstruct the same query state, replay the exact producer terminal nominal,
and reproduce every serialized-critic prediction and SQP result. Local config,
source, shell, and focused regression checks pass 14 tests with the synthetic
OSQP case skipped only because desktop Python lacks allocation numerical
dependencies. Allocation-backed producer/replay/validation are next.

Initial producer `41817` is pre-simulation apparatus history. It exits before
creating a run root or starting the policy server because the allocation-side
`nvidia-smi` receipt command fails under `set -e` without reaching its explicit
H100 diagnostic. Capture that command failure explicitly and retry from a new
immutable source/run identifier; no state, model, nominal action, SQP setting,
constraint, or scientific gate changes.

Receipt-instrumented retry `41819` fails identically before artifact creation,
again on `worker-2`, with an empty exact Slurm log. Two consecutive node-local
preflight failures are sufficient to exclude only `worker-2` for this small
diagnostic. `worker-1` remains live and mixed. This is resource routing only;
the frozen scientific protocol remains unchanged.

## 2026-08-17: Late-flow pullback QP diagnostic

ADR-0197 replaces the unfinished post-terminal ADR-0196 apparatus with the
specific inference-only question required by the method. At the immutable E05
Query-2 abstention state, replay only the ten already validated prefix actions,
then numerically differentiate the frozen compact EE/palm/L5 future-risk critic
through the final two unchanged pi0.5 Euler steps. The 15 variables are
physical output-displacement coordinates for first-five-action XYZ at late
flow step eight. Three 13-row terminal batches carry centered derivative probes
only; they are not the frozen candidate bank and no discrete row is selected.

Solve one minimum-L2 multi-constraint QP with rows 0--4 kept separate, a 0.25
action-unit infinity trust region, zero risk margin, and a `1e-4` linearized
tightening. Terminalize the proposed correction once and rescore the nonlinear
frozen critic. Accept only predicted feasibility in all EE/palm/L5 rows. Do not
execute the QP output, simulate a candidate future, collect a label, train a
model, run the released EE-QP, or make a control/safety claim. This diagnostic
tests whether continuous late-denoising authority can cross the critic's
predicted boundary where the fixed bank abstained. It does not test Query-1
foresight or task completion.

Focused local contract and terminal-branch regression tests pass 18 tests with
three optional numerical skips. The repository-wide desktop gate still fails
only because the desktop Python lacks NumPy in 20 historical test modules; the
allocation preflight must run the new NumPy/OSQP case in the registered H100
environment before inference.

H100 job `41830` completes ADR-0197 on `worker-1` in `00:01:27` from
immutable commit `ddf36cbf3e039dd23f458b83a2175e16c7b1e1bd`. All ten
allocation tests pass, including the numerical Jacobian and tightened OSQP
case. The exact Query-2 nominal EE/palm/L5 predictions are
`[+0.0976876, -0.0569615, -1.8494414, -2.0890114, -4.3970320]`.
The one hard multi-constraint QP is primal infeasible in the registered 0.25
trust box, so no nonlinear proposal is terminalized and no action is executed.

The cause is not conflict with palm/L5. For diagnostic EE row 0 the pullback
Jacobian L1 norm is only `0.2740811`; even the unattainably optimistic corner
bound `0.0976876 - 0.25 * 0.2740811 = +0.0291673` remains unsafe. Palm and all
L5 rows are already predicted safe. Thus the current late-flow correction has
insufficient authority to satisfy the oversized released-EE-proxy head at this
state. Removing row 0 would make the unchanged nominal feasible and the QP
would return zero; it would not demonstrate collision avoidance or task
progress. No data, label, candidate future, or QP action was generated.

The result file/payload SHA-256 values are
`a7160ae4b938519f484382616f0806e5ea97091ead37829ffb85e43c8c173700`
and `1db92e6de8f75e18e30254ab971d66001a34d56b49aa59b0efc42b4ade26542c`.
This closes the one-query diagnostic as predicted-infeasible and does not
authorize another simulator run.

## 2026-08-17: Smooth late-flow waypoint authority diagnostic

ADR-0198 freezes one inference-only follow-up at the exact ADR-0195 E05
Query-2 abstention state. It does not recollect data, train a model, execute a
candidate, or step candidate futures in the simulator. Instead it generates
504 deterministic nonzero two-control-point cubic-Bezier waypoint deformations
plus the authoritative ordinary pi0.5 nominal. Every route has zero endpoint
correction, remains inside the `0.25` training-support trust box, passes through
only the final two unchanged flow steps, and is scored only after decoding and
clipping to executable terminal actions.

The released oversized EE ellipsoid is recorded as a diagnostic proxy rather
than a hard physical constraint. Predicted physical feasibility is defined by
palm and the three L5 rows; L6 remains an explicit unsupported audit. Candidate
selection is frozen before outcomes as predicted physical feasibility followed
by minimum intervention and temporal roughness. Two independent H100 replicas
must reproduce all candidate hashes, scores, and the selected route exactly.
This diagnostic isolates nonlinear waypoint authority from the failed local QP;
even a pass cannot authorize execution or a safety/task-success claim.

H100 array `41851` completes both 505-route inference replicas in about two
minutes each with zero candidate simulator rollouts and zero executed actions.
Both replicas produce the same qualitative and selection result:

- palm/L5-predicted-safe routes: `505/505`;
- diagnostic-EE-proxy-inclusive safe routes: `0/505`;
- same minimum-risk route: `waypoint_0136`;
- producer minimum per-row risks:
  `[+0.0954336, -0.0667797, -1.8722836, -2.1029776, -4.4163186,
  -1.6140208, -1.0726217]`;
- producer best route effective correction: L2 `0.0659765`, Linf `0.0414299`;
- minimum-intervention physical selection: unchanged nominal.

The producer ordinary-terminal hash
`d36d9c55c82f05dcc31a605f016a68ff345ffffa933f568eb75d02566ef54050`
exactly matches the immutable ADR-0195 Query-2 record. The replica regenerates
a slightly different ordinary hash even though all 504 stored nonzero effective
candidate chunks are exactly equal. Since nominal action is part of the critic
feature, the maximum per-row prediction disagreement is `0.0031784`. Strict
validator `41852` therefore fails exact scientific-view equality. Producer and
replica result file SHA-256 values are
`da1242fb9dad3b3d1ba05b37f2177ae223107ca055ad6ada1e31d907f9662037`
and `af1a2f6410750ebd333aca9cd1dbf96f3d0e9075bed5d0c66e302c03699b2a4f`;
payload hashes are
`7ea0e1deed1bf5a6c0735fa9166492de42145ffec302de67d05d104aacf1d575`
and `1a7d69a7ed7dac9849479f792f5fc3b4cf0ab89934b8f4980eb59814dbd987c2`.

Scientific decision: smooth waypoint parameterization is not the current
blocker. Palm/L5 already accept nominal, while the diagnostic EE proxy remains
unreachable. No route is executed and no further simulation is authorized from
this result. The unresolved Query-1/Query-2 failure requires a longer deployed-
horizon target, not more waypoint samples at Query 2.

Full tight-EE producer/replay arrays `41906/41907` complete all four frozen
cases exactly. The contact-free E01 control has no raw EE-group contact and
minimum normalized slacks palm/finger1-base/finger1-pad/finger2-base/
finger2-pad of `0.753981/1.875621/4.037093/2.410093/4.194896`; the released
proxy nevertheless overlaps on 858 contact-free internal samples. Validator
`41908` fails before reading results because its Python environment requires
unittest discovery rather than module-name loading. Repair only that test
launcher and validate the immutable producer/replay artifacts in place.

## 2026-08-17: Contact-aligned tight end-effector geometry repair

ADR-0199 removes the released AEGIS EE ellipsoid from physical-constraint
authority. The old grip-site ellipsoid with semiaxes `[0.06, 0.12, 0.11]` m
remains only a visual/false-unsafe comparator. Its replacement is a union of
five independently fitted, contact-aligned compiled collision primitives:
`gripper0_hand_collision`, both finger-base collision geoms, and both finger-pad
collision geoms. Meshes use certified compiled-vertex MVEEs; registered
primitive geoms use their closed-form enclosing ellipsoids. Contact outcomes do
not enter fitting, and the five rows are never collapsed into one ellipsoid over
the empty finger space.

The frozen real-simulator audit replays E09, E17, E05 and contact-free E01 from
the immutable Table-1 action ledgers. It measures exact ellipsoid-versus-compiled
box radial slack and raw MuJoCo contacts at every internal substep. Passing
requires exact independent replay, all expected palm/finger contact groups,
verified primitive enclosures, zero physical false-safe samples, and positive
clearance for every tight primitive in the contact-free control. E09 also
renders the released proxy and five tight primitives at the first observed raw
EE contact frame. This geometry gate does not retrain the MLP or authorize QP
or control.

Initial paired canaries `41889/41890` fail before simulator construction under
Python 3.8 because one test used parenthesized context-manager syntax. Repaired
canaries `41899/41900` pass the allocation tests and construct the exact E09
simulator, then stop before replay because the new config accidentally used an
unregistered `1e-9` Khachiyan convergence tolerance. The repair restores the
already validated compiled-geometry fit settings (`1e-4`, 20,000 iterations,
`1e-9` numerical padding) and the archived 1024-pixel pairing resolution. Both
attempts are pre-replay apparatus history and provide no geometry outcome.

Paired E09 canaries `41903/41904` then complete all 143 archived actions and
3,575 internal physics steps with identical scientific hashes. They reproduce
28 palm-contact and 24 finger-1-base-contact samples, all contact points lie
inside their matching tight primitive, and every per-group physical false-safe
count is zero. The tight five-piece volume sum is `0.4475704` of the released
proxy volume; the released proxy overlaps on 762 contact-free internal samples
within this episode. The canary receipt is not yet a formal pass because the new
config confused five actions per VLA chunk with the already registered 25
MuJoCo substeps per executed action. Correct only that fidelity assertion before
the remaining cohort; no geometry or outcome changes.

Full producer `41906`, independent replay `41907`, and validator `41918` pass
ADR-0199. All four scientific views are exact. Across the three contact cases,
raw-contact sample counts are palm `866`, finger-1 base `1,139`, finger-1 pad
`171`, finger-2 base `489`, and finger-2 pad `24`; every per-group physical
false-safe count and every contact-point-outside count is zero. Contact-free E01
keeps positive normalized slack for all five tight primitives, while the old
released proxy overlaps on 858 contact-free samples. The five-piece volume sum
is `0.4475704` of the released proxy volume. The exact E09 raw palm-contact
frame is action 27/substep 21; its preview SHA-256 is
`39e52577d6a26f51bbf8934c6ce7ac87e7ad6c71f7f2aff0f1f7c17854194cce`.
Validation file/payload SHA-256 values are
`80a8e24e9664a48a3cbf944cb97e9a27d0624f3c860cdb20e22abe54ae470624`
and `5f6926c5c0c1d992cfd7aebfe221f9548ab9acabe80b1c3f63292d9a604ac03d`.
The tight geometry replacement is authorized; retraining the old proxy-target
critic and control remain separate, unperformed steps.

## 2026-08-17: Tight five-action prefix-risk relabeling

ADR-0200 is implemented as a new opt-in path without deleting or changing any
historical artifact. The config binds all 24 immutable ADR-0177/0180 episode
roots, preserves their 14 train / 4 validation / 6 already-opened diagnostic
test split, and reuses the frozen 13-name bank. Each case restores the original
state and executes only the candidate's five stored actions through unchanged
OSC. Continuation, terminal hold, QP, waypoint search, new policy inference and
candidate multiprocessing are excluded.

The physical target contains independent palm, finger-1 base/pad, finger-2
base/pad and L5 rows; L6 is retained as a diagnostic. The released oversized EE
proxy is absent. Local JSON, shell, Python 3.8 syntax, focused prefix/default-
continuation regression and tight-geometry tests pass. The next step is a
single-case sequential producer/replay Slurm canary; only exact equality
authorizes the remaining 23 cases.

Sequential canaries `41944/41945`, full producer/replay arrays `41954/41955`,
and validator `41956` complete ADR-0200. All 24 cases and 312 five-action
candidate prefixes reproduce exactly, every primitive certificate passes, and
per-group represented-geometry physical false-safes are zero. Diagnostic
Q-only training is authorized; action correction, QP, a paper-scale claim, and
an untouched-test claim remain unauthorized.

The prefix-only boundary evidence is concentrated in L5. Two-sided
train/validation/diagnostic-test root counts are L5 `6/2/4`, palm `2/0/0`, and
diagnostic L6 `1/1/0`; all four tight finger rows are safe-only. Global safe
support is present in `12/14`, `4/4`, and `6/6` roots, respectively. Thus the
clean next experiment is a matched diagnostic prefix-risk predictor with L5 as
the supported head and the other rows reported as auxiliary diagnostics—not a
multi-link action-correction claim. Validation file/payload SHA-256 values are
`96b52b5a494351ca8e5f27d32908d4072cfbda8a5f63a4d37e38449ddfaf1ff5`
and `85bc470ca24587d54b0b6cf622edb67c5ac1b7b69f217ea528023e5919721c42`.

ADR-0201 freezes the resulting training experiment without new simulation. It
reuses the established compact shared 7D feature and 32x32 MLP, fits all ten
tight/distal rows jointly on the 12 initially-safe training roots, and retains
the two initially-unsafe roots as recovery diagnostics. Normalization and
state/constraint balancing use training data only; weight decay is `1e-4` and
the fixed final epoch is selected without validation tuning. L5 is the only
supported boundary head. Tight palm/fingers and L6 remain diagnostic, and the
already-opened six-case test cannot become an untouched claim. The next step is
two parallel independent H100 fits followed by a CPU exact-result validator.

Independent H100 fits `42041/42042` complete in 24/23 seconds with identical
model metrics. CPU validator `42043` fails before comparison because it calls
the H100-only allocation receipt on a CPU allocation. The two training results
are immutable scientific inputs. Repair only the validator receipt and bind the
accepted training commit explicitly; do not retrain or change data, features,
loss, seed, epochs, or metrics.

CPU validator `42045` passes the repaired receipt and confirms exact independent
training. Model SHA-256 is
`135a97ba840b3299f4e8cd74c2d7e76f6b5e5a0938a6af0379a5258a9aed2102`.
L5 validation/test diagnostic RMSE is `0.342721/0.213667`, near-boundary RMSE
`0.142140/0.179642`, and Spearman rank `0.754264/0.858285`. The model preserves
useful action ordering but its zero threshold is optimistic: L5 false-safes are
`2/6` validation-unsafe candidates and `14/23` test-unsafe candidates. The
least-intervention predicted-safe rule yields exact-safe selections in `3/4`
validation roots and `4/6` diagnostic-test roots.

Every tight EE candidate in validation and test is actually safe, so its zero
false-safes and high rank do not test collision-boundary transfer. L6 test is
also safe-only. ADR-0201 therefore validates a deterministic compact prefix-risk
ranker, not a safe acceptance gate. Correction, QP, denoising, closed loop,
paper-scale transfer and an EE-collision prediction claim remain blocked.
Validation file/payload SHA-256 values are
`1bf793a4f497d004ce9a4542b2bc427407e7c9d30f331183d5f203a6792f5bda`
and `6a8aeed21494ce09c3129d77b9d001a0480de09a1e42b75677f11a86f5fa7ac0`.
