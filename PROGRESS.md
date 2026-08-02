# AEGIS SafeLIBERO reproduction and static-Poisson feasibility

## Objective

The paired translational SafeLIBERO reproduction is terminal. The active work
now tests, as an opt-in research arm, whether simulator-ground-truth static
obstacle geometry plus a full-body Poisson-CBF safety filter can prevent the
arm-link contacts that the released end-effector-only AEGIS model misses.

## Current gate

`P01-static-poisson-runtime` is the sole active gate. It depends on the now
verified `A05-analysis` gate. `A06-openvla` remains optional and pending; no
OpenVLA result is inferred from the pi0.5 population.

P01 must preserve the released Table-1 baseline and make every new controller,
measurement, and result schema opt-in. The first claim-bearing comparison must
pair exact settled state, policy noise, nominal action history, controller
horizon, and case identity. Optimizer clearance and simulator contact/clearance
remain distinct measurements.

## Verifier promotion of A04 and A05

Independent verification on 2026-07-31 promoted `A04-population` and
`A05-analysis` in dependency order.

### A04-population: passing

- Terminal run:
  `vlsa-table1-contact-authority-population-20260718a`.
- Runtime source commit:
  `1592aa59361f431ba96c6ddcbebcb596f6c20853`.
- The prepublish validation receipt has file SHA-256
  `05df4c759a478069f1df3fe318c6f6237e65aeb667212f208ec94e2ab2a13eaf`,
  payload SHA-256
  `a5afb0d99fe7968be974a6b7d53489b0c5d5554876fe5fa651d3862cee03b9ff`,
  and status `validated`.
- It records all 32 Slurm array tasks as `COMPLETED`,
  `complete_paired_population=true`, `no_results_dropped=true`, and exactly
  3,200 result artifacts with status `complete`. The result inventory SHA-256
  is `f7f28e88b43ac62c183284ef55cff61bd1a195109c02bf67058ae73845269d50`.
- The population summary has SHA-256
  `c2702d40d53436b89e48a5e68d743a76403d076a5c5539fa6c74a850af698330`.
  It binds 1,600 paired cases, two arms, 3,200 results, 32 task-level groups,
  no dropped results, and accepted-result ledger SHA-256
  `28822a58683cc54dab915e6f6bc56005bdd51f779138509c71a7c1aae2969285`.

The complete reproduced suite metrics are:

| Arm | Suite | CAR (%) | TSR (%) | ETS |
|---|---|---:|---:|---:|
| pi0.5 | Spatial | 15.50 | 58.75 | 201.52 |
| pi0.5 | Goal | 21.75 | 55.75 | 206.82 |
| pi0.5 | Object | 22.00 | 55.00 | 217.33 |
| pi0.5 | Long | 14.00 | 38.50 | 472.05 |
| pi0.5+AEGIS | Spatial | 77.00 | 74.25 | 183.77 |
| pi0.5+AEGIS | Goal | 82.25 | 78.00 | 174.82 |
| pi0.5+AEGIS | Object | 78.75 | 81.25 | 197.88 |
| pi0.5+AEGIS | Long | 72.25 | 34.00 | 494.58 |

Across all suites, pi0.5 is `18.3125 / 52.0 / 274.42875` and
pi0.5+AEGIS is `77.5625 / 66.875 / 262.759375` for CAR/TSR/ETS.

### A05-analysis: passing

- The exhaustive case ledger contains exactly 1,600 unique paired rows and
  has SHA-256
  `2280c3f1dc7e25755f650e7af0deb140263edec0679e3395db13a539b5a6a780`.
  Its source failure report has SHA-256
  `6d1d74040c55512ad9035cdd4a2976c6eeae15c6d32e17aff6164164cdc44de4`.
- The strict indexed gallery contains 1,600 paired case cards and 3,200 video
  entries; its SHA-256 is
  `40781fa0a817931ad23bb12b2b7be2b858e16033f97ed18b7f76b86d291b2bb2`.
- The complete-population diagnostic report has SHA-256
  `9457175a69b84895cd2c8aa18b3fe29e291992e80178e9a692f66588a4492430`;
  its 1,600-row case audit has SHA-256
  `acfbb7d3d3f2ebfccb556807a8f980fcc35f0510b7ef148deda5a7080573a7e8`.
- An independent verifier recomputed every case-row payload hash and rehashed
  all 3,200 mirrored MP4s, totaling 2,263,857,516 bytes, with zero mismatch.
  The adversarial validation addendum has SHA-256
  `cb854b3d25be6452af67ec62c3d3b404883bac428d31d59b34a9238aab4fb259`.
- The analysis retains all 359 AEGIS CAR failures and all 530 AEGIS task
  failures. It distinguishes paper CAR from sampled MuJoCo contact, identifies
  109 link-5/link-6 contact cases, and reports that 257 cases satisfy the
  registered strict useful-rescue gates. There are zero strict-zero-
  translation episodes.

Publisher job `28940` timed out only after writing the validated prepublish
receipt, exact population summary, 1,600-row failure ledger, and 3,200-entry
gallery; it did not write the planned final publication receipt. Gate
promotion does not treat the timeout itself as success. It relies on the
complete prepublish evidence above plus the later independent result, case,
aggregate, and video-integrity audits. The timed-out attempt remains preserved
as an apparatus event.

## P01 prerequisite evidence

- Branch: `codex/full-body-poisson-cbf-feasibility`.
- Feasibility review SHA-256:
  `596b59661beaf4746781ef57afaf4aa6b09a7225db91996b183c5c587109b63d`.
- Runtime prerequisite probe SHA-256:
  `7532bdce73597d1e0581e9af891846df289b80a9dc7777a8fe6acc0585249a38`.
- H100 Slurm job `33249` completed on `worker-mig-3g40gb-0`; its result
  SHA-256 is
  `a5d5a5cd17564d16873607b4cb7684d819e075a6088ba26932a72b99d015ec5f`.
  All eight runtime checks passed, including seven-arm-DOF joint-velocity
  control, 8-dimensional controller actions, link-5/link-6 arbitrary-point
  Jacobians, active-obstacle collision geoms, and the physical velocity scale.
- The probe also established that independently settling OSC and
  JOINT_VELOCITY controllers changes the robot state. P01 must therefore
  settle once under the registered OSC apparatus and restore that exact state
  into every experimental arm.

## P01 first-canary gate history

- Immutable run `vlsa-poisson-link56-first-canary-20260731c` used clean commit
  `80ebe9a68ffb901f2cd45f3c4f354a8dc16aab4f`.
- Numeric job `33447` and exact-parity job `33449` completed. The numeric
  artifact file SHA-256 is
  `8f6192b3cb266dbd9b5f3274af6c97e6349c28e012cf4a1e5335d6426fe6743c`;
  the parity artifact file SHA-256 is
  `06318ab6aa7f665dfdd06c02f39035a4df670a4ed0807e493c3449e002633120`.
- Identification job `33451` completed all 5,925 callbacks over 237 historical
  actions, then failed after 17,684.27 seconds in a post-run validator that
  incorrectly coerced the typed cadence tuple `measurement.first_index` with
  `int()`. Its immutable failed artifact has file SHA-256
  `5fe2aef4eb93c28452fde86b87d0b925b95544fe43c2d16b59550e6fd5a2a318`
  and payload SHA-256
  `16c1feccb78d7da013f6063be09c27b018f77c52c0a5f46bf0a76942ac796925`.
  This is an apparatus failure and carries no scientific result.
- The rerun change validates typed three-level cadence directly, executes the
  complete post-run path in tests, reconstructs first contact from the raw
  MuJoCo ledgers, reconstructs the registered warning from the raw per-geom
  Poisson trace, binds exhaustive samples and drift thresholds to the frozen
  protocol, and rechecks exact-parity hashes in the active consumer.
- Warning lead now uses physical phase boundaries: post-state contact `N` is
  compared at boundary `N`, while live-solver contact `N` is compared at
  boundary `N-1` in trace-index coordinates. This prevents a simultaneous
  live-solver contact from being mislabeled as a one-substep-early warning.
- A positive 500 Hz lead is not sufficient by itself. The active gate also
  requires at least one scheduled 100 Hz filter-update boundary after the
  warning state and strictly before the contact boundary.
- Adversarial validation now rejects cross-link sample-ID substitution,
  contacts outside the resolved selected-obstacle pair authority, and physical
  contacts absent from the corresponding candidate-contact ledger. It also
  rejects a forged negative CBF residual when the recorded Jacobian and joint
  velocity imply a different directional derivative, or when diagnostics in
  one callback claim different instantaneous joint velocities. Active physics
  also rebuilds and exactly matches the shadow obstacle, contact authority,
  resolved geometry, field bundle, sample ledger, arm DOFs, and settled-state
  hash before either arm can start. The final local structural gate passed 412
  tests with 130 expected allocation-only skips; the focused Poisson suite
  passed 236 tests with 112 expected local MuJoCo/NumPy skips. A fresh H100
  numeric gate with zero skips remains mandatory on the frozen revision.
- Because source changed, numeric, parity, and identification must all rerun
  from one new clean commit and unused immutable run root. Active physics is
  permitted only if the new identification artifact independently validates
  and reports a strictly positive phase-correct pre-contact warning.

## Immediate P01 verification order

1. Freeze an immutable targeted-case manifest from the verified 109 link-5/6
   population, with the 60 positive-proxy/no-settled-contact cases as the
   primary static stratum and all 109 retained in intent-to-treat reporting.
2. Add an exact-action, shadow-only replay that records qpos, qvel, obstacle
   pose, arbitrary-point Jacobians, every physics-substep contact, and
   simulator clearance without changing executed actions.
3. Validate conservative collision-geom voxelization, sample coverage,
   buffered-set semantics, Poisson residuals/gradients/interpolation, and
   finite-difference Jacobians on synthetic geometry before a SafeLIBERO case.
4. Validate joint-velocity scaling and an adapter-only matched controller
   before enabling the hard CBF-QP.
5. Run the first H100 active canary only after shadow-mode action and outcome
   parity pass. Report infeasible QPs and all failed cases; never drop them.

No learned perception, moving-obstacle model, grasped-object model, or learned
steering controller is authorized by P01. Those are later gates contingent on
the static simulator-oracle result.

## P01 tracking-validity adversarial audit

Run `vlsa-poisson-link56-first-canary-20260731d` produced passing immutable
numeric and exact-parity gates on clean commit
`ad738550ef3a8fef18cab73139a4d1ecbe0d18c1`. Numeric job `33491` ran 127
tests with zero skips; its file and payload SHA-256 values are respectively
`72581ba35e7789269f9eb1418658089ef1aa37a8454fb3af19706cce2fee3b74`
and `351c53b377b68ef83deda59fb65a910ea7861063d49d6c7a22823688f9e96a70`.
Parity job `33492` reproduced all 237 actions and 5,925 callbacks exactly;
its file and payload SHA-256 values are respectively
`2d9856060265845a35c8fc8581610f56959c775af79e9f74b6fae7a31e092ab7`
and `93c648ba57c4195c4c539648286fb45b9495297d4f78b8d3ab80d02cc9cb17d4`.

An adversarial audit then proved that the compact episode validator accepted
an `executed` result whose reported joint-velocity tracking Linf and RMSE both
crossed the registered apparatus thresholds. The live runner already stops on
that condition, but a rehashed forged compact result could otherwise reach the
pair eligibility logic. The validator now rejects every `executed` arm with a
tracking-threshold crossing, and a regression reproduces the former forgery.
The complete local structural gate now passes 413 tests with 130 expected
allocation-only skips; the focused Poisson suite passes 237 tests with 112
expected local MuJoCo/NumPy skips. Both changed Python files also parse under
the Python-3.8 grammar used by the H100 evaluation environment.

Identification job `33494` was launched from the superseded clean commit.
Regardless of its eventual terminal state, neither it nor the earlier gates
can authorize active physics after this source change. No active canary was
submitted. Numeric, parity, and complete shadow identification must rerun from
one new clean commit and unused immutable run root.

The user's engineering ladder is sequential, not advisory. The coupled
protected-sample field/Jacobian audit must pass before the one-step
counterfactual; the counterfactual must pass before the manual static-box
trial; and the manual trial plus an independently validated adapter-only trial
must pass before any VLA-linked active canary. The earlier exception that
would have allowed one exploratory active canary before those gates is
withdrawn. No active physics was submitted under that exception.

## P01 complete-run integrity audit and superseded jobs

Live Slurm inspection on 2026-08-01 established the exact terminal states of
the superseded jobs. Identification job `33494` completed with exit `0:0` on
commit `ad738550ef3a8fef18cab73139a4d1ecbe0d18c1`, but its source predates the
tracking and deep-trace fixes, so its preserved artifact is implementation
evidence only. On later clean commit
`c7dd260dcb0023a3cc81b1209aa3b521ae6e8be1`, numeric job `33575` and parity
job `33576` completed with exit `0:0`. Their immutable artifacts have these
file/payload SHA-256 pairs:

- numeric: `5f14fd126cd7e3839285c7e2b32b33aada90bf3925ad185f158c0fb3c3d04ef7` /
  `5f15c8c1150dfcef833207577327e34109a854380ca88e7a64af4000734fb2f5`;
- exact parity: `aeeb0132cca102a153e5e6cf12c19f188610b7fe56e7af5515b37e35e71b09b4` /
  `8bf1e45762ce273b3be558efc1ae9a51a4bf174eaacbadd836b3709b58454e5a`.

Identification job `33577` from `c7dd260` was inspected by exact job ID and
canceled after 11 minutes 48 seconds because a new adversarial audit had
already superseded its validator. It produced no final identification
artifact. Nothing from jobs `33494` or `33575`--`33577` can authorize active
physics on the next commit, and all old files remain preserved.

The current complete-run gate now treats `run_receipt.json` as a receipt-last
commit marker. Before it can be published or resumed, it deeply validates both
arm traces, reconstructs the paired result, verifies file and payload hashes,
and binds the exact 237-action/5,925-callback canary exposure. The audit closes
the concrete rehash attacks found during independent review:

- physical contact subsets, counts, flags, first records, and contact-clamped
  `D_sim` are reconstructed from the settled/live/post MuJoCo candidate
  ledgers;
- the selected moka-pot geometry, every declared robot--obstacle collision
  pair, literal link-5/link-6 identities, exact Panda grip site, settled state,
  obstacle position, native BDDL goal, and translation-only source actions are
  fixed to the registered canary;
- every issued PSF command has one solved OSQP record, positive provider-time
  `h` and `D_opt`, exhaustive protected-sample counts, bounded optimizer
  diagnostics, and an execution record that binds nominal velocity, filtered
  velocity, correction, normalization, and unchanged gripper;
- every completed 2 ms callback has one exhaustive realized-CBF evaluation,
  while restore, obstacle-static, joint-velocity, and tracking tolerances are
  fixed rather than trusted from mutable result fields;
- an `executed` arm cannot contain a fail-closed attempt, missing filter
  update, invalid tracking, nonpositive barrier value, hidden contact, or
  safety-by-no-execution path.

The QP now canonicalizes a solver result to the hard joint-velocity bounds
before recomputing residuals and returning the command. This prevents a
few-ulp OSQP bound tolerance from creating different declared and executed
commands. Independent adversarial audits found no remaining locally
reconstructible claim-impacting false positive or producer/schema mismatch.
The focused four-module suite passes 61 tests; the broader Poisson suite passes
263 tests with 112 expected allocation-only skips. The final full local
structural gate passes 439 tests with 130 expected allocation-only skips after
the collision-pair assertion was added.

The v2 trace still does not embed raw protected-sample coordinates, obstacle
OBB poses, field arrays, compiled model arrays, or upstream manifests and
checkpoint bytes. Consequently, the validator can recompute typed ledger and
arithmetic consistency but cannot independently regenerate the exact positive
sample-to-OBB `D_sim` magnitude from the run tree alone. MuJoCo nonpositive
contact remains the safety authority. The run contract, selected manifest row,
runtime protocol, checkpoint, historical result, and all three H100
prerequisites must still be checked byte-for-byte outside the run receipt.

The next immutable root is
`/mnt/data/quanth/experiments/vlsa-aegis-poisson-feasibility/vlsa-poisson-link56-first-canary-20260801b`.
After the final clean commit is pushed, the exact next operation is a fresh
read-only VinUni preflight, followed by clean remote checkout synchronization
and new numeric plus exact-parity submissions on that same commit. Shadow
identification remains blocked until both gates are terminal and deeply
validated. Active physics remains blocked until identification also proves an
actionable warning at a scheduled 100 Hz update strictly before contact.

## P01 frozen-replay state-evidence hardening

Two later immutable roots are preserved as apparatus history. In root
`vlsa-poisson-link56-first-canary-20260801b`, numeric job `33605` completed,
while parity job `33606` failed before simulation because it was given one
historical result file instead of the registered population root. Root
`vlsa-poisson-link56-first-canary-20260801c` corrected that operator error on
clean commit `8cce5bc4241f70cd608217759246353efc964b0b`. Numeric job `33607`
completed 127 tests with zero skips; its file/payload SHA-256 pair is
`d50a1164c2ad4be7caf69ec187f9b49ec96362f1098177eeedd80f85d453b1b0` /
`6956cc754727f15c92f61c0056cccd6cd14eee3ad387c71675722edc67dfc170`.
Parity job `33608` completed all 237 actions and 5,925 callbacks; its
file/payload pair is
`cb66e8aaf092deb673e6550daa7135b6339e3c2586549bd4bc39e3c22df2388c` /
`c786d2a538fae2fb26cd82d1cd3276d319ebcc3b7b457505173b8cd8046c2fec`.

Identification job `33610` was inspected by exact job ID and canceled after
10 minutes 2 seconds because an adversarial audit found that its source
serialized callback and terminal state summaries without binding every value
to an independently consumed upstream ledger. It wrote no final
identification artifact. No active canary was submitted, and all old artifacts
remain preserved.

The replacement parity-v2 contract records and validates two complete state
ledgers: all 237 action-boundary hashes are bound directly to the frozen
historical replay, and all 5,925 callbacks use MuJoCo's official
`mjSTATE_INTEGRATION` representation. Identification records before/after
callback hashes, proves read-only equality at every typed callback index, and
must exactly match parity. The active consumer independently repeats the
237-state historical binding and the 5,925-state parity binding. Coherently
rehashing both ordinary and callback parity ledgers, or both parity and
identification summaries, therefore cannot replace the frozen replay oracle.
The obsolete `sim.get_state().flatten()` callback summary was removed rather
than retained as a second, non-authoritative simulator-state definition.

Adversarial tests cover missing, reordered, middle-row, terminal, cadence,
trace-shape, and collusive-rehash attacks. The broader Poisson suite passes
264 tests with 112 expected allocation-only skips. An independent refreshed
audit passes 36 focused tests with 8 expected skips, Python-3.8 grammar, and
the full `./init.sh` gate at 440 tests with 130 expected allocation-only
skips. P01 remains active: these are apparatus guarantees, not safety results.

The next unused immutable root is
`/mnt/data/quanth/experiments/vlsa-aegis-poisson-feasibility/vlsa-poisson-link56-first-canary-20260801d`.
After the hardened source is committed and pushed, numeric and parity-v2 must
rerun on that exact clean commit. Identification-v2 may run only after both
terminal artifacts validate completely. Active physics remains forbidden
unless the final identification proves a warning that a scheduled 100 Hz
filter update can consume strictly before physical contact.

## P01 realized-motion eligibility hardening

Root `vlsa-poisson-link56-first-canary-20260801d` ran on clean commit
`b3abd5c93e7789267f9e709ed8a9074cbeed18b4`. Numeric job `33640`
completed with exit `0:0` in 1 minute 37 seconds. Its immutable artifact has
file/payload SHA-256 values
`8c3dd38e4698a5d89c84031d4817f48c25c50eea822d0803c332517eebe34a13` /
`2fdd5368636aac3b713bac61994a1caed79505a7f6a0544a85ca10b37aea83a1`;
independent reconstruction confirmed 128 tests, zero skips, one H100, and the
exact clean commit. Parity-v2 job `33641` completed with exit `0:0` in 3
minutes 10 seconds. Its file/payload SHA-256 values are
`2a957a382389a6d5ef452c61e2dca906be055ad71301deb0946174a1a511462c` /
`bcc06df4ef43435efe4864ef65b3ce02ba75280e275bfa8b89887d7001350251`.
Independent reconstruction matched the frozen historical result, all 237
action-boundary states, all 5,925 official MuJoCo integration states, exact
typed cadence from `[0,0,0]` through `[236,4,4]`, and every acceptance gate.

Before active physics, another independent adversarial audit found a distinct
claim-impacting false positive. The pair eligibility gate required nonzero
issued safe-joint motion, but did not require nonzero realized joint motion.
A coherently valid pair with `0.008` rad commanded-motion integral, nonzero
correction, and exactly zero measured joint and end-effector motion therefore
made all six prevention and task-utility eligibility flags true. This would
confuse command issuance with physical motion and could label stopping as a
useful correction.

The pair gate now also requires
`measured_joint_motion_integral_rad > 0`. Zero realized motion propagates the
reason `psf_realized_no_nonzero_arm_joint_motion` into link-5/6, all-robot,
Paper-CAR, task-success, preservation, and rescue ineligibility. The measured
integral is included in the pair's continuous-motion diagnostics. End-effector
path length remains a diagnostic rather than a universal contact-prevention
gate because legitimate null-space arm-link avoidance can move joints while
holding the end effector nearly fixed. A stronger claim that motion was
materially useful, rather than merely mathematically nonzero, still requires
a preregistered material-motion or retention threshold; this first canary does
not invent one after seeing an outcome.

Identification job `33642` was inspected by exact job ID and canceled after
12 minutes 28 seconds once this source change made same-commit active
authorization impossible. It produced no final identification artifact. No
active canary was submitted. Root `20260801d` and all logs remain preserved as
implementation evidence, but its prerequisites cannot authorize the next
commit.

The dedicated adversarial regression and all inherited pair flags pass. The
focused active/trace/schema suite passes 62 tests; the broader Poisson suite
passes 265 tests with 112 expected allocation-only skips; and the final local
`./init.sh` gate passes 441 tests with 130 expected allocation-only skips.
P01 remains active. After a new clean commit is pushed, all H100 prerequisites
must rerun in a new unused immutable root, preferably
`vlsa-poisson-link56-first-canary-20260801e`.

## P01 exhaustive protected-sample differential audit

The next pre-identification audit found that the numerical gates did not yet
evaluate every actual link-5/6 protected-surface sample at the exact settled
SafeLIBERO state. Root `vlsa-poisson-link56-first-canary-20260801e` is therefore
superseded for active authorization: numeric job `33653` and parity job `33654`
completed with zero exit codes on commit
`8fd43304cf16091035a8c9ebb2f5dd81a535fb2e`, while identification job `33655`
was inspected and canceled by exact job ID after 18 minutes 27 seconds. It
produced no final identification artifact, and no active physics was
submitted. Job `33494` is terminal `COMPLETED|0:0` but remains preserved
obsolete evidence.

Runtime protocol v2 now preregisters an exhaustive settled-state differential
audit. For every ordered protected sample, it checks all seven arm-DOF point-
Jacobian columns using central differences generated by `mj_integratePos`,
reconstructs the full-`nv` tangent with `mj_differentiatePos`, and rejects
non-arm leakage. It then checks nine registered joint-velocity directions
against the coupled `d h / d t = grad(h)^T J qdot` chain rule. The adaptive eta
ladder may select only the largest base/plus/minus stencil inside one exact
valid trilinear cell. Every rejected attempt, typed invalid-query reason, cell
identity, tangent, matrix, tolerance, arithmetic result, and classification is
retained. Missing stencils, cancellation, incomplete sample populations, or
threshold failures stop before active physics.

Shadow identification schema v3 embeds the exact sample/component ledger and
complete audit. Active trace schema v3 independently carries the registered
12-section parameter block, protected samples, seven arm DOFs, settled-state
hash, audit, and reconstructed validation receipt. The pure consumer
recomputes component coverage, the protected-sample hash, state/config/sample
binding, all finite-difference arithmetic, counts, and paired-arm equality.
Raw obstacle OBB poses and field arrays remain external byte-bound authorities,
so positive `D_sim` magnitude is not independently regenerated from the trace
alone; nonpositive MuJoCo contact remains the safety authority.

Local evidence after this hardening is: 102 focused integration tests passed
with 22 expected dependency skips; the full Poisson suite passed 281 tests
with 116 expected skips; the real-MuJoCo Jacobian module passed 8/8 using a
temporary local dependency; and `./init.sh` passed 457 tests with 134 expected
allocation-only skips. Python-3.8 parsing, `py_compile`, and diff checks also
pass. P01 remains active. A new clean commit must rerun same-commit numeric,
exact-parity, and complete schema-v3 identification in one unused immutable
root. An actionable 100 Hz warning authorizes only the mandatory one-step
counterfactual next. The manual static-box and adapter-only gates must then
pass in order before the active link-5/6 canary.

## P01 Stage-13 one-step counterfactual preflight

Stage 13 now selects its intervention state from the first controller boundary
scheduled at or after the physical warning: `B = ceil(W / 5) * 5`. This
supersedes selecting a latest boundary derived backward from contact. In the
preserved obsolete complete diagnostic, the late candidate `B = 4695` is
outside the epsilon-buffered field for 352 of 1,531 protected samples. The
warning-derived candidate `B = 4510` is diagnostic all-valid and precedes the
physical contact boundary `C = 4697` by 187 simulator substeps.

That same obsolete diagnostic indicates that the exact endpoint-derived
nominal joint velocity appears to exceed the frozen `[-0.5, +0.5]` rad/s
controller envelope on joints 2 and 4. The implementation therefore publishes
a complete typed `preflight_inadmissible_nominal_velocity` negative when the
fresh exact-boundary reconstruction confirms this condition. This completion
class executes no QP and no arm physics, and it neither clips nor projects the
nominal command into the registered envelope. It diagnoses controller-envelope
incompatibility, not a Poisson failure. The obsolete evidence cannot establish
the fresh outcome; same-commit H100 production and independent artifact
validation remain mandatory.

Stage 13 is scoped only to the preregistered one-step exact-state
counterfactual and its typed admissibility outcome. It does not establish
multi-step safety, useful retained motion, task preservation, or active-canary
eligibility. If the fresh nominal velocity is out of bounds, Stage 14 and all
active physics remain blocked. P01 remains `active`, never `passing`, pending
fresh allocation-backed evidence. Local implementation checks are not a final
Stage-13 result.

The post-fix authority audit also closes two concrete artifact attacks. The
Stage-13 protocol freezes the installed joint-velocity controller source,
configuration, Panda XML, timing, scaling, gains, indices, and actuator
identity; the independent consumer rehashes those installed files and fails
closed when any is unavailable. The physical-model v3 contract fixes
`nq`/`nv`/`na` and requires the flattened state to be exactly
`[time, qpos, qvel, act]`, so a rehashed appended state tail is rejected in
both executable and inadmissible branches. Only the seven Panda arm velocity
entries are claim-bearing; other full-`nv` values are retained as producer
diagnostics. The independent adversarial audit reports PASS. Local evidence is
114 focused tests with 14 expected dependency skips, 368 broader Poisson tests
with 117 expected skips, and a complete `./init.sh` gate of 544 tests with 135
expected allocation-only skips. These remain implementation checks, so all
upstream H100 prerequisites and Stage 13 must be produced afresh from the next
exact clean commit.

## P01 retained H100 differential-audit negative

Fresh root `vlsa-poisson-link56-first-canary-20260801g` used clean pushed commit
`ad87170549d9b865fadb89de0f99ecf054f1d50b`. Numeric job `33726` completed
`COMPLETED|0:0` on an NVIDIA H100 80GB HBM3 with 154/154 tests and all nine
acceptance checks true (file SHA-256
`bee0b6109513662e9b70704a87a70b78c37738108dbd9b0fff5d0518350e3980`).
Exact-parity job `33727` completed `COMPLETED|0:0` with 237 actions, 5,925
callbacks, and all six acceptance checks true (file SHA-256
`5c6100a8851e651c40da8c995ddeb41c8c20993919f3b7d1320946627cfaa785`).
Independent prerequisite consumer job `33730` completed `COMPLETED|0:0` and
bound those artifacts to the immutable manifest, selected row, runtime, and
historical replay.

Identification job `33731` terminated `FAILED|1:0` only after atomically
publishing a complete 346,300,863-byte schema-v3 diagnostic (file SHA-256
`8418f3f0d07cd2d30f87b13f51736db1ab6610426fc3d13acfae59a05d3c9c41`,
canonical payload SHA-256
`dda149ee6b262e1dafec790e0608e119717ff241ca33ed63447b64b9c82ca76f`).
Independent CPU consumer job `33732` completed `COMPLETED|0:0`, reconstructed
the complete failed audit, classified it `validated_negative`, and explicitly
set both Stage-13 and active-physics authorization false. No Stage-13, manual,
adapter-only, or active VLA physics was submitted.

The complete audit contains 1,531 protected samples and 13,779 eligible coupled
directions. All 13,779 coupled field/Jacobian checks pass, and the actual
analytic-versus-numeric point-Jacobian errors pass their registered absolute
and relative tolerances for every sample. The sole producer-local failure is
the shared tangent roundtrip record: columns 3 and 5 reconstruct requested
velocities `+/-1.0` as `+/-1.000000000139778`, an absolute velocity error of
`1.397779669787269e-10` against a fixed `1e-10` threshold. The same physical
displacement roundtrip error is approximately `1.39778e-16` rad. At half the
requested velocity and twice the finite-difference interval, the identical
displacement error becomes `6.988898348936345e-11` in velocity units and
passes. Jobs `33734` and `33735` independently extracted these exact retained
values from the final artifact.

This is a scale-dependent finite-precision audit false negative, not evidence
that the point Jacobian, Poisson chain rule, collision filter, or research idea
failed. It also is not positive feasibility evidence because active physics was
correctly blocked. Root `20260801g` is retained unchanged. The next protocol
change must be preregistered on a new clean commit and root, derive the tangent
roundtrip criterion from displacement-space roundoff or a scale-aware numerical
bound, retain the present diagnostic, and rerun every upstream gate. The
observed result must not be used to tune the old threshold in place. P01 remains
`active`.

## P01 runtime-v3 roundoff gate and parity-v3 boundary-zero authority

The root-g negative is preserved and not reinterpreted. Runtime protocol v3
replaces its scale-dependent `1e-10` rad/s tangent authorization threshold with
an exact-rational, two-stage binary64 scalar-hinge roundoff check. Every arm
coordinate must separately pass the registered integration and differentiation
bounds derived from `u = 2^-53` and `gamma_2 = 2u/(1-2u)`. The old velocity
threshold remains a serialized non-gating diagnostic. The root-g signature and
its scale-equivalent displacement both pass the derived bound; wrong sign, DOF
swap, nonzero unresolved perturbation, non-arm leakage, non-finite or subnormal
inputs, `1e-8` rad/s velocity corruption, and `1e-12` rad position corruption
remain rejected.

Differential audit v2 now serializes the complete source
`mjSTATE_INTEGRATION` binary64 bits and freezes MuJoCo 3.2.3 plus the Panda
seven-hinge DOF/joint/qpos topology. Exact source-state identity uses binary64
bit equality, while zero-motion observability correctly treats `+0.0` and
`-0.0` as the same mathematical position. Shadow identification v4 and active
trace v4 carry the new evidence. The Stage-13 protocol is v2; its result and
validation-receipt outer schemas remain v1 because their field sets are
unchanged and the embedded protocol identity is hash-bound.

An adversarial audit then found that parity v2 began its official-state ledger
only after the first physics callback. It could not independently bind the
settled official state used to construct the differential audit. Exact parity
therefore advances to v3 and captures, in both ordinary and callback replay
before action zero, the exact record `{physical_boundary: 0,
mujoco_state_specification: mjSTATE_INTEGRATION, state_vector_length, sha256}`.
The records must match exactly. Identification, its independent consumer v2,
the active prerequisite helper, and the Stage-13 producer/core/consumer all
bind the construction read-only audit and serialized differential state to that
external hash and length, and bind the length to physical-model v3. Exact
acceptance fields, Boolean type confusion, collusive rehashes, replay-arm
divergence, and malformed or missing authority are rejected. The final
independent audit found no remaining boundary-zero authorization path.

Root-g exact-parity job `33727` is parity v2 with six acceptance fields. It and
all other old artifacts remain immutable history and cannot authorize
identification v4 or Stage 13. The selection manifest and all 109 rows were
regenerated against runtime v3. Local post-hardening evidence is 120 targeted
tests with 8 expected dependency skips, 400 broader Poisson tests with 117
expected allocation-only skips, 18/18 real-MuJoCo Jacobian tests, and the final
`./init.sh` gate at 576 tests with 135 expected allocation-only skips. Static
Python compilation, Slurm shell syntax, JSON parsing, and diff checks also
pass. These are implementation evidence only; P01 remains `active`.

The next run must use a clean pushed commit and unused immutable root
`/mnt/data/quanth/experiments/vlsa-aegis-poisson-feasibility/vlsa-poisson-link56-first-canary-20260801h`.
After a fresh live preflight and exact clean remote sync, rerun numeric and
exact-parity-v3 gates on that same commit, independently validate both complete
artifacts, and only then submit identification v4. Stage 13 remains blocked
unless the complete identification has an actionable warning with a scheduled
100 Hz update strictly before contact. No partial artifact may be interpreted.

## P01 lean one-case H100 feasibility result

The user requested a direct feasibility test before continuing the formal
multi-stage ladder. This exploratory branch leaves every earlier artifact and
formal gate unchanged. It tests one known link-contact window with simulator
geometry and no policy queries; it does not claim task success, full-episode or
population safety, learned perception, moving-obstacle coverage, or real-time
control.

- Source branch: `codex/poisson-fast-feasibility`; clean pushed producer commit:
  `b1854d6b82836757fdfed50ad0788791a62e651b`.
- Local implementation gate: 613 tests passed with 135 expected
  allocation-only skips. Independent runtime and semantic reviews passed.
- Immutable H100 run:
  `vlsa-poisson-fast-window-20260802e`; Slurm job `33907` completed
  `COMPLETED|0:0` on `worker-mig-3g40gb-0` in `00:22:56` with an NVIDIA H100
  80GB HBM3.
- Final `result.json`: 1,801,847 bytes, raw SHA-256
  `fc74b73582b392b484be5d208c3bc20c676f66c6c4783408b99ba780284e391b`,
  canonical payload SHA-256
  `d6805095ebec1eca0617b604002e500a9c5d68ce6dcd2330451598e44f36d4ff`.

The paired case is `vlsa-t1-goal-ii-t0-e05` with static selected obstacle
`moka_pot_obstacle_1`. Actions 0--179 reconstruct exact boundary 4500 once.
Both fresh 100 Hz joint-velocity arms then restore the same complete
`mjSTATE_INTEGRATION` state and consume the same recorded actions 180--187 for
40 controller updates and 200 two-millisecond physics substeps. The adapter-
only arm first contacts the selected obstacle with `robot0_link5_collision` at
boundary 4677. The link-5/link-6 Poisson-CBF arm activates at boundary 4500,
354 ms before that baseline contact, and completes all 200 substeps with zero
literal contact from any robot collision geom to the selected obstacle.

All 40 hard QPs solved and passed independent residual, velocity-bound, and
one-step joint-limit checks. The minimum safe CBF residual is
`-2.7755575615628914e-17 m^2/s`, and the conservative full-robot surface
clearance lower bound remains strictly positive at `0.02482047062948021 m`.
The filter is not merely stopping: over the 32 materially active updates, the
maximum correction is `0.5625295109410573 rad/s`, maximum safe command norm is
`0.8410539123239245 rad/s`, safe/nominal command-motion retention is
`0.7335847886013575`, measured joint-motion integral is
`0.2036842792363826 rad`, Cartesian path is `0.06754197871836201 m`, and
target-error progress is `0.04156768990794801 m`.

Independent reconstruction from the raw contact, command, and physics ledgers
found zero discrepancy and classifies the narrow result as empirical contact
prevention with a positive clearance certificate and motion-preserving
correction. The follow-up independent validator passes 56 focused tests,
including pairing, geometry-authority, adapter-command, QP, contact,
`STOP_ONLY`, `UNCERTIFIED_CLEARANCE`, missing restore authority, hidden
settled/rollout contact, command-to-physics, clearance-certificate, and
apparatus-flag mutations; the complete repository gate passes 632 tests with
135 expected allocation-only skips. Tracking is not
certified: pre-contact adapter and full-window PSF
L-infinity errors are respectively `1.7544031695017919` and
`1.7405264821616433 rad/s`. Solver timing is also not real-time: 35/40 OSQP
solves exceed 10 ms, with mean `0.032117390175 s`, p95 `0.078985875 s`, and
maximum `0.244499723 s`; field, Jacobian, controller, and simulator overhead is
additional. The 50,000-iteration exploratory budget was chosen after the
retained 10,000-iteration convergence failure; the accepted run used at most
15,150 iterations.

Motion preservation is not task preservation. Across the complete window, the
PSF arm reduces target error by 53.33 mm, versus 116.93 mm for adapter-only.
Both arms also fail the same tracking diagnostics, and PSF tracking remains
material after the initial transient. The observation is therefore
controller-mediated empirical avoidance, not realized-velocity barrier
invariance.

This is positive evidence that full-body link-local Poisson-CBF correction can
prevent the observed arm-link contact without freezing the robot in this one
static, post-hoc window. It is sufficient to continue the research direction,
but `P01-static-poisson-runtime` remains active for broader or formal claims.
The next research priority is a shadow persistent-OSQP implementation with one
fixed 1,531-by-7 sparsity pattern, numeric matrix/vector updates, cross-step
primal/dual warm starts, all hard rows retained, and full pipeline timing. It
must keep the present full-row postcheck and fail closed while targeting 40/40
updates within 10 ms. Low-level tracking follows, then a small preregistered
multi-case evaluation that reports contact avoidance and task success
separately.

The complete artifact is mirrored locally at
`/Users/quanth238/personal/Research/probe_vla/output/vlsa_poisson_fast_feasibility/vlsa-poisson-fast-window-20260802e/result.json`.
The exact handoff validation command is:

```bash
python3 scripts/validate_poisson_fast_feasibility_artifact.py \
  --result /Users/quanth238/personal/Research/probe_vla/output/vlsa_poisson_fast_feasibility/vlsa-poisson-fast-window-20260802e/result.json \
  --expected-job-id 33907
```

## P01 full recorded-episode feasibility protocol and result

The next test asks the narrower result's unresolved question: can the same
link-5/link-6 3D Poisson-CBF correction prevent the arm-link collision and
still finish `put_the_bowl_on_the_plate`? The smallest valid integration replays
the exact successful AEGIS/OSC prefix actions 0--179 once, snapshots complete
`mjSTATE_INTEGRATION` state at pre-contact boundary 4500, and restores that
same state into adapter-only and adapter-plus-PSF suffix arms. Both arms consume
the complete recorded suffix actions 180--236. Each no-contact arm therefore
executes 57 high-level actions, 285 safety-filter updates, and 1,425 two-
millisecond physics substeps; together with the shared prefix this is the full
237-action recorded VLA episode.

The protocol is explicitly an offline, single-case, hybrid controller-
feasibility test. It makes no online policy queries after the two suffix arms
diverge, so it is not a closed-loop VLA evaluation. The native BDDL goal is
false at the branch. Both arms record reward, `done`, native predicate values,
argument poses, and terminal simulator/observation hashes at every completed
suffix action. Reaching the goal early is latched but does not shorten the
registered exposure.

Only `SAFE_TASK_SUCCESS_USEFUL_CORRECTION` is positive. It requires the paired
adapter to reproduce the first link-5/moka-pot contact; the PSF arm to have no
selected-obstacle contact from any robot collision geom and a strictly positive
full-robot clearance certificate; material correction before baseline contact;
complete QP, field, bound, and exposure evidence; nontrivial post-correction
joint and Cartesian motion with nonzero commands; and native task success after
the correction and at the terminal action. Separate negative labels distinguish
shifted/remaining contact, useful collision prevention with task failure, and
stop-only behavior. Tracking and timing remain disclosed diagnostics rather
than feasibility gates.

Implementation lives in
`configs/vlsa_poisson_full_episode_feasibility.v1.json`,
`main/poisson_fullbody/full_episode_feasibility.py`, the reusable paired runner,
and an independent trace consumer. A literal PSF contact terminates that arm as
a valid scientific negative: continuing from `h <= 0` would require an unsafe
fallback and cannot support the requested feasibility claim. No-contact claims
must complete the entire suffix. Terminal H100 evidence is recorded below;
P01 remains `active` for broader claims beyond this one-case experiment.

The consumer independently binds the exact suffix actions at every controller
update, QP arithmetic, command-to-physics trace, all-robot contact records,
signed MuJoCo contact distance in `D_sim`, clearance certificate, task ledger,
and post-correction motion. The final local gate passes 672 tests with 136
expected allocation-only skips; the focused full-episode contracts pass 54
tests with one expected local NumPy allocation skip. One retained
limitation is explicit: static-obstacle envelope booleans and their aggregate
drift/speed diagnostics are code-bound producer evidence, rather than a second
reconstruction from serialized raw MuJoCo obstacle poses and velocities. The
clean source commit and runtime hashes bind that simple calculation, but the
experiment must not be described as independently remeasuring obstacle
staticity.

## P01 full-episode producer and consumer row-namespace repair

H100 producer job `34120` completed `COMPLETED|0:0` in 2h21m34s from clean
commit `0e51fc023a37bf218ed5e8bad740096fcbc7b21a`. It atomically published the
immutable 13,836,352-byte `result.json`; file SHA-256 is
`2601fa9087a19bfaea5e2ec3c3af86df41f5cffac07d1df3653c83feef755111`
and payload SHA-256 is
`f9f42e94e472f4051e185bf5fa373d8534c7103c56a65222b4904602749e3f2c`.
No producer output was interpreted while the job was nonterminal.

The first CPU-only consumer, job `34137`, terminated `FAILED|2:0` after five
seconds and is retained as apparatus evidence only. Its sole discrepancy was
`activation_first_material_row_differs`. The producer correctly serialized
`activation_evidence.first_material_correction` from `activation_trace`; the
consumer incorrectly compared it to the differently shaped `command_trace`
row for the same independently reconstructed update. This is a row-namespace
comparison bug, not a changed safety threshold or a physics outcome.

The repair preserves the independently reconstructed command row for QP
arithmetic and carries the separately matched activation row as
`activation_raw` for the serialized activation-evidence comparison. The CPU
launcher now binds the immutable producer commit separately from the clean
consumer checkout commit, permitting a later validator to audit the unchanged
producer artifact without rerunning or rewriting physics. No protocol,
threshold, producer, classifier, result, or acceptance condition changed.
Scientific interpretation remains pending a zero-exit clean consumer with zero
discrepancies. The corrected consumer passes 55 focused contracts with one
expected local dependency skip and the full 673-test repository gate with 136
expected allocation-only skips. A separate read-only fail-open audit concludes
that the key-matched namespace repair cannot create materiality or change the
classification and recommends rerunning only the CPU consumer.

## P01 accepted full-episode feasibility outcome

Corrected CPU-only consumer job `34139` ran from clean consumer commit
`1acd7ef233055f59344ef0418b066047c511c9fc`, remained bound to producer commit
`0e51fc023a37bf218ed5e8bad740096fcbc7b21a` and producer job `34120`, and
completed `COMPLETED|0:0` in seven seconds. Its terminal summary has SHA-256
`f8b3ca7feba5e116e2d24ed09612ccf143f918cb7e4fb02d9567fc5497f68b63`,
reports `artifact_valid=true`, zero discrepancies, and independently assigns
`SAFE_TASK_SUCCESS_USEFUL_CORRECTION`. The rejected consumer job `34137` and
its SHA-256
`fe3f7613167a7a97ada6e32da55a2c27aa20873af79db1444a249e0403825971`
remain preserved and do not authorize this outcome.

Both arms start from the exact paired boundary, execute all 57 suffix actions,
285 controller updates, and 1,425 physics substeps, and achieve the native BDDL
goal first at source action 182 and terminally after action 236. The adapter-only
baseline reproduces link-5 contact with the selected moka pot at physical
boundary 4677 and reaches `D_sim=-50.64` mm. The Poisson-CBF arm begins material
correction at boundary 4500, 177 physics substeps (354 ms) before that baseline
contact, uses 32 material correction updates, and has no selected-obstacle
contact from any robot geometry. Its conservative full-robot clearance lower
bound and `D_sim` are both +24.82 mm. All 285 QPs, independent residual checks,
and joint-limit checks pass.

The PSF does not obtain safety by stopping. From first correction through the
terminal boundary it executes all 285 updates and 1,425 substeps, with zero
zero-command updates, 0.3484 rad correction integral, 0.5625 rad/s maximum
correction norm, 1.4167 rad issued-command integral, 1.3262 rad measured joint-
motion integral, and 0.5948 m Cartesian path. Relative to adapter-only over the
suffix, it retains 88.87% of executed-command integral, 93.17% of measured
joint-motion integral, and 74.02% of Cartesian target progress. The accepted
answer to the registered feasibility question is therefore yes: useful
link-aware correction prevents the observed arm-link collision while preserving
full task success in this case.

The claim remains narrow. This is one static simulator-oracle case, with a
shared historical OSC prefix and open-loop recorded actions after the paired
branch. It is not a closed-loop VLA population result, learned-perception or
dynamic-obstacle evaluation, tracking-certified invariance result, or real-time
deployment claim. Full-window tracking error is diagnostic only (0.0958 rad/s
RMSE, 1.7405 rad/s maximum absolute error). The static-envelope independent-
reconstruction limitation from ADR-0041 is unchanged; P01 remains `active` for
those broader research gates.

## P01 closed-loop suffix canary preregistration

The accepted open-loop result answers the controller-feasibility question but
does not test whether pi0.5 can react to the trajectory changed by Poisson-CBF.
The next canary keeps the exact validated AEGIS/OSC prefix through action 179,
branches at physical boundary 4500, and runs two matched live suffix arms over
actions 180--236. Both arms query pi0.5 at global query indices 36--47 with the
registered seeds. After the states diverge, every query is constructed from
that arm's own immediately preceding native observation. Both then apply a
fresh released translational AEGIS QP and the same fresh joint-velocity
adapter; only the treatment adds the link-5/link-6 Poisson-CBF rows.

The first live query must be identical across the paired arms. Historical
action-180 obstacle geometry, end-effector proxy, and virtual-direction state
remain branch authorities; the current live action and AEGIS output must match
between arms. The adapter-only arm must reproduce the selected moka-pot
link-5/6 contact. The treatment must have no selected-obstacle contact from
any robot collision geom at any measured 2 ms substep, receive a material CBF-
attributed correction before the baseline contact, make at least one fresh
policy query after that correction, keep moving, and satisfy the native task.
Task success is latched as in the SafeLIBERO benchmark; terminal persistence is
reported separately because the fixed post-success exposure deliberately
continues to the known collision window.

This lean canary uses exact MuJoCo contact reconstruction at every 500 Hz
physics substep. The expensive 12,469-point whole-robot clearance sweep runs at
the 20 Hz action boundary only and is diagnostic, not a continuous-clearance
claim or acceptance gate. Each arm publishes a real-simulator branch-to-
terminal suffix video; the 30 fps playback is not wall-clock timing. A distinct
CPU-only consumer must independently validate the complete immutable result
before any producer label is interpreted. This paragraph records the contract
as preregistered before the first H100 submission; the terminal apparatus
outcome and replacement protocol are recorded below. All prior roots remain
immutable.

## P01 closed-loop q36 apparatus correction

Producer job `34175` ran from clean commit
`3eab3e57f1b11ebd0f2e16800e76174f4218e393` and terminated `FAILED|1:0`
before either paired arm entered physics. Its immutable failure artifact has
file SHA-256
`9ce3dba7f6c8ddd2b1f2f67f6b2a41c8c43c3688469101c2c90a78cfa807ab24`
and payload SHA-256
`1be5780195e1b9dd3fadd609377845f7171e91abd45c02d907007be651947a36`.
The sole failure was a bitwise mismatch between the current live pi0.5 query-36
chunk and the July historical chunk. No controller command or MuJoCo substep
ran, no scientific consumer was submitted, and the artifact provides no
Poisson outcome.

Bitwise equality to a policy response from a different server process is not
needed for the paired feasibility question. The corrected protocol executes
query 36 once from the exact shared branch observation and seed in the
adapter-only arm, then reuses that exact current chunk for the Poisson arm.
This makes the first five policy actions exactly paired without treating the
historical response bytes as controller authority. The historical chunk hash
is retained as a diagnostic. Historical branch state, obstacle geometry,
end-effector proxy, and AEGIS virtual-direction state remain mandatory; the
current action-180 AEGIS inputs and outputs must agree across arms. Queries
37--47 remain fresh, separate, seeded live inferences from each arm's own
observations. Baseline link-5 contact reproduction remains the empirical gate.
Baseline task success remains a reported diagnostic; the treatment alone must
achieve native task success after material correction, matching the stated
feasibility question and allowing an unsafe-baseline-failure/treatment-rescue
outcome.
The failed root is immutable; any rerun must use a new commit and run ID.

## P01 closed-loop suffix terminal result

H100 producer job `34185` completed `COMPLETED|0:0` in 12m17s on
`worker-mig-3g40gb-0` from clean commit
`a15aa5f8a5e0c7c783a2dc7450eb83a65eae4133`. Immutable run
`vlsa-poisson-link56-closed-loop-20260802b` has result file SHA-256
`9467811a975d92dce6e3a3d9fb10f32d6bc73e453c0302e24eb351db69df8d9d`
and payload SHA-256
`50a07b75f5f95f897a386333f513c9fb4a604b65f7ab33aff15e7a84feb3e3c7`.
The exact current q36 response was executed once and paired across arms; all
q37--47 queries used the respective arm's own observation.

CPU consumer `34187` exposed no result because its wrapper deleted a rejected
summary; this observability defect was fixed without changing physics.
Consumer `34189` then preserved a single rejection
(`baseline_periodic_clearance_invalid`). That check contradicted the frozen
protocol: the sampled conservative clearance is diagnostic, while exact
MuJoCo contact at every 2 ms substep is authoritative. Removing only that
accidental diagnostic gate produced consumer job `34191`, which completed
`COMPLETED|0:0` from commit
`bd6890f3ee4ecfa61f7be899c12df46709ca3a5e`. Receipt SHA-256
`d0e7e0d30d282f93d70c53ead31c48dd21df4ba800daf6c4355a34134e2a0ab1`
reports `artifact_valid=true`, zero discrepancies, a complete pair, and
`BASELINE_CONTACT_NOT_REPRODUCED`.

The live adapter-only arm completed all 57 suffix actions, 285 controller
updates, and 1,425 contact-observed physics substeps without any robot--moka-
pot contact. It achieved the task at action 182 but did not retain it at the
terminal fixed-exposure boundary. The Poisson arm also had zero selected-
obstacle contact, solved and independently postchecked all 285 QPs with zero
invalid field queries, achieved the task at action 182, and retained terminal
success. From the first correction at boundary 4500 it executed 0.4020 rad of
filter correction, 1.6420 rad of measured joint motion, and 0.6597 m of
Cartesian path with a zero-command fraction of 0.0. Thus it demonstrably did
not stop, but the absent paired baseline collision prevents attributing contact
avoidance to that correction.

This is a valid negative for the live collision-prevention canary, not a
Poisson failure. The prior recorded-suffix experiment remains positive
one-case controller-feasibility evidence; this live-feedback experiment adds
compatibility evidence (active correction, motion, and task success) but no
causal safety evidence. P01 remains `active`. The producer/consumer v1
`stop_only=true` secondary flag is not applicable when the baseline collision
is absent; future classification now requires baseline reproduction before
emitting stop-only, without changing this immutable run's label or feasibility.

Post-validation trajectory audit identifies the controller/feedback handoff as
the dominant observed reason. The shared action-179 state is exact and the five
actually executed q36 actions differ only slightly from historical AEGIS
(mean L2 0.001744, maximum 0.003120). Once those Cartesian commands pass
through a fresh joint-velocity adapter, however, the end effector is already
9.8 mm from the historical OSC trajectory after action 180 and the bowl is
76.7 mm away by action 182. Query 37 therefore sees a different state; all
q37--47 chunk hashes differ, their 52 executed action vectors have mean L2
2.125 from historical, and every gripper sign is opposite. At the historical
contact action 187 the live adapter baseline's end effector and bowl are 19.6
mm and 22.7 mm from their historical poses. This evidence does not isolate the
small q36 server difference from the controller handoff, but it strongly rules
out interpreting the missing live contact as Poisson prevention.

Handoff: no further H100 physics is authorized from this result. The next code
gate is to implement and preregister the control-only 109-case eligibility
screen described in ADR-0046, then run the exact local gate `./init.sh` before
a fresh live Slurm preflight. Until that protocol, immutable manifest order,
and stopping rule exist on a clean commit, there is intentionally no screening
submission command. The final local structural gate passes 729 tests with 137
expected dependency/allocation-only skips.

## P01 original-OSC e03 CAR-phase apparatus correction

The report-aligned original-OSC canary now targets
`vlsa-t1-spatial-i-t3-e03`: the complete historical AEGIS result supplies the
unsafe control, while the live arm differs only by a link-5/link-6 post-OSC
Poisson torque shield. Allocation-backed numeric job `34235` completed
`COMPLETED|0:0` on clean commit
`33bf9a4e4654d9c1c943a0eb7cca5a17b59aaab4`, passing 159 tests with zero
failures, errors, or skips. Its result file SHA-256 is
`049ed11e0509c22a6c4d891a43b7bc059b94147c51d5206011fa51767e65d221`.

Producer `34236` then stopped before action zero with `FAILED|1:0`. Its complete
apparatus-failure result has file SHA-256
`d0fed4fac7749bc2d85d20725da2fded1c93bf3bfdc19450320cd43a0e373003`
and payload SHA-256
`8a0e0769bfa42be424d742fb5ae03b3b94efa82bb3cb55c12711736931566477`.
The runner incorrectly compared the paper-CAR object observation with a
separately re-forwarded post-integration MuJoCo clone. Those values represent
adjacent simulator phases. No controller command or physics substep ran, so
this immutable attempt has no Poisson safety or task outcome.

The replacement contract leaves the treatment, threshold, contact authority,
and success gates unchanged. Paper CAR remains the released Table-1
observation-to-observation L1 displacement at completed 20 Hz endpoints. The
observable is structurally bound to the same root body ID used by contact
authority and exactly to its same-phase live `body_xpos`; forwarded
post-integration pose is serialized only as a phase-labelled diagnostic. A new
clean commit, zero-skip H100 numeric result, unused run root, terminal producer,
and distinct complete consumer are required before interpretation. P01 remains
`active`. The corrected apparatus passes the complete local gate (797 tests,
167 expected dependency/allocation-only skips), and three independent
adversarial audits report no remaining pre-submission blocker.
