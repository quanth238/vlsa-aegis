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

The original engineering ladder also recommends a one-step counterfactual,
a manual static-box active trial, and a coupled protected-sample
field/Jacobian finite-difference test. Their absence does not permit a false
positive because the first real canary is development-only and all tracking,
completion, baseline-hazard, and no-motion paths fail closed. They remain
required before scaling beyond the first exploratory link-5/6 canary or making
a broader full-body claim.

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
