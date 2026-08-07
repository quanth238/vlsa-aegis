# AEGIS SafeLIBERO table reproduction

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
