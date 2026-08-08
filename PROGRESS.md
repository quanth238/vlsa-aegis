# AEGIS SafeLIBERO table reproduction

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
