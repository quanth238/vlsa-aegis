# AEGIS SafeLIBERO table reproduction

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
