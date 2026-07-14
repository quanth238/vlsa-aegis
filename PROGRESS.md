# Progress

Last updated: 2026-07-14 (Asia/Ho_Chi_Minh)

Active branch: `agent/crfs-oracle-harness`

Baseline: THU-RCSCT/VLSA-Aegis commit `57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b`

## Verified

- The baseline repository, including SafeLIBERO and its OpenPI fork, is the Git worktree foundation.
- The ordinary policy and environment entry points remain available; CRFS controls are opt-in.
- Local dependency-free unit suite passes, including geometry, paired artifacts, restart semantics, and synthetic end-to-end flow.
- Live VinUni access, Slurm/QOS, storage, and existing OpenPI environments were inspected without login-node compute.
- Commit `2bbd53a` is pushed to `quanth238/vlsa-aegis:agent/crfs-oracle-harness`.
- Slurm job `27077` reached the MIG worker and failed cleanly before conversion because the existing OpenPI environment lacked its patched Transformers modules. The final checkpoint path was not created; the diagnostic `.incomplete-27077` path is retained.
- Slurm job `27079` completed conversion in 1m57s. `model.safetensors` is 6.8 GB with SHA-256 `988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed`; normalization assets are present.
- Slurm job `27082` loaded the converted policy and opened the WebSocket server, then failed before simulation because SafeLIBERO attempted an interactive first-import config prompt. No research result was written; server cleanup succeeded.
- Slurm job `27083` passed noninteractive SafeLIBERO import and began the paired run. It exposed `hard_reset=True` recompiling the MuJoCo scene for every branch; after exact-job inspection and confirming no result existed, job `27083` alone was canceled. The runner now uses soft reset plus fixed-state restoration and emits structured progress events.
- Slurm job `27091` isolated a false raw `mj_geomDistance` signal to the `gripper0_hand_collision` mesh versus an obstacle box. The raw query was negative while the collision-enabled pair had no contact, so projection was stopped before optimization.
- Slurm job `27096` calibrated the controlled sphere--box primitive on 261 states: zero boundary error over 23 samples within $\pm1$ cm, zero sign disagreements, and first contact at -0.603 mm for a 1 mm sweep step.
- Slurm jobs `27116` and `27117` completed the strict H03 manifest: 50 unique saved states, five identical replays per state, 126 measurements per replay, zero clearance variation, zero endpoint-coordinate variation, and no conservative-proxy contact miss.
- H03 population composition was 20 proxy-colliding, 30 proxy-safe, and 4 physical-contact cases. Sixteen cases reproduced the raw mesh-query inconsistency, which is now advisory-only evidence rather than the primary safety signal.
- The H03 summary and minimum-pair visualization are committed under `evidence/h03/`; raw allocation artifacts remain under `/mnt/data/quanth/experiments/crfs-oracle/h03-population-20260714a`.
- Slurm job `27123` passed H04 on 10 calibration states and 20 held-out prefixes: median endpoint error 0.971 mm, 95th-percentile absolute `D_opt`/`D_sim` error 1.082 mm, and zero false-safe predictions at a 10 mm margin.
- H05 added an optimizer-independent endpoint certificate: zero-sum translation fixes the H04 endpoint, so an unsafe endpoint proves the swept-path constraint infeasible before optimization.
- The H=5 population run completed all 20 H03-colliding states: 20/20 certified infeasible, `F_proj=0.00`, with endpoint-clearance upper bounds from -48.221 mm to -7.723 mm.
- The one registered H=10 refinement also completed all 20 states: 20/20 certified infeasible, `F_proj=0.00`, with endpoint-clearance upper bounds from -75.234 mm to -44.895 mm.
- The preregistered H05 stop rule fired. H06--H09 were not run, and no learned probe was trained.
- R00 apparatus run `r00-calibration-20260714a` completed all 120 cases on Slurm array `27276` with no case failures. Its allocation-backed summary job `27285` correctly refused promotion because the scientific config hash included per-array WebSocket ports. This run also measured scene motion only at the endpoint, so it remains apparatus evidence rather than calibration evidence.
- R00 now excludes host, port, output root, and run ID from the scientific config identity; it records immutable manifest/case hashes, verifies exact reset branches, and measures target/obstacle maximum displacement over all 125 physics substeps. A fresh allocation-backed R00 run is required.
- Strict R00 smoke job `27290` exercised the new measurements and then failed closed before finalization because the legacy MIG smoke was not an array and therefore had no `SLURM_ARRAY_TASK_ID`. No scientific artifact was written; the smoke template is now a one-element `%1` array like the full-run provenance contract.
- Strict one-element smoke array `27291_0` passed at commit `0adbd67`, including exact replay, 126 safety measurements, 125 body-tracking substeps, and transport-neutral scientific config hash `e89abe6ddf1bef370809a029706b1b530ddf1f6477f85641236fcfe0adcf19b2`.
- Strict R00 array `27292` completed all 120 cases with zero batch failures. Allocation-backed verifier `27298` passed: all 120 chunks were eligible positive examples across 30 groups disjoint from the frozen 20 evaluation groups, and the registered inverted-CDF lower quartile froze `p_min = 0.029897349105658888 m`.
- The passed R00 artifact is committed as `evidence/r00/r00-summary.json`, SHA-256 `90fe09e075b30e680e5cfbd81c802afa03c94b58ba7f0e95e66080c61af2096f`. R01 is now content-bound to that threshold and artifact.
- R01 smoke job `27303` produced the first repeated simulator-verified endpoint-free safe-progress witness. Its translation correction was already far outside the H04 local calibration domain, so the smoke was treated as apparatus evidence only.
- Full R01 array `27306` completed all 20 frozen cases with zero task failures. All 20 paired nominal replays reproduced the registered collision; 17 cases had changed-action witnesses with repeated `D_sim >= 5 mm`, reach progress at least `p_min`, no forbidden contact, and sub-millimetre scene motion.
- The original queued verifier `27314` failed before analysis because Python 3.8 evaluated a PEP 585 type alias. No summary was written. The compatibility fix has a regression test and did not change any case artifact or scientific criterion.
- Allocation-backed verifier `27364` independently recomputed the gate from raw actions and repeated rollouts and passed R01 at 17/20. The committed summary is `evidence/r01/r01-summary.json`, SHA-256 `715d9326f02df531e0c6d7aa7b94629569b320800b4283c36fff41335d9b8bc5`.
- All three R01 negatives began below the immutable 5 mm branch margin; one began in penetration. Under the registered branch-plus-125-substep predicate, no same-instant action can make those initial samples pass. All 17 branch-margin-valid cases produced witnesses.
- Physical existence did not establish a local flow edit: every selected witness was outside the H04 `+/-0.15` calibration domain and saturated at least one translation component. Median correction L2 was 3.243 and median RMS over the 15 translation coordinates was 0.837.
- Allocation-backed sampler-parity job `27389` passed the frozen R02 numerical gates on the public-JAX default path versus the PyTorch eager trace-only path. The validated artifact is `/mnt/data/quanth/experiments/crfs-oracle/sampler-parity-r02-20260714c/sampler-parity.json`, SHA-256 `26083b0a71ec41cf67e0978b815a77bfe71e4b4a4a08a8ddcdaece608de9818a`; first-five unnormalized LIBERO XYZ maximum error was `0.001148` action units and full 10x7 maximum error was `0.005306` action units.
- The frozen R02 apparatus now contains six distinct arms: frozen, direct R01 witness, equal-L2 seeded random residual, equal-L2 analytic-geometry residual, distributed oracle residual, and one-shot bridge diagnostic. Raw R01 cases, sampler parity, directions, simulator transforms, substep measurements, policy traces, bounds, and terminal failures are independently revalidated before aggregation.
- ADR-0012 separates the R03 oracle-steerability claim from permission to train a learned probe. A learned probe is authorized only if R03 passes, the matched analytic arm does not, and oracle exceeds analytic under both a grouped bootstrap and an exact one-sided paired discordance test.
- The first R02 smoke, Slurm job `27393`, failed closed before any simulator action. The fresh branch snapshot used tuple coordinates while the JSON-loaded R01 snapshot used lists, and the runner also imposed invalid byte equality between MIG and full-H100 eager outputs. The exact launch-failure artifact and logs remain under `r02-oracle-flow-smoke-20260714a`; no `r02-paired.json` was produced.
- Independent comparison showed that R01 MIG smoke `27303` and the MIG parity eager path agree within `5.9e-8`, while the same H100 R01 case differs by at most `0.003532818` over first-five XYZ. ADR-0013 therefore canonicalizes the branch exactly, bounds historical drift with the already-frozen ADR-0010 tolerances, and defines the applied oracle direction as the immutable witness minus the fresh paired eager baseline. The stale R01 delta remains a hashed norm/angle diagnostic only.
- The revised combined local R02 gate passes: 155 tests passed and 35 allocation-runtime tests were skipped by `./init.sh`; the NumPy-backed focused R02/parity suite passed 61/61. These are apparatus checks only, not research evidence.
- Replacement smoke job `27403` completed all six paired arms for case `crfs-1069f29a8d76463a`. The frozen arm reconfirmed collision (`D_sim=-4.356 mm`), the immutable direct witness reconfirmed (`31.806 mm` clearance, `33.372 mm` progress), and the distributed oracle residual passed (`28.997 mm`, `32.984 mm`) while equal-L2 random and analytic controls failed progress and the one-shot bridge failed clearance. This one case is smoke evidence only, not a population claim.
- The job-27403 artifact validated on its allocation host, but an independent CPU-runtime validator reconstructed the random/analytic directions with maximum `2.22e-16` float64 differences and the near-parallel angle with `7.42e-12` degree difference. ADR-0014 confines numerical tolerance to independently derived float64 reconstructions (`1e-12`, or `1e-10` degrees after `acos`); stored arrays, applied corrections, discrete geometry choices, pairing, actions, outcomes, and scientific gates remain content-bound or exact. A new smoke from the updated clean commit is required before population launch.
- Commit `83bd671b892d7d8829a34dc986c0b86d9e4c0c16` froze the ADR-0014 portability fix. Clean-commit smoke `27404` completed all six arms, validated independently with no errors, and reproduced the earlier smoke's actions, traces, directions, and scientific outcomes. Its artifact SHA-256 is `220139382fadc6dd90eb08b6f057e6d1ab8fde3a6dfa50e9df0659c63fdc51ce` and records a clean worktree.
- Full H100 array `27405` completed all 20 immutable cases with exit code 0 and produced exactly 20 final `r02-paired.json` artifacts, with no launch failures. All frozen collisions and all 17 applicable direct witnesses were reconfirmed; there were no missing, duplicate, invalid, direction, bounds, drift, pairing, or source-hash failures.
- Allocation-backed CPU verifier `27450` passed R02/R03. The committed artifact is `evidence/r03/r03-summary.json`, SHA-256 `dea3e66c854faa0b659777bfed4b651b19d8d4d0710696ff7bfe52c44d4ee76e`, with ordered result-set digest `fa79a132f2fecbedab5917111ef71f022e52c933d8e535aaca118add5f5b7895`.
- On the 17 R01-feasible groups, direct witnesses passed 17/17, distributed oracle passed 9/17, and frozen, equal-L2 random, equal-L2 analytic, and one-shot bridge passed 0/17. Oracle minus random was `0.5294118`; its 10,000-replicate grouped 95% bootstrap LCB was `0.2941176`, so every preregistered R03 criterion passed. On the original 20-case population, direct was 17/20 and oracle was 9/20.
- The analytic matched gate failed with analytic minus random equal to zero and LCB zero. Oracle minus analytic had LCB `0.2941176` and exact one-sided paired `b=9`, `c=0`, `p=0.001953125`. ADR-0012 therefore authorizes R04; this is permission to test a learned probe, not evidence that ECG works or is novel.
- The eight eligible oracle misses split exactly into four clearance-only and four progress-only failures; none failed both and every direct witness passed. All 17 registered equal-L2 one-midpoint analytic arms passed clearance but failed progress, while all 17 registered `t=0.5` one-shot bridges failed clearance. The failures therefore occur after the physical-existence gate, under the registered flow-intervention mapping; they are consistent with gain, timing, parameterization, nonlinear continuation, and a safety-progress tradeoff, but R03 does not isolate those mechanisms.
- Independent Python 3.12 validation exposed one additional one-ULP reach-distance mismatch caused by Python's changed float `sum` implementation. ADR-0015 freezes explicit left-to-right three-coordinate accumulation, preserves the full exact annotation, and exactly validates the original H100 case without changing any artifact, action, threshold, or outcome.
- The final local gate passes after the R03 transition and ADR-0015 fix: `./init.sh` reports 156 passed and 35 allocation-runtime skips; the dependency-backed focused R02 suite passes 84/84; the copied H100 case reconstructs with zero validation errors. These are regression checks, not new research evidence.
- Re-auditing `main.tex`, `Chat - Probe in Diffusion Model.md`, the R00--R03 artifacts, and the revised ECG manuscript confirms that ECG is a plausible next experiment but is not yet evidence of a solution. The exact-endpoint premise was the primary failure; the remaining R02 misses occur after direct safe-progress witnesses exist and are consistent with intervention gain/timing/parameterization, nonlinear continuation, and a safety-progress tradeoff.
- The revised manuscript now narrows the evidence to a five-action pregrasp reach pilot with a controlled 6 cm EEF sphere against active-obstacle OBBs, distinguishes `D_opt` from `D_sim`, uses continuation-matched labels and groupwise false-safe control, retains progress in the causal outcome, and marks R04 as prospective. Its compiled PDF is nine pages and visually clean; the sparse final references page remains a presentation/page-limit risk.
- ADR-0017 freezes R04A as a no-learning, one-case real-continuation label-contract smoke. The frozen config SHA-256 is `561128a5a05710e50b282582463127ee3f8cd87c9ebbaee822f8c4602fda1c25`. The reused R00 row is `apparatus_only` and may never enter training, calibration, validation, testing, or a claim.
- The opt-in R04A apparatus requests duplicate eager no-intervention traces at steps 1--5, enforces `predicted_clean = x_t - t*v_base`, exact physical 10x7 actions across all ten calls, two exact five-action simulator replays, raw 126-sample `D_sim`/tracked-reach reconstruction, world-frame padded 21-OBB geometry, complete source/allocation/hash provenance, explicit optimizer/`D_opt` not-applicable status, atomic finalization, and clean-reviewed-commit resume semantics. It does not edit/resume a latent, train a probe, or apply guidance.
- An independent patch audit found and closed endpoint-only-label acceptance, approximate-clean semantic, stale-resume, and source-episode split-leakage defects. The dependency-backed focused R04 suite passes 20/20; the complete `./init.sh` gate passes 176 tests with 43 allocation-runtime skips; shell/JSON/whitespace checks and Python 3.8 grammar compatibility pass; malformed artifacts fail validation without throwing.
- The exact remote worktree was cleanly fast-forwarded to reviewed commit `1138be311e255d891e8a3dfaace1a4e631da7479`. A fresh preflight found no user jobs, excluded drained/not-responding `worker-mig-3g40gb-1`, and confirmed 90,771 MiB fresh host memory on healthy `worker-mig-3g40gb-0` before submission.
- R04A Slurm array task `27514_0` completed in 1m28s with exit `0:0`. Its final artifact is `/mnt/data/quanth/experiments/crfs-oracle/r04a-label-contract-smoke-20260714a/crfs-93365b8b851365f2/r04-label-contract.json`, SHA-256 `b820793ec42a5228c858e297d13806d8ae7f02e5cc7a765769316473d795f285`.
- All five registered trace steps had exact duplicate requests, all ten complete sampler calls returned the exact same physical action, both five-action simulator replays were exact, and the fixed geometry encoded 21/21 OBBs without truncation. The retained descriptive result had `D_sim=11.531 mm`, no contact, and `26.739 mm` reach progress, below `p_min`; retention of that low-progress row confirms that the apparatus did not filter an unfavorable outcome.
- Independent CPU Slurm validator `27516` reconstructed the artifact on `worker-2` with zero errors and the same SHA-256. Descriptive inspection job `27518` changed no artifact. The compact record is `evidence/r04a/r04a-validation.json`. This passes the label-plumbing apparatus only; feature R04 remains active and no probe is trained.
- The post-smoke design audit found that the current bridge cannot label an edited feature, plain BDDL resets are not released Level-II states, the released 50-state task-0 file is exhausted, the proposed `[-0.15,0.15]^15` perturbations do not support doses `0.25`--`4.0`, and current boundary/false-safe/causal group counts are not powered. ADR-0018 freezes the stop before perturbation labels or training.
- ADR-0019 freezes R04B as the final apparatus-only subgate. Its config SHA-256 is `a0edb7edac86d2abd887218002d14e28e072cd472d631797d18d540f5140be33`: one compiled call before and after duplicate source/zero-resume/nonzero-resume calls at steps 1--5, for 32 calls total on one reused apparatus state.
- Independent review caught and closed three pre-submission defects. R04B now records its one reset plus 20 dummy settle-control steps instead of claiming no simulator execution; zero resumes require byte-exact latent, velocity, predicted-clean in both coordinate spaces, normalized final, and physical final parity; and the current eager path is bound exactly to the validated R04A observation/noise/trace/action hashes while the ordinary compiled path is rerun under the unchanged R02 limits.
- The final local R04B contract passes 13/13 Python 3.12 tests with Draft 2020-12 schema validation. The complete `./init.sh` gate passes 204 tests with 58 allocation-runtime skips. The allocation wrapper also runs the dependency-backed PyTorch/OpenPI control tests before loading the policy.
- R04B source task `27558_0` completed on `worker-mig-3g40gb-0` at clean commit `15b97b63f77b1b8b60fa53b5efec334ff2d32a3e`. All 6 dependency-backed seam tests passed, the exact-job validator returned zero errors, and the raw artifact SHA-256 is `977084df18cdccf1a8cdca42a1af1faaa4ef039ce2d83df716f14b8b43e3dae1`.
- Independent CPU Slurm validator `27559` completed on `worker-2`, was bound to source identifiers `27558/27558/0`, reconstructed the same artifact SHA-256, and returned zero errors. `evidence/r04b/r04b-validation.json` is the compact record. This passes exact-resume apparatus only; no perturbation label, probe training, sampled-policy efficacy action, or guidance outcome ran.
- ADR-0020 now stops apparatus expansion. The direct research question is whether one locked learned clearance-gradient arm improves paired Safe-Progress Success over frozen, realized-norm random, strong time-structured margin analytic, and AEGIS/system baselines while retaining progress and reducing online cost. R04 remains active.

## Active gate

R04 — learned ECG probe. R02/R03 authorize a bounded investigation, and R04A
and R04B have now passed their allocation-backed label-plumbing and exact-resume
apparatus contracts. ADR-0020 ends sampler apparatus work. A defensible new
state estimand, support-matched perturbations, and powered group manifests
remain prerequisites to the decisive probe experiment.
ECG effectiveness, necessity, novelty, and transport-phase validity remain
open.

## Next

1. Seek the benchmark authors' original Level-II state generator in parallel.
   Since no generator exists in the repository, implement the additive
   `task0_single_obstacle_generated_v1` estimand with finalized XML, full
   pre-settle state/history, hash-bound observation/geometry, structured
   rejection records, and fresh-load replay checks. Do not label it Level II.
2. Run a retired allocation pilot to estimate valid-state, nominal-collision,
   boundary, unsafe, trigger-positive, and direct-witness prevalence. Freeze a
   joint-powered production count, then repeat R00--R03 on disjoint generated
   groups. A failed privileged-flow transfer stops probe training.
3. Replace one-sided local perturbations with a preregistered antithetic,
   multiscale design. Measure actual post-edit feature support and forbid causal
   doses outside it.
4. Freeze the boundary band, equal-group evaluation unit, conformal count,
   unsafe/boundary/trigger-positive counts, paired discordance assumptions, and
   joint power simulation before any label or final outcome is opened.
5. Run the direct research ladder: held-out clearance/false-safe prediction,
   one-step learned-gradient versus equal-norm random causality, and then full
   margin-stopped guidance. Pair frozen, learned-gradient,
   equal-realized-norm random, registered analytic, margin-stopped/time-structured
   analytic, AEGIS, the closest reproducible constrained-flow optimizer,
   privileged oracle-direction, and direct-planner arms by state,
   observation, policy noise, and executed horizon. Keep safe-progress success
   as the joint outcome, report paired effect sizes/confidence intervals and
   inference latency, and reject clearance-only gains. Do not submit training
   before the preceding prerequisites pass.

Exact next local starting command:

```bash
rg -n "get_task_init_states|_reset_internal|set_init_state|model.xml" safelibero/libero/libero main/crfs_oracle
```

No learned-training submission is authorized yet.

## Open scientific risks

- Raw MuJoCo mesh--box `mj_geomDistance` is unreliable in this stack. The primary controlled metric is the proposal's EEF-sphere/known-box signed distance evaluated from simulator transforms; physical contacts remain a one-way conservatism check.
- Released SafeLIBERO obstacles are movable; the proposal preregisters a static asymmetric-convex controlled pilot.
- The endpoint certificate is relative to the frozen H04 linear response and static branch geometry; it rejects this registered local-equivalence formulation, not all possible collision-avoidance planners.
- Sampler parity passed for the frozen public-JAX/PyTorch comparison, but compiled and eager PyTorch paths are numerically rather than byte-for-byte identical; R02 therefore uses the validated eager trace-only path for every paired arm.
- Eager BF16 actions are hardware/kernel conditioned at small numerical scale. Historical R01-to-R02 drift is bounded by the pre-outcome ADR-0010 limits and reported; all causal arm comparisons remain exact within one allocation.
- R03 is a small, single-task development population. The oracle passed at the minimum integer count above the registered 0.50 threshold (9/17), and its eight failures remain unresolved.
- A bounded endpoint-free search miss is not an infeasibility certificate. Only a simulator-verified witness proves existence.
- The R01 witness corrections are large and action-saturating. Physical action existence may therefore lie outside the reachable or task-preserving support of a local flow intervention.
- The three branch-margin failures show that the registered intervention instant is already too late for some cases. An earlier-state or recovery-barrier study must be a separately frozen experiment, not a post-hoc R01 relabeling.
- The current controlled metric covers a 6 cm EEF sphere against the active obstacle's oriented boxes, not full-arm mesh safety.
- The controlled result remains limited to an EEF sphere against the active obstacle's OBB union; narrowing the manuscript does not provide full-arm safety evidence.
- The oracle direction is privileged planner-witness information, not a learned clearance gradient. Oracle steerability does not guarantee that a scalar continuation-clearance probe can recover a task-preserving direction.
- The exact deterministic post-edit continuation seam and fail-closed validator now pass locally and in source task `27558_0` plus independent validator `27559`; this closes plumbing, not efficacy.
- Boundary MAE and rank correlation do not control false-safe predictions at the 5 mm decision threshold. R04 needs a frozen one-sided boundary calibration or false-safe criterion in addition to scalar regression metrics.
- Excluding low-progress actions from probe training does not guarantee that a clearance gradient remains on the progress-preserving action manifold. The gradient-causality gate must therefore retain safe-progress success and cannot be replaced by clearance improvement.
- Analytic 0/17 rejects only the registered static, equal-L2, one-midpoint geometry normal. It does not establish that learning is necessary or rule out stronger margin-aware or time-structured analytic guidance.
- Every current result is a five-step pregrasp reach study. Post-grasp transport still requires a separate phase-specific oracle ladder and cannot inherit R04 evidence.
- Approximate-clean gradient guidance is not itself novel relative to OmniGuide, QGF, Guided Action Flow, and constrained-flow safety guidance. R04 must demonstrate held-out prediction, gradient causality, and a paired advantage with clearly stated sensing/runtime assumptions before supporting an ECG claim.
- The passed R00 source has only 30 state groups, a minimum clearance of 10.327 mm, and zero rows in the inclusive 0--10 mm band around the 5 mm decision margin. R00 can exercise R04A plumbing but cannot support probe fitting, calibration, boundary evaluation, or a claim.
- All 50 currently saved same-task initial episodes are already partitioned across prior development/evaluation work. Repeated policy seeds or replay-derived intermediate branches are not new independent states; claim-bearing R04 needs a newly frozen source-state population or an explicitly expanded scope.
- R04A follows an unedited deterministic trajectory and does not expose exact latent resume. The existing one-shot bridge trace is pre-edit, so it cannot label a perturbed feature; a post-edit continuation seam remains a prerequisite for the perturbation study.
- Plain BDDL resets instantiate all declared workspace obstacles and are not equivalent to the released one-active-obstacle Level-II states. The repository contains no author generator for the pruning/parking transformation; a custom generator changes the estimand and requires renewed prerequisite evidence.
- Sixteen one-sided perturbations from `[-0.15,0.15]^15` have root-mean-square population scale `sqrt(E[RMS^2]) = 0.0866` and maximum RMS `0.15`; they cannot support the proposed `0.25`--`4.0` causal doses or the `0.77`--`1.15` witness range. Support must be measured in the actual post-edit feature space.
- With 30 unsafe groups, the registered exact false-safe UCB passes only at zero errors. Independent boundary, conformal, trigger-positive, and paired causal group counts remain unpowered until a prospective joint design calculation is frozen.
