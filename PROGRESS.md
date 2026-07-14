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

## Active gate

R02 — with allocation-backed public-JAX/PyTorch numerical parity established, test whether the frozen flow can realize the known endpoint-free witness direction and outperform an equal-norm random direction. The direct witness, analytic geometry direction, distributed residual, and one-shot bridge remain distinct arms.

## Next

1. Finish the ADR-0014 portability regression gate, commit and push the clean R02 apparatus, synchronize the exact commit to `/home/quanth/working_space/vlsa-aegis-crfs`, and verify the remote worktree is clean.
2. Run the exact strict replacement smoke command below. This uses a new immutable smoke-only run ID because `r02-oracle-flow-smoke-20260714b` is already complete and must not be reused:

   ```bash
   RUN_ID=r02-oracle-flow-smoke-20260714c scripts/hpc/submit_r02_smoke.sh \
     manifests/oracle_h05_colliding.jsonl \
     configs/experiments/r02_oracle_flow.json \
     /mnt/data/quanth/experiments/crfs-oracle/r01-endpoint-free-population-20260714a \
     /home/quanth/working_space/vlsa-aegis-crfs/evidence/r01/r01-summary.json \
     715d9326f02df531e0c6d7aa7b94629569b320800b4283c36fff41335d9b8bc5 \
     /mnt/data/quanth/experiments/crfs-oracle/sampler-parity-r02-20260714c/sampler-parity.json \
     26083b0a71ec41cf67e0978b815a77bfe71e4b4a4a08a8ddcdaece608de9818a
   ```

3. Independently validate the smoke artifact. If it is valid, run all 20 cases on H100 under a new immutable population ID with concurrency at most two, then run the CPU allocation-backed R03 summary.
4. Report the feasible-conditioned 17 and original intent-to-treat 20 populations separately. Train no learned probe unless both R03 and the stricter ADR-0012 authorization pass.
5. Collect new immutable post-grasp states and a new progress calibration before making any transport-phase claim.

## Open scientific risks

- Raw MuJoCo mesh--box `mj_geomDistance` is unreliable in this stack. The primary controlled metric is the proposal's EEF-sphere/known-box signed distance evaluated from simulator transforms; physical contacts remain a one-way conservatism check.
- Released SafeLIBERO obstacles are movable; the proposal preregisters a static asymmetric-convex controlled pilot.
- The endpoint certificate is relative to the frozen H04 linear response and static branch geometry; it rejects this registered local-equivalence formulation, not all possible collision-avoidance planners.
- Sampler parity passed for the frozen public-JAX/PyTorch comparison, but compiled and eager PyTorch paths are numerically rather than byte-for-byte identical; R02 therefore uses the validated eager trace-only path for every paired arm.
- Eager BF16 actions are hardware/kernel conditioned at small numerical scale. Historical R01-to-R02 drift is bounded by the pre-outcome ADR-0010 limits and reported; all causal arm comparisons remain exact within one allocation.
- Only one allocation-backed R02 smoke intervention outcome exists. It is useful apparatus evidence but cannot establish an oracle/random or oracle/analytic population effect.
- A bounded endpoint-free search miss is not an infeasibility certificate. Only a simulator-verified witness proves existence.
- The R01 witness corrections are large and action-saturating. Physical action existence may therefore lie outside the reachable or task-preserving support of a local flow intervention.
- The three branch-margin failures show that the registered intervention instant is already too late for some cases. An earlier-state or recovery-barrier study must be a separately frozen experiment, not a post-hoc R01 relabeling.
- The current controlled metric covers a 6 cm EEF sphere against the active obstacle's oriented boxes, not full-arm mesh safety.
- Approximate-clean gradient guidance is not itself novel relative to OmniGuide, QGF, Guided Action Flow, and constrained-flow safety guidance. A learned probe is justified only by a measured advantage over the analytic geometry arm or a material runtime/representation benefit.
