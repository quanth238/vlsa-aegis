# Factorized nominal-safety formula audit result

## Verdict

H100 job `38155` independently validates that the dominant safety error is
introduced by the learned joint rollout, not by MuJoCo forward kinematics,
ellipsoid attachment, the analytic support-gap function, or QP
linearization. All seven preregistered checks pass.

The implemented composition is exactly:

\[
(x,A)\xrightarrow{F_\theta}\widehat Q\in\mathbb R^{51\times7}
\xrightarrow{\texttt{sim.forward}}p_i(\widehat q_k)
\xrightarrow{\phi}\widehat h_{ik}.
\]

Both frozen structured model artifacts declare output shape `[51, 7]`. The
evaluator writes each predicted q into the seven robot joint positions, calls
MuJoCo `sim.forward()`, reconstructs the seven attached L5--L7 ellipsoids, and
calculates their analytic minimum support gaps to the frozen initial
exact-box obstacle union. It does not use the QP or linearize h.

## Decisive decomposition on 960 test actions

| Chain | Boundary RMSE | False-safe actions | Exact-safe recall |
|---|---:|---:|---:|
| Exact q -> FK -> phi versus cloned rollout | 0.321 mm | 0 | 99.51% |
| Predicted q -> same FK -> same phi versus exact-q geometry | 2.302 mm | 44 | 95.17% |

The exact-q chain has `2.36e-16 m` maximum error at substep zero. This is the
direct FK/attachment/phi implementation check because the static and dynamic
obstacle pose are identical there. Its later `0.321 mm` boundary discrepancy
is consistent with freezing the obstacle at k0 while the cloned rollout
records its small live motion.

The original factorized MLP's joint error is `11.560 mrad` over the trajectory
and `20.587 mrad` at substep 50. Passing those predicted joints through the
same exact geometry produces `44` false-safes. Predicted-q boundary error is
`7.17x` the exact-q geometry recomposition error.

The newer bounded-orientation structured models confirm the same mechanism:

| Model | Joint RMSE | Terminal joint RMSE | Boundary RMSE | False-safes |
|---|---:|---:|---:|---:|
| Flat | 19.549 mrad | 34.488 mrad | 4.484 mm | 64 |
| Time-conditioned | 17.307 mrad | 29.287 mrad | 5.214 mm | 63 |

## Interpretation

The MLP does predict joint configurations, but not accurately enough near the
safety boundary. Milliradian joint errors accumulate through the arm and
become millimetre-scale L5--L7 clearance errors. The nonlinear FK and
ellipsoid safety conversion are already accurate when supplied exact joints.

Therefore the current root problem is execution-trajectory prediction,
especially late-horizon signed joint error. Replacing or retuning a QP
linearization cannot repair incorrect \(\widehat q_k\), and closed-loop control
remains unauthorized. The next model test should improve joint-trajectory
prediction and then reuse this unchanged explicit geometry backend.

## Reproducibility

- Slurm job: `38155`, H100 `worker-2`, `00:00:07`.
- Source commit: `ee35d1244b967fca7ff42a309b4ddf906b3d45f6`.
- Result file SHA-256: `1d56b9f1a4e95ab80d02f9cf5cb4d6c0c22edd4c61f22abfbbf1f41b80e1ac55`.
- Result payload SHA-256: `1aeaaa47484747bd97c99f935064b2a72d179ca40f11316c1e339f701b1cc569`.
- Validation file SHA-256: `8413606405dd9870b9351012b4e603885891ac9c9456063816bb5d88a14b0658`.
- Validation payload SHA-256: `833ce74a5396ef91c4acc74a411f77782fa15c994e0c3c7dab4c07c25fd0f8d5`.
