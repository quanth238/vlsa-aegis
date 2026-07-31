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
