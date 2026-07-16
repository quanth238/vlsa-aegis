# 0049 — Accept the numeric diagnostic and retire autograd linearization

Status: accepted from terminal jobs `28222_0` and `28223` on 2026-07-16.
The exact CFS-00A execution identity is consumed, the checked-in apparatus is
closed, and no further H100 run is authorized by this decision.

## Evidence

Immutable run `r05a-constrained-flow-fd-diagnostic-20260716b` used exact
release `1e2d36de9cb009a07e75d3535ab6f7070c5ea34c`, source contract
`956b4f8b109e4a7f2498ad0fe6871db1249078152bed03aecc43d98494658b78`,
and submission receipt
`7f36201e847672c7c67887ff6176cbc899c934b1b594c1420b9d5e985b55e16c`.
The source contract bound 66 repository paths to the clean release tree.

GPU task `28222_0` completed `0:0` on worker-1 in `00:02:56`. CPU
`afterany` publisher `28223` completed `0:0` on worker-0 in `00:00:06`. The
CPU job independently validated and atomically published result
`955d7b88f1b9c55dff77dfeb2ee19a3e055670f59e1ddab5acb41d67b13659aa`
and receipt
`29e926ad9edba1e94d448d2938ca7965b73117647e0b78900d77d04bb8011200`.
The raw terminal payload is
`37edb7bc6f0961563cbcbfbf938e156211efc9397f6e7431ec97bde3ee846f41`.
The raw payload was semantically validated, the result was schema-valid, and
the publication receipt passed with no errors. The payload and result classify
the run as `apparatus_inconclusive`, exactly as preregistered for a failed
numerical gate.

The transport repair from ADR-0048 worked. The paired client accepted the
unchanged WebSocket server's standard timing metadata, extracted the typed
terminal, and preserved the complete float32 `35 x 75` Jacobian plus all six
registered central-difference comparisons. The CPU validator rebuilt every
Jacobian-vector product, central derivative, error, and pass Boolean from the
raw arrays rather than trusting the GPU's summary.

The frozen budget was `3.6398398876190186`. The fixed perturbations were
`0.0028436249122023582` and `0.0014218124561011791`. Every comparison failed
the unchanged relative `0.10` or absolute `0.001` rule:

| Direction | `||Jd||` | central norms at the two epsilons | relative errors | cross-epsilon central discrepancy |
|---|---:|---:|---:|---:|
| all ones | 0.651180 | 0.838582 / 0.858240 | 0.716165 / 0.625675 | 0.818860 |
| alternating | 0.388031 | 0.656742 / 0.644389 | 0.675810 / 0.867750 | 0.937045 |
| registered modular | 0.384851 | 0.596768 / 0.726316 | 0.702921 / 0.993221 | 1.186637 |

Here the cross-epsilon discrepancy is
`||g(epsilon_1)-g(epsilon_2)|| / max(||g(epsilon_1)||, ||g(epsilon_2)||)`.
Its corresponding cosines were `0.657144`, `0.552738`, and `0.162468`.
The smaller perturbation did not consistently move the forward response
toward the autograd product: it improved the all-ones direction but worsened
the other two. This is direct evidence that the recorded autograd Jacobian
is not a faithful local model of the actual frozen sampler response at the two
registered operating scales on this canary.

The exact model path is mixed precision. The bound pi0.5 configuration uses
`bfloat16`, and the bound Gemma implementation explicitly converts the action
expert hidden input to `bfloat16` when its first projection is `bfloat16`.
Autograd propagates a formal derivative through this cast, whereas actual
finite forward evaluations include the cast's rounding. This is a credible
mechanism for the observed mismatch, but this run did not repeat identical
plus/minus calls or run a separate precision control. It therefore does not
isolate quantization from all other forward numerical effects and must not be
reported as a proved sole cause.

All 99 allocation tests passed with zero skips. Sampled host high-water was
16,072,122,368 bytes under the unchanged 64-GiB hard limit; sampled device
high-water was 8,513 MiB; and `max`, `oom`, and `oom_kill` event deltas were
zero. No FISTA update, Arm-B candidate, nonlinear replay, Arm-C refinement,
four-gate evaluation, or generated simulator action ran.

Compact evidence is
`evidence/r05a/cfs00a-fd-diagnostic-20260716b.json`.

## Decision

1. Accept this run as a valid terminal numerical diagnostic and permanently
   consume run `r05a-constrained-flow-fd-diagnostic-20260716b` and jobs
   `28222_0`/`28223`. Never resume, reuse, or reinterpret that identity.
2. Preserve its formal class as **apparatus-inconclusive**. The registered
   experiment required a validated Jacobian before FISTA, Arm B, or Arm C;
   those arms did not run. The result is neither `mechanism_pass` nor
   `frozen_method_negative`, and it is not an infeasibility certificate.
3. Retire the **autograd-linearized CFS-00A submethod** on the frozen native
   pi0.5 path. A matrix rejected in all six fixed checks cannot be supplied to
   FISTA, used as a warm start, or relabeled as the model's actual vector
   field. Do not weaken the gate, tune its tolerances or epsilons, or bypass
   it.
4. Do not reject the broader research hypothesis. This run never tested
   whether an actual-forward, time-dependent residual schedule can transport
   the privileged action target. It tested and rejected only the proposed
   autograd local-linearization interface on one case.
5. The next useful experiment must touch the central question directly: use
   the frozen sampler's **actual forward evaluations** to search for a
   budgeted time-dependent residual schedule, compare it with the existing
   equal-split residual baseline on the same paired case, and replay the
   selected candidate through the exact recurrence. A small deterministic
   repeatability check may fail closed first, but there must be no additional
   chain of apparatus-only retries.
6. Prefer a bounded zeroth-order optimizer or fixed finite-secant surrogate
   whose objective is evaluated by the actual sampler. The teacher may be
   expensive offline; only after a real teacher schedule passes would a
   separately authorized MLP learn the observation/state/time-to-residual
   field for fast inference.
7. Before any such run, separately preregister its parameterization, forward
   query budget, deterministic seeds, projection, candidate-selection rule,
   baseline, and stopping rule. Do not choose those values from this canary's
   outcomes. Preserve the source action target, paired observation/noise,
   first-five-XYZ support, physical scale-only displacement conversion,
   source path budget, per-step cap, and no-action-overwrite rule.
8. Close both checked-in CFS configs. No resubmission, IFT-01, population,
   label collection, probe training, or residual-field MLP training is
   authorized here.

## Root problem, in plain language

The current method asked autograd for a local map from a small flow correction
to the final action. The real mixed-precision sampler did not move according
to that map. It also moved differently at the two registered perturbation
sizes. Therefore the optimizer would be following the wrong map before it
ever created a candidate.

This is why the result does not say “flow steering cannot work.” It says “do
not learn or optimize from this unvalidated autograd Jacobian.” The direct
continuation is to optimize against what the frozen sampler actually returns,
then learn that successful time-dependent correction only if it exists.

## Exact next action

Write a new preregistration for one paired, actual-forward, derivative-free
transport canary. It must compare the existing equal-split residual with one
frozen bounded zeroth-order search under the same target and budget, stop
before simulator efficacy, and remain fail closed until reviewed. Do not
submit it under this ADR.
