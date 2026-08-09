# Direct controller-rollout margin result

Clean H100 job `37508` completed on `worker-2` in `00:01:44` from preregistered
commit `d70c6c5f16397645f1de5dd164cca65de97da9fa`. The allocation test,
training, 260 fresh exact two-step rollouts, and independent validator all
completed with exit code zero.

## Decision

`research_direction_go=false` for this grouped direct-margin model. Closed-loop
E05 remains unauthorized.

The model did improve E05 active-boundary value error over the current-
clearance/no-motion baseline (`23.076 mm` versus `29.279 mm` RMSE), but failed
the remaining gates:

- E05 active-gradient cosine mean/minimum was `0.643/0.613`, below `0.8`;
- gradient magnitude relative error averaged `0.908`;
- conservative false-safe counts were `272/289/222` on E05/E10/E15 (`783`
  total);
- the exact radius-`0.1` learned direction improved the worst E05 margin by
  `0.741 mm`, but 43/256 random directions improved it at least as much,
  giving add-one `p=0.1712` rather than the required `<=0.05`;
- the model predicted every nominal E05 row positive (`21.803--33.409 mm`),
  while the exact active L5-part-1 margin was `-3.442 mm`.

Because both raw and validation-calibrated predictions called the unsafe
nominal chunk safe, neither projection invoked OSQP. Exact verification of
that unchanged chunk remained proxy-unsafe. This is a model/state-
generalization and calibration failure, not a QP solver failure.

The continuous target is more informative than binary contact—the learned
direction raised exact clearance while the random mean change was
`-0.062 mm`—but it is not sufficiently discriminative or conservative on the
unseen state. Merely replacing binary BCE with direct margin regression does
not validate the proposed controller.

## Artifacts

- model SHA-256: `406710efb29894ea10ed2563173d7472493f0114ff40d88f937fdd3cc8166476`
- training SHA-256: `ed6ed8264fade59d8b76e1880382d26a6f155d5173f0c9312fbbdfd79028eaae`
- result SHA-256: `1fd5a445a9a39f1d2f35b9748b5a33d01c7da8998cd754ce8439b9417c2000bc`
- validation SHA-256: `25677a62e783176cad9cf0a0df33a8f14428eb2c8e447c32ec90fd7a4a4c9352`
- result payload SHA-256: `a5d28a470befd822a98980dc2900f9ce49a6e6d69a230aa6bf0e631a462f6bcc`

The retained next research target is state-conditioned prediction of the
local affine safety coefficients and their conditional lower-bound error,
rather than another absolute-margin MLP trained longer on the same episode
coverage. The privileged coefficient representation already passed job
`37280`; a future learned test must collect many distinct states and keep
complete state/episode groups separated.
