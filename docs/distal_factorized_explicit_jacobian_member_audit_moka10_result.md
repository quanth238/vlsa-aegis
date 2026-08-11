# Frozen Explicit-J Ensemble-Member Audit Result

## Verdict

Clean H100 job `38696` independently validates the frozen audit and diagnoses
**member-level optimization/checkpoint failure**. None of the five immutable
job-`38673` members learns a useful OSC execution Jacobian on its own training
or validation episodes. Ensemble cancellation is real but secondary: the
ensemble averages already-wrong member Jacobians rather than destroying four
or more individually correct ones.

This is a strict NO-GO for calibration, QP, closed-loop control, new unseen
episodes, or a research claim based on the current explicit-J model.

## Immutable execution

- Slurm job: `38696`
- Host/device: `worker-1`, NVIDIA H100 80GB HBM3
- Runtime: `00:00:18`
- Source commit: `072e719162c273a698be3994ee10f466bfb8fc74`
- Source dirty state: clean
- Run:
  `/mnt/data/quanth/experiments/vlsa-distal-explicit-execution-jacobian-audit/explicit-jacobian-member-audit-20260811a`
- Config SHA-256:
  `40a77cb25e1ef18ae0a80dc2eaec2f19dd4ea8bb8128132ef148ad4090b8733e`

The independent validator reloaded all five model members, recomputed every
reported metric and decision, and reproduced all four stored audit arrays with
matching NaN masks and zero finite difference.

## Member-level evidence

The registered pass required both train and validation aggregate and terminal
cosine at least `0.8`, median norm ratio in `[0.5, 1.5]`, and mean relative
norm error at most `0.5`. No member passed.

| Member | Frozen epoch | Train cosine | Validation cosine | Train terminal cosine | Validation terminal cosine |
|---:|---:|---:|---:|---:|---:|
| 0 | 10 | 0.093 | 0.109 | 0.091 | 0.107 |
| 1 | 20 | 0.124 | 0.126 | 0.107 | 0.090 |
| 2 | 49 | 0.003 | -0.005 | -0.019 | -0.025 |
| 3 | 28 | -0.119 | -0.125 | -0.131 | -0.125 |
| 4 | 184 | 0.211 | 0.195 | 0.085 | 0.064 |

The member-level gain is also generally too large. Validation median norm
ratios are `3.385`, `2.902`, `2.748`, `3.261`, and `1.670`; validation mean
relative norm errors are `4.322`, `4.062`, `2.886`, `3.630`, and `2.014`.
Member 4 is the best of the five, but remains far below the `0.8` direction
gate and still over-amplifies the aggregate response.

Trajectory prediction does not rescue these models. Validation joint RMSE by
member ranges from `27.294` to `42.244 mrad`; terminal validation RMSE ranges
from `37.918` to `50.892 mrad`.

## Action-family and cancellation diagnosis

Translation is the clearest failure. Across members, validation translation
mean cosine ranges from `-0.197` to `0.141`, while median norm ratio ranges
from `2.496` to `5.712`. Member 4's validation rotation response is better
(`0.386` cosine and `1.313` median norm ratio) but remains far below the
registered direction gate. No paired gripper secant has nonzero exact support,
so this artifact provides no empirical gripper-sensitivity claim.

The five members are mutually inconsistent: pairwise mean cosine is only
`0.024` on train and `0.028` on validation. The ensemble norm is approximately
half the mean member norm (`0.495` train, `0.491` validation), so averaging
does introduce cancellation. It is not the primary diagnosis because the
registered ensemble-cancellation explanation required at least four
individually passing members; the observed count is zero.

The frozen checkpoint epochs (`10`, `20`, `49`, `28`, `184`) are descriptive.
They do not distinguish inadequate optimization from checkpoint selection,
but the failure already on training rules out unseen-state coverage as the
primary cause of this result.

## Scientific interpretation

The explicit affine construction is structurally correct, as established by
job `38673`, but its learned Jacobian parameters are not correct. The current
objective/checkpoint process can fit some nominal trajectory behavior while
leaving the action-to-joint derivative wrong. A calibrated residual bound
cannot repair incorrect correction directions without becoming unusably
conservative.

Only a separately preregistered experiment that isolates intercept/nominal
trajectory optimization from Jacobian optimization and selects checkpoints by
held-out paired-secant fidelity is justified next. It must demonstrate useful
member-level train and validation sensitivity before any ensemble, unseen
episode, calibration, QP, or closed-loop evaluation.

## Artifact hashes

- Preflight: `f0264d66ca9403b3c53f295cc63610349d90c3dc65806b847106521eb32c4f5e`
- Records: `b9cbd99f5041a343856e95a8d360f1ab9152a994d32bb8d1e0c81892bcad9549`
- Result file: `f0c8ac63a7a57b00a63e795bb45efd7b0c434f8f031fe08eddacc188438922fe`
- Result payload: `7d8eca26d0f362c0f3040f99a0ae5cc9385cafcf99f733bcb2373b126ef0dc24`
- Validation file: `8204336a9759ab67b18b7e0017d0877b17b3e1b8ea984607a26adfc322735ff4`
- Validation payload: `fca74d7c66e2fca1b2d7715352786b779a95a61fafca0c92cf917d82f2c5bbbb`
