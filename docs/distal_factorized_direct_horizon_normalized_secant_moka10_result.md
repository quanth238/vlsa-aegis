# Direct-Horizon Normalized Paired-Secant Result

## Verdict

Strict NO-GO for the current direct-horizon decoder. Replacing only the raw
secant Huber loss with train-RMS-normalized paired vector MSE does not make the
model learn the correct action-to-joint-motion gain on fitted states.

## Matched experiment

H100 job `38586` ran on `worker-1` from clean commit `9ce9db5`. It retained the
job-38376 architecture, complete structured input, trajectory and symmetric
safety-normal losses, 7,905 actions, grouped splits, five seeds, optimizer,
schedule, and early stopping. It added no one-sided loss, calibration, QP,
closed loop, or new rollout labels. The frozen future task-3 population was not
opened.

The train-only normalization retained 450 nonzero action-coordinate/horizon
cells, excluded 264 zero/noisy cells, and clipped 22 high inverse-scale
weights.

## Fitted sensitivity gate

| Metric | Train | Validation | Required |
|---|---:|---:|---:|
| Aggregate cosine | 0.691 | 0.658 | >=0.8 |
| Aggregate median gain ratio | 0.046 | 0.049 | 0.5--1.5 |
| Aggregate mean relative gain error | 0.946 | 0.939 | <=0.5 |
| Terminal cosine | 0.902 | 0.838 | >=0.8 |
| Terminal median gain ratio | 0.017 | 0.018 | 0.5--1.5 |
| Terminal mean relative gain error | 0.980 | 0.977 | <=0.5 |

Validation joint-trajectory RMSE is `19.275 mrad`, below both the frozen
`19.662 mrad` baseline and the registered `21.629 mrad` limit. The network can
still fit average motion, but its response to changing the Cartesian action is
far too weak. At substep 26 the gain temporarily reaches `0.632/0.977` for
train/validation, then collapses to `0.017/0.018` by substep 50.

## Validation and decision

The independent validator reloaded the five-member ensemble, recomputed all
7,905 joint/FK/ellipsoid traces, reproduced five stored arrays with matching
NaN masks and zero finite difference, and exactly matched every metric and the
NO-GO decision.

Per preregistration, do not evaluate new unseen episodes and do not add
calibration, a QP, or closed-loop control. The next authorized experiment is a
matched time-decoder change without additional losses. This result rejects
normalized loss scaling as a sufficient repair under the frozen optimizer,
schedule, and early stopping; it does not reject execution prediction as the
broader research direction.

Artifacts:

- model: `60791198b705cd36bc6ad7765d72089483f62fa68a7873c02d99d6e77ef90d2b`
- predictions: `1a5da8391be988980c4078e09dd943db85fadaa97a56ba5a5f8aa6b5d9571c29`
- result: `265e2a16d5dfd49877d48b8eab9961f4fa9b65d3680a9d8c9a1279fcf3abf868`
- validation: `5661919f60f8f250a7afa8939feeeeedc217780e95fe08c8b72397cf4a4b8312`
