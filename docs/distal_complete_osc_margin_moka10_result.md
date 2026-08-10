# Complete-OSC action-conditioned margin result

## Verdict

The missing-OSC-input hypothesis is rejected as a sufficient explanation for
the earlier MLP failure.

The complete recorded input uniquely determines the registered deterministic
cloned-OSC rollout: all 10,625 snapshot/action pairs were executed twice, and
all 21,250 replays produced bitwise-identical seven-row margins, identical raw
protected MuJoCo contact receipts, and identical next-state hashes. The
maximum repeated margin difference was exactly zero.

However, the complete-input action-conditioned MLP is a strict NO-GO. On the
untouched E05/E10/E15 test episodes it produced:

- 271 proxy false-safe actions (required: 0);
- 100% exact-safe recall and support in 15/15 states, but only because it
  effectively accepted all 1,875 test actions;
- 7.529 mm overall RMSE (required: at most 3.808 mm);
- 8.497 mm RMSE over 822 near-boundary rows (required: at most 2 mm).

There were 1,604 exact-safe and 271 exact-unsafe test actions. Thus the 100%
recall is not useful conservative behavior.

## What the experiment establishes

The mapping is deterministic given the stored state and full two-action
command, so omitted stochastic or hidden OSC state is not the immediate
blocker. Adding the complete raw state did not improve the learned safety
field; accuracy became worse than the earlier 56D model.

The fixed input has 2,110 complete-state dimensions and 2,131 dimensions after
adding both 7D actions and row identity. Because SafeLIBERO task models have
different simulator-state schemas, the representation uses a lossless union
of named values plus presence masks. All five ensemble members selected epoch
0, and their validation scores were extremely large (2.61e6--3.17e6 mm).
This is strong evidence that raw padded simulator state with train-only scalar
standardization is a poor cross-task representation. It does not show that a
compact structured or latent state representation cannot work.

No calibration, QP, pi0.5 feature, or closed-loop E05 execution ran. A QP
cannot be evaluated from a predictor that marks 271 unsafe actions safe.

## Evidence

- H100 job: `37910`, `worker-2`, completed in `01:29:22`.
- Source: clean commit `e6eab3f3ba70e0b9d741990512a3292534a0ca65`.
- Run root:
  `/mnt/data/quanth/experiments/vlsa-distal-complete-osc-margin/complete-osc-margin-20260810d`.
- Dataset file/payload SHA-256:
  `585f696eedef9bd9a4c9d0ba576d6d86e5660638696862ec2d5d1af434fb426b`,
  `b84415c650cf5b11bce9f344bcd4fd288a136d3aaa15ce4458f23d0f3a20aa6a`.
- Collection result file/payload SHA-256:
  `bd85f3175c790a153833a7c903a4e119c71f771271b2ecb92cc1c579269ce966`,
  `4e4a415ef25ca217fc43a23f94dd82af66f782434647f79c29f72f47a90226a5`.
- Model SHA-256:
  `146e4e16745844fe50bbc802f958956239fb440aed99648eebf4e56358aa8a5d`.
- Result file/payload SHA-256:
  `3d75740f073129f4e71342c3b48a49ffc66db5c5bcec69bcd1bc705443bc50d2`,
  `4cfdb62a938b09b18b432336ca018ad1f5451519f012d3eee6e264f112cfd925`.
- Validation file/payload SHA-256:
  `93e99a7eb74a7f3bde4bce909bf4556e2793dd5d2ef581bf519375471286f92b`,
  `586e59310d88bda883a7b3613ef228451b70da22a4ce5a44b67b59761bc0a180`.
- Preflight SHA-256:
  `2dfbf59755afb87922414cc566b0b1de65e61f7b6e52e9a6fbb43fe5f6d0c413`.

The earlier jobs `37891`, `37892`, and `37900` are recorded apparatus
failures. They produced no final scientific dataset and were not resumed.

## Next decision

Do not add calibration or a QP to this model. The next research experiment
must change the representation or model, not merely add more OSC fields. A
reasonable separate preregistration would compare a compact structured
controller/geometry state or learned per-task latent encoder against the
validated 56D baseline while keeping the same deterministic dataset and
episode-grouped split. It must first beat the margin and false-safe gates.
