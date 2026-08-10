# Shared time-conditioned execution decoder result

## Verdict

H100 job `38076` and its independent replay completed successfully. The shared
time-conditioned decoder is a strict NO-GO. It preserves useful action
sensitivities and zero test false-safes, but it substantially worsens boundary
accuracy, safe-action recall, terminal joint prediction, and eligible-state
support relative to the validated flat one-sided model.

No new controller rollout labels, candidate expansion, calibration,
classifier, surface loss, Poisson/SDF, QP, or closed loop ran.

## Matched result

| Metric | Flat one-sided job 38070 | Time-conditioned decoder |
|---|---:|---:|
| Test false-safes | 0 | 0 |
| Test safe recall | 93.84% | 47.29% |
| Test support | 12 states | 6/13 eligible states |
| Test near-boundary RMSE | 2.671 mm | 15.700 mm |
| Test overall margin RMSE | 2.648 mm | 13.542 mm |
| Test joint RMSE | 10.864 mrad | 14.128 mrad |
| Test terminal joint RMSE | 18.158 mrad | 22.889 mrad |
| Joint-sensitivity cosine | 0.977 | 0.893 |
| Margin-sensitivity cosine | 0.921 | 0.883 |
| Validation false-safes | 1 | 30 |
| Validation support | 3 states | 6/9 eligible states |

The gradients remain above the registered 0.8 thresholds, but accurate
gradients are insufficient when the predicted trajectory and safety boundary
are wrong. The model passes only four useful gates: zero test false-safes,
validation safe recall, and the two sensitivity cosines. It fails both
eligible-support gates, validation false-safety, test recall, boundary RMSE,
test terminal improvement, and strict test support improvement.

## Validation pathology

The grouped validation split exposes a separate representation/support
problem. Terminal joint RMSE is 1341.234 rad for the source flat model and
1152.759 rad for the time decoder. Every saved time-decoder member selected a
checkpoint with a validation score above 1094. These magnitudes are not usable
robot predictions even though the shared head is numerically smaller than the
source on this split.

This is consistent with complete-input features that are unsupported or
explosively normalized in some held-out episodes, for example a feature with
near-zero training variance but a nonzero validation value. It does not rescue
the time decoder: on the normal-scale diagnostic test split, its terminal
joint error still increases from 18.158 to 22.889 mrad and its safety metrics
degrade sharply.

## Consequence

Do not train this decoder longer and do not add calibration, Poisson geometry,
a QP, or closed loop. The next decisive gate is a no-training audit of the
exact normalized complete-input representation by episode:

1. identify dimensions responsible for extreme validation z-scores;
2. distinguish truly unsupported robot/controller states from padding,
   presence-bit, constant-field, or normalization artifacts;
3. verify that the model-selection split has physically meaningful joint
   predictions before comparing another decoder;
4. collect grouped boundary episodes only if the unsupported dimensions are
   causal controller state rather than representation artifacts.

The earlier factorized mechanism evidence remains, but neither the flat model
nor this shared time decoder is a reliable safety predictor. Newly reserved
episode evaluation, QP, and closed-loop E05 remain blocked.

## Provenance

- Slurm allocation: `38076`, `worker-2`, one NVIDIA H100 80 GB HBM3
- elapsed time: `00:27:50`
- source commit: `1a3ca68c0cef97334da062a1164500d21e4ca749`
- model SHA-256: `6112d1c936375d9aaf54225b09ae1a9c6de2794d5417c1ad7c8d277c3f04021a`
- predictions SHA-256: `29068f5506ffb45ea51f672040aeb7a650c46d253344730c86a68b9f891c0ec2`
- result SHA-256: `4b98951a387aa1f7f21ece321e611c085274ede19b7b1a2825558c7d5a56c30f`
- result payload SHA-256: `b30c21fc0b6b05ef1636f86e7525665fc049ae9c6bcb1e317f502deb542f03e5`
- validation SHA-256: `54fefd2ed6d756f93e382bf657aabec2e909a57c571ddf999d28001711796eb4`
- validation payload SHA-256: `0546ae055a13ce32d3d0958903de27f49d18593a070687bcfb0bffc1e9eabf37`
