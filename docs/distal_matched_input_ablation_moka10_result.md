# Matched input ablation result

## Verdict

The matched experiment rejects missing OSC inputs as the root cause for the
current plain MLP. Both input arms failed the preregistered near-boundary
training-fit gate, and the complete input was substantially worse.

H100 job `37925` completed on `worker-1` in `00:03:06` from clean commit
`4027618aa727783b000d2dfbbdf091087612c39c`; independent validation passed.
No simulation or new data collection ran.

## Pairing

Both arms used exactly 74,375 common rows. State, action, constraint, current
margin, and target margin bytes matched. The reconstructed 56D start
clearances differed from the fresh stored clearances by at most
`1.11e-16 m`.

| Arm | Train overall / boundary RMSE | Test overall / boundary RMSE | Test false-safe actions | Test safe recall | Test support |
|---|---:|---:|---:|---:|---:|
| old56 | 4.158 / 3.285 mm | 3.060 / 3.058 mm | 72 / 1,875 | 97.69% | 15 / 15 |
| completeOSC | 7.338 / 7.751 mm | 7.529 / 8.497 mm | 271 / 1,875 | 100% | 15 / 15 |

The complete arm's apparent recall is invalid as a safety result: it accepted
all 271 unsafe test actions. Its five members again selected epoch zero and
its cross-task validation values exploded under the frozen lossless padded
snapshot/scalar-standardization representation.

## Interpretation

The frozen decision is
`output_target_or_plain_MLP_representation_problem`. The compact arm also
failed to fit the training boundary, so this is not the registered
train-fit/test-fail signature of missing state/setup coverage. Complete OSC
inputs are sufficient to determine the deterministic rollout target, but
concatenating them into the same plain MLP does not learn that target.

This does not show that controller state can never help a better structured
model. It shows that adding those inputs is not the missing fix for the current
action-conditioned residual MLP. Calibration, QP, Poisson fields, more data,
and closed-loop E05 remain unauthorized by this gate.

## Artifacts

- immutable dataset file/payload:
  `585f696eedef9bd9a4c9d0ba576d6d86e5660638696862ec2d5d1af434fb426b` /
  `b84415c650cf5b11bce9f344bcd4fd288a136d3aaa15ce4458f23d0f3a20aa6a`;
- old56 / completeOSC model:
  `fbb986053ad8d2f1343c007635e8aa3d4977e531078cc1dfa6d9f8f6471d18c5` /
  `146e4e16745844fe50bbc802f958956239fb440aed99648eebf4e56358aa8a5d`;
- result file/payload:
  `dc76f3b6d4f8f78d08861bc13b9cec2a396c931bb5479f5646b729fe0e76698a` /
  `3a93f3d85097c9621a017999416d29cccababb316bbfd7d6d703362f78ba67c3`;
- validation file/payload:
  `fb28af937b906883c3502f9be6560503e41d7139922b01a8a458aaa3bc74b55f` /
  `8fcadb23b46c9e68e1ba39a3387b252f868c0a4b23ec7dc6b8d205f9fd7509c1`;
- preflight: `1cff17bdfb35737b15bcf1bc62641cf5f9ecc0ed208296b5fd8fdc330471b4a7`.
