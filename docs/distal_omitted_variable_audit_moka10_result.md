# Omitted-variable sufficiency audit result

## Verdict

The historical 56D input is provably insufficient for the two-action L5--L7
rollout-margin target. Complete OSC inputs still do not make the current plain
MLP pass prediction.

Clean H100 job `37933` completed on `worker-1` in `00:16:58` from commit
`cdd10eb830654d344ca0040ef9c00be01ffaa09f`; independent validation passed.

## Determinism

All 10,625 complete snapshot/action pairs and 21,250 duplicate rollouts had
bitwise-identical seven-row margins, identical canonical contact receipts, and
identical next-state hashes. The maximum repeated margin difference was zero.

## Controlled omission result

All 2,106 effective interventions preserved the historical 56D bytes. Among
117 boundary-selected state/candidate pairs, 83 identical old inputs produced
both safe and unsafe proxy margins; 41 also changed raw-contact class.

| Omitted group | Proxy-class crossings | Raw-contact crossings | Maximum margin change |
|---|---:|---:|---:|
| rotation action | 83 | 41 | 23.367 mm |
| goal orientation | 73 | 29 | 13.294 mm |
| controller memory | 37 | 5 | 6.098 mm |
| gripper command | 8 | 0 | 0.967 mm |

Thus the old56 representation cannot define a deterministic safety function
on this controlled population. Rotation action is the strongest omitted group
and is the only group authorized for the next targeted two-arm input ablation.
No six-arm ablation is authorized.

## Complete-input model

The immutable complete-input test result still has safe support in 15/15
states but 271 false-safe actions. It therefore fails the required prediction
gate. The result is not “missing inputs alone solve learning”; it is
`old56_insufficient_but_complete_plain_MLP_still_fails`.

The QP, calibration, and closed-loop E05 remained frozen.

## Artifacts

- result file/payload:
  `27335bce0fb20c1cc9f86d7edd78abf034fcc83f73fe8ab3e97488766659069c` /
  `bd66ef991ac631f46ef44768da4f9bc823dde4ad09b7c6c98e850f0c4c719531`;
- validation file/payload:
  `1116d9b909adda1c8d4af11590514c643ff26aef83e0c94e5acf2465d234c80a` /
  `fd957374c5d78fe98b5213f194dc64a5e55e7c95705c269af28d0e468131f20e`;
- preflight: `feacd7a6fb4c594b0f010fcc6721f81601ce714a2d4cc011075495f967e9544f`.
