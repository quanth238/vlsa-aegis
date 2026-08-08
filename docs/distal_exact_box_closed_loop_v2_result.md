# Distal-only 8 mm closed-loop v2 result

Clean H100 job `37187` completed on worker-1 in `1m49s` from commit
`77c767913b26391a9a75ad69d0bc6fa3dfd1b2e3`; all 46 allocation tests and the
independent validator passed. It failed closed at action `135` after one
accepted correction at action `134`. There was no robot contact, protected
contact, CAR, or measurable obstacle motion, and the task had not completed.

No distal row caused activation: all were `114--376 mm` clear. The new
discrete EE/MVEE minimum-substep row alone was `-0.814 mm`. The released
continuous AEGIS QP had solved normally, but none of 85 candidates met the
additional discrete target. Thus even with the original EE geometry, v2
still augmented AEGIS with a different, stricter next-state rule.

Result/validation/preflight/MP4/JPG SHA-256 values are
`c327ddb30b7a0e9b1dc521c7f30d97c6c7217230d049001bc6792674352953d5`,
`0cfb53040abbbccf02d60c7e0d9e6e66c34d407b3c4b84a721bb3a5bc901eae7`,
`7e15d15e745b608389cf4d805ae06f1dc3943d9ac6fccd07b2a5b85b7125f7fa`,
`9fb0af065596bf0e07e8e189e520a1a8efde82c6ae76b0934da2d643450a5ee0`,
and `2719ff022fcb9915100e1dd011d6cf65c34fca8045b80f76559d8ce90ca33931`.
Payload SHA-256 is
`21c85e2d1150f2bcc6c8f6b982d37f85c0d9e344bdbbf35dfb0ef3828f342157`.
