# Distal-only 8 mm closed-loop v1 result

Clean H100 job `37185` completed on `worker-1` in `2m10s` from commit
`169c52ff5f9ecb9649a81618c82adb0fb7948ac2`; all 45 allocation tests and the
independent result validator passed.  Completed Table-1 artifacts remained
read-only.

The v1 method failed closed at action `242`, after `242` executed actions and
`49` live pi0.5 queries.  It had no robot or protected-link contact, no paper
CAR event, and maximum obstacle displacement was only
`2.275e-11 m`, but the task had not completed.

This is not a distal-margin failure.  At the rejected transition, all seven
exact-box distal minimum-substep gaps were `104--378 mm`, far above `8 mm`.
The only violated row was the released EE proxy measured against the exact
box union (`-2.892 mm`).  The QP was valid (`1.224 ms`) and the 85 exactly
tested actions remained raw-contact-free, but none could make that conservative
EE/box support gap nonnegative.  The best was `-0.212 mm`.  The prior action
`241` had already required an EE-only `0.0998`-L2 correction.  Therefore v1
changed the EE obstacle representation instead of retaining the original
AEGIS EE constraint.

The frozen result/validation/preflight/MP4/JPG SHA-256 values are:

- `31f47bf797ebc19a49455d85185d369d30eae288043cbb2d999ff13de439cb46`
- `aaabfd8c111350261f06a75ee05c722c4734e558ed242da9e9a37f7c2b08846a`
- `6593126a9593e018b2e1cc6fb3575a831af532cbb22a343f8bda40a6dca233bf`
- `094e3b9f5d3f54d80aaa7f3635601c443a35267f2b46224b57499894b8a90b9c`
- `160c2ece4cfd5341bf5b8814df5bd5174d3c8869eb79d7383d490ba6f99569f9`

The result payload SHA-256 is
`8cff9c60a73f6c7b1aaba41f75cfac357aa563118de85b7039b1b1777b665589`.
The artifact remains a valid no-go and is not overwritten or reinterpreted as
a distal collision result.
