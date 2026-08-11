# Frozen direct-horizon root-cause audit

The direct-horizon result is already independently reproduced and remains a
strict NO-GO. This read-only audit distinguishes failure already present in
the fitted train population from held-out episode coverage/generalization.

It reloads the immutable five-member model and existing source/reserved
rollouts. For train, validation, old diagnostics, and new reserved episodes it
reports joint error by horizon, signed safety-normal joint error, false-safe
link/substep, state-only distance to training support, action-source error,
joint/safety sensitivity direction and magnitude, ensemble disagreement, and
state-level safe support. The six reserved states rejected despite exact-safe
candidates are listed explicitly.

After the non-scientific job-`38481` apparatus failure, the user sharpened the
decisive question before retry: paired joint and per-step safety sensitivities
must be reported for all 14 action coordinates at all 51 horizons, separately
for train, validation, old diagnostic, and reserved groups. Direction must
reach cosine `0.8`; useful magnitude requires median norm ratio in `[0.5,1.5]`
and mean relative norm error at most `0.5`. Train/validation failure identifies
loss/decoder underfitting; train/validation pass with reserved failure identifies
state coverage/generalization. V2 config SHA-256 is
`69c40e545aeff59a603b4afa6a95c01b5d6c0bea0c8f3370b52a4a7e0d915f49`.

Predicted and exact joints are passed through the same frozen FK, attached
L5--L7 ellipsoids, and fixed-k0 obstacle union. A per-state diagnostic
`dh/dq` is fit only from the registered 28 finite-difference candidates to
evaluate `dh/dq^T(q_hat-q*)`; it is not a learned control component.

No labels are recollected and no model is trained. Calibration, QP, Poisson,
classifiers, and closed loop remain forbidden regardless of the outcome. The
future task-3 intervention manifest remains unopened.
