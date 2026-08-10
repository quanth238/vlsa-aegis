# Factorized false-safe surface-error diagnostic

## Question

Do the 44 held-out factorized false-safe actions have larger L5--L7
obstacle-facing surface-position errors than comparable correctly rejected
unsafe actions? A positive answer authorizes a later surface-position-loss
pilot. It does not authorize training in this diagnostic.

## Frozen source and population

The diagnostic reads only the validated job-37980 dataset, predictions, and
result. It uses the 960 random-antithetic actions from untouched E05/E10/E15.
No new controller rollout or label is collected. The expected source count is
44 ellipsoid false-safe actions.

For each action, MuJoCo forward kinematics reconstructs the predicted and exact
51-step joint traces using the same fixed-k0 exact-box obstacle geometry and
seven L5--L7 ellipsoid rows as the factorized pilot. At the globally worst
exact-joint static constraint, the diagnostic records:

- link-center position error;
- exact-normal ellipsoid support-surface position error;
- signed clearance overestimate from center translation;
- signed clearance overestimate from ellipsoid support/orientation;
- active link, substep, obstacle witness, and witness-switch status.

The ellipsoid support point in direction `d` is

```text
p_surface = c + Q d / sqrt(d^T Q d).
```

This is the position whose obstacle-normal projection directly enters the
registered support-gap safety value.

## Matched comparison and decision

Each false-safe action is matched to a correctly rejected unsafe action from
the same state with the nearest exact dynamic margin. Controls are used once
when possible and reused only if a state has fewer controls than false-safes.

A later surface-position-loss pilot is supported only if all conditions pass:

1. all 44 source false-safes and all stored FK margins reproduce exactly;
2. false-safe median worst-witness surface error exceeds matched control by at
   least `0.5 mm`;
3. at least `65%` of paired false-safes have larger surface error;
4. unsafe-action surface retraction and static margin overestimation have
   Spearman correlation at least `0.5`.

If the gate fails, surface position is not the decisive missing supervision;
the next discussion should address one-sided uncertainty. Regardless of the
outcome, this job forbids training, surface-loss training, Poisson/SDF,
calibration, QP, VLA inference, and closed-loop execution.

Frozen config SHA-256:
`0dc6aafc413ebbd116a4a3651b5ee1aadbae44c19e75fb0808ec51da30eb93d8`.
