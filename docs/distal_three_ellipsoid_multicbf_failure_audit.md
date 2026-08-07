# Distal three-ellipsoid multi-CBF failure audit

## Audited evidence

This audit concerns only primary case `vlsa-t1-goal-ii-t0-e05` and the
preregistered active result from H100 job `36782`. The accepted `result.json`
file SHA-256 is
`d28e29c06820817157d01c27d3f5fd366825992bcc75234086b36a868d9d80f0`.
Table-1 artifacts are immutable inputs and were not changed.

H100 job `36788` replayed the accepted 237-action multi-CBF ledger and encoded
an annotated MP4. Its receipt verifies all paired-state hashes and reports
exactly zero replay error for end-effector coordinates, active-obstacle
displacement, and raw contact distances. The video SHA-256 is
`dc7df3d03b5e893b0566e97a57c90be9593775948f8984aa27c90a98b0e17df0`.
The receipt SHA-256 is
`5bc7dd8a97b8f0b23e2dc88b1c4ac74f68e49924ac611021e59e29e472027883`.

## What passed

- The L5, L6, and L7 MVEEs enclose every compiled MuJoCo collision-mesh
  vertex. Their maximum normalized quadratic values are below one after exact
  farthest-vertex inflation.
- Every action supplied exactly three constraints to one QP. All 237 QPs were
  valid. The minimum recorded constraint residual was approximately
  `-5.6e-17`, numerical zero.
- The controller began materially modifying XYZ at action 182, seven actions
  before raw contact. Therefore the result is not caused by a constraint that
  was never activated.
- The QP is already inexpensive: solver-only mean `0.0862 ms`; total QP mean
  `1.736 ms`, p95 `2.297 ms`, maximum `4.831 ms`.

These checks rule out missing rows, loose robot-link bounds, QP infeasibility,
and solver runtime as the primary cause.

## Where forward invariance failed

At each state the implementation enforces the continuous, linearized row

`a_i(q_k)^T u_k >= -alpha h_i(q_k)`

where `u_k` is the normalized translational OSC action. It maps `u_k` through
a damped end-effector resolved-rate inverse and the instantaneous MuJoCo link
Jacobians. SafeLIBERO instead applies the action through a discrete OSC target,
internal feedback, and physics substeps before the next 20 Hz observation.
The QP does not roll out that transition and does not post-check or shorten a
step that crosses the barrier.

The accepted ledger exposes the mismatch directly for L5 (`dt = 0.05 s`):

| action | pre-step `h_opt` | QP-predicted `h_dot` | observed finite-difference `h_dot` | result |
|---:|---:|---:|---:|---|
| 182 | +43.562 mm | -435.625 mm/s | -285.943 mm/s | QP becomes active |
| 184 | +16.012 mm | -160.116 mm/s | -231.436 mm/s | simulator closes faster |
| 185 | +4.440 mm | -44.398 mm/s | -177.620 mm/s | next state crosses `h=0` |
| 186 | -4.441 mm | +44.412 mm/s | -143.104 mm/s | predicted recovery has wrong sign |
| 188 | -14.368 mm | +143.680 mm/s | -16.544 mm/s | L5 and L6 rows bind |
| 189 | -15.195 mm | +151.952 mm/s | +32.448 mm/s | raw L5 contact; recovery is too slow |

For example, action 185 predicts a next L5 barrier of about `+2.220 mm`, but
the next observed barrier is `-4.441 mm`. Once the sampled system has left the
buffered safe set, satisfying `h_dot >= -alpha h` in the inaccurate local
model cannot undo penetration within the same discrete step.

## Outcome and contributing limitations

Raw MuJoCo contact with `robot0_link5_collision` begins at action 189 and
persists through action 193; the minimum contact distance is `-0.8023 mm`.
Paper CAR also begins at 189, maximum obstacle L1 displacement reaches
`26.615 mm`, and the native task never succeeds. Relative to released AEGIS,
the controller delays contact by two actions and CAR by one, so the rows have
an effect but are insufficient.

Additional limitations worsen the counterfactual:

- only XYZ is executable, so distal-link motion is controlled indirectly
  through the damped end-effector inverse rather than by joint-space commands;
- the obstacle MVEE is the released AEGIS frozen perception geometry, so it
  becomes stale after the moka pot starts moving;
- the nominal 237-action AEGIS ledger is open loop after the first safety
  correction. It is not replanned from the divergent state, helping explain
  the lost task success and later repeated corrections;
- the QP uses no robust one-step margin, nonlinear simulator rollout, or
  post-solve step reduction.

## Decision

The three separate ellipsoids and three simultaneous QP constraints are
implemented correctly for their stated continuous model, but that model is
not a valid discrete safety certificate for the executed SafeLIBERO OSC
transition. Training a KKT/VI-informed network to reproduce this QP would
reproduce the same certificate faster; it would not repair the model-to-
execution mismatch. A next experiment must be separately preregistered around
joint-space execution or a robust discrete-time/nonlinear rollout constraint.
