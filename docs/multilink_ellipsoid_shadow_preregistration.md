# Multi-link ellipsoid shadow preregistration

## Question

Does released end-effector-only AEGIS miss physical link-5/link-6 contact on
`vlsa-t1-goal-ii-t0-e05` while continuing useful motion and completing the
SafeLIBERO task, and can a task-independent whole-arm ellipsoid QP be evaluated
at every unchanged AEGIS action with measured runtime?

## Immutable source evidence

- Table-1 result:
  `/mnt/data/quanth/experiments/vlsa-aegis-table1/vlsa-table1-contact-authority-population-20260718a/tasks/task-12/results/aegis/vlsa-t1-goal-ii-t0-e05/result.json`
- Expected file SHA-256:
  `273d77cba3fd5b1e457817ad20f8572b5628aeaa8b5b9f768b32e4f095fbad8b`
- Expected result payload SHA-256:
  `ec58c8581297751de33756e76efbf5b18e24a5f836aa6a8f147d0caebdd92d1c`
- Manifest row SHA-256:
  `d7a11ccf75af4b9f9823cad820fe261a4a791fc9891fe1f605cbed23648c88df`
- Initial state SHA-256:
  `714334cdae0ad6cd8540115802e9bf9b3591449e29d712ccc4651d3e89c22cf6`
- Policy-noise schedule SHA-256:
  `b50949ebc3b6a774f3800922487c13b00ee76d1f26cec5d8d7cdd8d0197f63e5`

The validator reads this result but never writes below the Table-1 run root.

## Geometry and QP

The protected set is always `robot0_link1` through `robot0_link7`. Every
contact-participating MuJoCo collision geom contributes one enclosing
ellipsoid and one constraint against the released obstacle MVEE. The QP is

```text
minimize    1/2 (qdot - qdot_nominal)' G (qdot - qdot_nominal)
subject to  A qdot >= b
            -0.5 <= qdot <= 0.5 rad/s
```

with `G = I + 10 J_ee' J_ee`, `b_i = -10 h_i`, and
`h_i = support_gap_i - D_opt`. Exact parameters live in the immutable JSON
configuration. The observer returns a proposed joint velocity but never
executes it in this gate.

## Required validation

1. Run only inside one Slurm H100 allocation.
2. Pass the allocation-side geometry, derivative, configuration, and OSQP
   unit tests before simulation.
3. Reproduce the archived initial state, policy noise, AEGIS action sequence,
   first physical contact, CAR crossing, and native task success.
4. Preserve exact returned and executed action ledgers.
5. Produce one explicit multi-constraint QP outcome and timing record for
   every executed action. Infeasible outcomes remain in the artifact.
6. Cover all seven configured arm-link bodies and retain direct link-5/link-6
   contact evidence from MuJoCo.

Passing this gate does not establish that executing the QP prevents collision
or preserves task success. Those are dependent active-control experiments.

## Validated result

Slurm job `36757` ran on one H100 from clean commit `4822fd4`. Receipt SHA-256
`390683dae335f05ebcec8c339bb5fb8fc577a1a7129002c0139a333c9cdb22bb`
has status `validated`.

The live collision model contained seven mesh geoms, one per protected link.
The independent hand check reproduced each enclosing sphere directly from its
MuJoCo `geom_rbound`:

| Link | Collision geom | Radius (m) | Formula check |
|---|---|---:|---|
| 1 | `robot0_link1_collision` | 0.175118 | passed |
| 2 | `robot0_link2_collision` | 0.171559 | passed |
| 3 | `robot0_link3_collision` | 0.164639 | passed |
| 4 | `robot0_link4_collision` | 0.165694 | passed |
| 5 | `robot0_link5_collision` | 0.193834 | passed |
| 6 | `robot0_link6_collision` | 0.132899 | passed |
| 7 | `robot0_link7_collision` | 0.090883 | passed |

All 237 steps supplied exactly seven rows to one simultaneous QP. Of those,
182 solved and 55 were retained as primal infeasible. Total QP wall time was
1.478 ms mean, 1.931 ms p95, and 2.009 ms maximum; solver-only time was
0.0747 ms mean and 0.0967 ms p95.

The exact archived action ledger was preserved. Raw MuJoCo contact began at
action 187 on link 5 at `-0.0010594 m`, link 6 also contacted, and released
AEGIS still reported positive end-effector barrier `h=0.00931204`. The robot
continued 0.172512 m of end-effector travel and completed the native task at
action 236.
