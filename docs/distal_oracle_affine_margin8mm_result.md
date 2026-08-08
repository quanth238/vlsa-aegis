# Fixed 8 mm margin result

Clean H100 job `37175` completed on `worker-1` in 41 seconds from commit
`01bc8f42c25a718b028ca0800a686710db4ef99e`. The allocation-side tests and
the independent structural validator passed. Table 1 and the job-`37109`
false-safe ledger were read-only.

## Outcome

The fixed 8 mm intervention fails at the registered action-192 state.

- It does identify the nominal transition as unsafe.
- The interval-start frozen-proxy clearances for `L5_part_1` and `L6_part_0`
  are only `1.725 mm` and `5.608 mm`, already below the required `8 mm`.
- No one-step candidate can change that shared interval-start state, so none
  of the 60 trust-region candidates is proxy-margin-safe. Raw-safe candidates
  still exist.
- OSQP reports `primal infeasible` after 50 iterations. Rows 1 and 3 require
  affine increases of `6.539 mm` and `2.649 mm`, while their maximum possible
  values inside the registered action box are only `0.068 mm` and `0.121 mm`.
- Because the QP is invalid, no proposal is executed; therefore this test
  cannot claim raw collision avoidance.

The base audit's inherited `geometry_authority_failed_before_transition_learning`
stop reason also remains true: the frozen AEGIS obstacle MVEE is known to miss
the physical L6 contact surface. The margin-specific failure is more direct:
the fixed warning threshold is introduced too late at action 192. A meaningful
8 mm online test would need to activate at an earlier immutable state; that is
a different experiment and is not inferred here.

This negative result does not tune the `8 mm` value and does not change the
second user-requested test. The next experiment uses the exact 15 oriented
moka-pot collision boxes inside MuJoCo with the same action-192 state and QP
apparatus. That is privileged geometry feasibility evidence, not deployable
perception.

## Artifact identity

- Result file SHA-256:
  `8493d3cdd81ffb4b09cfe2c71af23e3f63d489bb34acbb7603877a011c1ca170`
- Canonical result payload SHA-256:
  `3088c8222add6a2d6fe3964edd39f84293f1dd9e25958897d1bca6cfd4832764`
- Validation file SHA-256:
  `8d39234a399fd85fd1896aa3ae28f23e78c7dff1e9565f75ef28b576c44ac1a9`
- Evaluation preflight SHA-256:
  `8f70b475421add0cbc1ab4969d09e18c8fc88a0d3c499d44daa3936b5e5d7f96`
- Remote result:
  `/mnt/data/quanth/experiments/vlsa-distal-oracle-affine-margin8mm-e05/margin8mm-20260809a/result.json`
- Local evidence copy:
  `/Users/quanth238/personal/Research/probe_vla/output/vlsa_distal_oracle_affine_margin8mm_e05/margin8mm-20260809a/result.json`
