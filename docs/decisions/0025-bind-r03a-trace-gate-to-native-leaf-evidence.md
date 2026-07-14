# 0025 — Bind the R03A trace gate to exact native-leaf evidence

Status: accepted on 2026-07-15 after H100 task `27637_0` failed its aggregate
trace boolean, but before either strong-analytic arm was requested or any
analytic action, rollout, latency, or outcome was observed.

## Context

Task `27637_0` executed on the immutable R02 source host, `worker-1`, at clean
commit `45317a9ac09e49829087e941a7a397f358078115`.  It reproduced the case,
observation, branch snapshot, geometry, noise, and frozen physical actions.
The bounded diagnostic then reported the same six source/fresh trace keys.  For
every leaf it reported the same dtype and shape, `np.array_equal=true`, native
bytes equal, identical SHA-256, and zero maximum and RMS absolute error.  The
leaves were `predicted_clean`, `predicted_clean_physical`, `step_index`,
`time`, `v_base`, and `x_t`.

Despite that complete exact evidence, the older aggregate `_same_trace`
boolean returned false.  The immutable launch-failure artifact has SHA-256
`51d4c28a14fd32c13f8b12856b40dd028b57fb0177b5607696a8aa30a5e165a9`.
The client log containing the diagnostic has SHA-256
`7561147205db3acf69b19f9ce1a7f770139657df92fb67cd997007e525a6a2ea`;
the allocation log has SHA-256
`81309d927a5cc6d796e4511d76db10fa661a7b4722dc822b70a383f060aac24b`.
The job stopped in `r03a_runner`; no analytic request or simulator rollout was
made.  The evidence proves an inconsistency between the aggregate predicate
and its leaf diagnostics.  It does not identify the lower-level Python cause,
and it is not evidence of numerical drift.

## Decision

1. Keep trace pairing exact.  Introduce no numerical tolerance and do not use
   ADR-0010 portability limits to pass this gate.
2. Replace the inconsistent aggregate decision at the historical source gate
   with one predicate computed from the same bounded diagnostic it reports.
   Passing requires all of the following simultaneously:
   - both mappings contain only string keys and their raw key sets are equal;
   - every paired leaf has the same supported Boolean/integer/float dtype and
     shape, and every numeric value is finite;
   - every leaf is element-exact under `np.array_equal`;
   - every contiguous native byte sequence and its SHA-256 are identical; and
   - the canonical source and fresh trace-record SHA-256 values are identical.
3. Fail closed on any missing/extra key, dtype/shape change, signed-zero byte
   change, unequal value, unequal native byte, unequal leaf hash, or unequal
   canonical record hash.  The diagnostic and the decision must therefore be
   unable to disagree again.
4. Retry only case 0 on its hash-bound source node with the unchanged config,
   arms, seeds, budget, checkpoint, and exact action gate.  Use a new immutable
   run identifier.  If this predicate fails, follow ADR-0024's contemporaneous
   privileged-control fallback; never tune the predicate from the mismatch.
5. Before that retry, make final-artifact semantic validation prove the retained
   pairing payloads rather than trusting equal hash strings.  Every binding must
   contain exactly `source`, `fresh`, `source_sha256`, and `fresh_sha256`; the
   source/fresh payloads and their verified identities must be equal.  Bind the
   fresh noise to provenance and the fresh frozen actions/trace to the retained
   frozen arm.  Recompute every retained primary/duplicate trace leaf identity,
   duplicate action identity, and deterministic duplicate science-trace
   equality.  Writer, resume, and population-summary trust paths continue to
   require both semantic and JSON-Schema validation.  The already frozen schema
   and config hashes do not change; schema-only validation is not a trust path.
6. The full grouped population remains forbidden until the replacement smoke
   produces a schema-valid final artifact and its semantics are reviewed.

## Consequences

This is a stricter exact predicate than the former key-plus-array-equality
helper because it additionally binds dtype, native bytes, leaf hashes, and the
canonical trace record.  It changes no scientific intervention or decision
threshold.  The paired-payload hardening only makes the pre-existing artifact
contract fail closed.  Task `27637_0` remains an apparatus failure and supplies
no strong-analytic result.
