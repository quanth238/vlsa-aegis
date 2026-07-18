# SafeLIBERO post-publication analysis v2

This is a derived analysis of an already accepted v1 population. It does not
edit, replace, or relax the immutable v1 publication.

## Scope

- Methods: `pi0.5` and `pi0.5+AEGIS` translational arms only.
- The AEGIS arm is conditioned on frozen per-case Codex obstacle labels.
- The denominator is the complete frozen 1,600-case SafeLIBERO population,
  not the earlier 20-case collision-conditioned diagnostic set.
- The Codex-conditioned result is not claimed to be an exact end-to-end
  reproduction of the paper's unavailable GLM-4.5V semantic selector.
- OpenVLA-OFT is not included.
- The v1 run has no continuous minimum-clearance/SDF signal.
- Sampled MuJoCo contact and the paper-compatible displacement CAR metric are
  reported separately.
- A safe episode with zero translation is an observed association, not proof
  that stopping caused safety.

## Added evidence

- Exact CAR/TSR numerators, denominators, and ETS/action-count sums.
- Four-way safety/task outcomes and exact paired transitions.
- Baseline and AEGIS sampled-contact strata, CAR/contact disagreement counts,
  and paired sampled-contact transitions.
- Baseline-versus-AEGIS native goal-progress deltas.
- Intervention, no-execution, and strict-zero-translation association counts.
- Failure cross-tabs by suite, safety level, task, and frozen label.
- Four exact no-usable-3D-points subclasses that distinguish no detector box
  from a detector box that produced no usable reconstructed points.
- An enriched paired gallery with filters for outcomes, physical-contact/CAR
  disagreement, observed failure class, and no-points subclass.
- A receipt binding every v2 artifact to the immutable v1 publication receipt,
  v1 summary, prepublish validation receipt, source commit, and accepted-result
  ledger.

## Build

Run on a compute allocation with read access to the immutable population:

```bash
python3 analysis/build_safelibero_postpublication_v2.py \
  --config /path/to/protocol.json \
  --manifest-receipt /path/to/manifest-receipt.json \
  --manifest /path/to/manifest.jsonl \
  --results-root /path/to/immutable-run/tasks \
  --v1-publication-receipt /path/to/population-publication.json \
  --v1-summary /path/to/population-summary.json \
  --prepublish-receipt /path/to/population-prepublish.json \
  --output-root /path/to/unused/analysis-v2
```

The output root must be unused. `analysis-v2-receipt.json` is written last;
its absence means the derived publication is incomplete. The source v1
publication must be bound to exact runtime commit
`1592aa59361f431ba96c6ddcbebcb596f6c20853`.

The derived directory contains:

- `population-summary-v2.json`;
- `aegis-failure-cases-v2.jsonl`;
- `aegis-failure-report-v2.json` and Markdown;
- `gallery-v2/index.html`;
- `analysis-v2-receipt.json`.

## Exact VinUni launch gate

The reviewed control-plane entry point is
`scripts/submit_vlsa_postpublication_analysis_v2_28610.sh`. It is shell-only
on the login node. It refuses to submit until exact publisher job `28610` is
`COMPLETED` with exit `0:0`, and it submits one CPU-only job with dependency
`afterok:28610`. The job is held, inspected, and immutably receipted before
release.

The launch is intentionally not performed by this implementation commit.
After publisher `28610` is terminal and the final analysis release commit is
reviewed, supply all of these exact identities:

```text
EXPECTED_PUBLICATION_RECEIPT_SHA256
EXPECTED_ANALYSIS_GIT_COMMIT
EXPECTED_BUILDER_SHA256
EXPECTED_RUNNER_SHA256
EXPECTED_SBATCH_SHA256
EXPECTED_SUBMIT_HELPER_SHA256
EXPECTED_CONFIG_SHA256
EXPECTED_MANIFEST_SHA256
EXPECTED_MANIFEST_RECEIPT_SHA256
EXPECTED_V1_SUMMARY_SHA256
EXPECTED_PREPUBLISH_RECEIPT_SHA256
```

The launcher fixes population array `28609`, publisher `28610`, runtime
source `1592aa59361f431ba96c6ddcbebcb596f6c20853`, all source/result descriptor
paths, and the unused analysis destination. Its intent, job ID, held-job
snapshot, submission receipt, and release receipt each have an immutable
SHA-256 sidecar. A rerun recovers the exact recorded job; it never uses a
broad cancellation or submission command.
