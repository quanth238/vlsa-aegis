# SafeLIBERO post-publication analysis v3

This is a read-only decision analysis derived from the accepted analysis-v2
publication. It never edits the immutable population, v1 publication, or v2
publication.

## Question and scope

Analysis v3 explains where AEGIS failures occurred after the complete paired
population has already been accepted. It adds:

- hand-subtree versus unprotected robot-link contact authority;
- an exact decision partition for the AEGIS control flow;
- paired temporal evidence around the first intervention, nominal-action
  divergence, and native goal-progress divergence;
- one retained v3 row for every one of the 1,600 frozen cases.

The inputs remain the translational `pi0.5` and `pi0.5+AEGIS` arms conditioned
on the frozen per-case Codex obstacle labels. This is not an OpenVLA result,
not an exact reproduction of the unavailable GLM-4.5V selector, and not a new
simulator experiment. Temporal association does not establish causality.

## Dependency chain

The v3 job is unusable until all of these conditions hold:

1. population publisher `28610` is terminal `COMPLETED` with exit `0:0`;
2. the reviewed analysis-v2 job is terminal `COMPLETED` with exit `0:0`;
3. its immutable v2 submission and release receipts, summary, and terminal
   analysis receipt have independently accepted SHA-256 identities;
4. the reviewed v3 launch release is clean and descends from accepted v3
   implementation commit
   `59b7cdf42acfcb2c76e5cd7a50703ee1dd375e74`;
5. the exact v3 destination has never been used.

The shell-only entry point is
`scripts/submit_vlsa_postpublication_analysis_v3.sh`. It submits one held
CPU-only job with dependency `afterok:<analysis-v2-job-id>`, inspects the
exact Slurm contract, publishes immutable control receipts, rechecks every
input, and releases only that job. The allocation uses four CPUs, 32 GiB RAM,
four hours, no GPU, and excludes `worker-3`.

No v3 job is submitted by the implementation commit. After v2 is accepted,
the launcher requires these terminal values:

```text
V2_ANALYSIS_JOB_ID
EXPECTED_V2_ANALYSIS_GIT_COMMIT
EXPECTED_ANALYSIS_V3_GIT_COMMIT
EXPECTED_V1_PUBLICATION_RECEIPT_SHA256
EXPECTED_PREPUBLISH_RECEIPT_SHA256
EXPECTED_SUMMARY_V2_SHA256
EXPECTED_ANALYSIS_V2_RECEIPT_SHA256
EXPECTED_V2_SUBMISSION_RECEIPT_SHA256
EXPECTED_V2_RELEASE_RECEIPT_SHA256
EXPECTED_CONFIG_SHA256
EXPECTED_MANIFEST_SHA256
EXPECTED_MANIFEST_RECEIPT_SHA256
EXPECTED_V3_MODULE_SHA256
EXPECTED_V3_BUILDER_SHA256
EXPECTED_V3_RUNNER_SHA256
EXPECTED_V3_SBATCH_SHA256
EXPECTED_V3_SUBMIT_HELPER_SHA256
```

## Terminal artifact contract

The unused v3 directory is fixed to:

```text
/mnt/data/quanth/experiments/vlsa-aegis-table1-analysis-v3/
  vlsa-table1-contact-authority-population-20260718a-
  publisher-28610-v2job-<analysis-v2-job-id>/
```

It contains:

- `aegis-failure-cases-v3.jsonl`, exactly 1,600 rows;
- `aegis-failure-report-v3.json`;
- `aegis-failure-report-v3.md`;
- `analysis-v3-receipt.json`.

`analysis-v3-receipt.json` is written atomically and last. It binds the exact
v1 and v2 artifacts, v2 job and control receipts, v3 job and control receipts,
protocol inputs, reviewed source hashes, exact CPU resource contract, all
1,600 cases and 3,200 results, and every derived v3 artifact. A directory
without this terminal receipt is incomplete and cannot be accepted or
resumed.
