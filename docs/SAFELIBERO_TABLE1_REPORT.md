# SafeLIBERO Table-1 report renderer

`analysis/render_safelibero_table1_comparison.py` creates the final concise
Markdown and Overleaf-ready LaTeX comparison only after the post-publication
population is complete.

The renderer is intentionally narrower than the population analysis:

- input one `vlsa_table1_population_summary.v2` with exactly 1,600 cases,
  3,200 paired results, 32 task-level groups, and no dropped result;
- require its terminal `analysis-v2-receipt.json`, validate the receipt payload
  hash and publication status, and bind the supplied summary's byte size,
  SHA-256, source-v1 provenance, population, and claim scope;
- require the accepted absolute publication topology
  `/mnt/data/quanth/experiments/vlsa-aegis-table1-analysis-v2/`
  `vlsa-table1-contact-authority-population-20260718a-publisher-28610/`
  `population-summary-v2.json` as provenance, while allowing a hash-identical
  local mirror with the same basename;
- accept only scientific terminal statuses (`complete`, retained hard method
  failure, or retained fail-open method failure), never an apparatus status;
- verify all group-to-suite and suite-to-average exact counts and sums;
- verify all paired CAR, task, and 16-way joint-outcome transition marginals;
- bind the paper values to the exact frozen
  `configs/vlsa_table1_translational.json` bytes, SHA-256
  `7dec2cdd473e6809a755d908711efd457c5fdd36ca9655ab6a15e0c9c4bdbfa6`;
- render the paper target, reproduced value, reproduced-minus-paper delta,
  and direct AEGIS-minus-pi0.5 effect;
- report exact CAR/TSR numerators and ETS sums;
- state that CAR is obstacle displacement rather than direct contact, that
  continuous clearance is unavailable, and that the paper semantic selector
  was not reproduced.

Run this locally only after the allocation-side post-publication analysis has
published its validated v2 summary:

```bash
python3 analysis/render_safelibero_table1_comparison.py \
  --summary /path/to/analysis-v2/population-summary-v2.json \
  --analysis-receipt /path/to/analysis-v2/analysis-v2-receipt.json \
  --paper-targets configs/vlsa_table1_translational.json \
  --output-root /path/to/unused/table1-report
```

The output root must be unused. The command writes:

- `table1-comparison.md`;
- `table1-comparison.tex`;
- `table1-report-receipt.json`.

The receipt is written last. If it is absent, the rendered directory is not a
complete accepted report. The LaTeX fragment requires `booktabs` and
`graphicx`. The report receipt records both the original allocation path from
the analysis-v2 receipt and the validated local mirror path.
