# H03 measurement evidence

Decision: **pass and advance to H04**.

The strict population used 50 unique SafeLIBERO saved states (episodes 0--49), one frozen policy-noise seed per state, and five identical simulator replays per state. Every replay produced the expected 126 samples: the branch point plus 5 actions x 25 physics substeps.

Allocation-backed sources:

- primitive calibration: Slurm job `27096`, `/mnt/data/quanth/experiments/crfs-oracle/measurement-calibration-20260714b.json`;
- population batches: Slurm jobs `27116` and `27117`;
- raw population root: `/mnt/data/quanth/experiments/crfs-oracle/h03-population-20260714a`;
- code commit used by the final batches: `99777cfd9b6674658ffa01447ad15df6f8818e11`.

Integrity:

- full primitive calibration SHA-256: `554abacdf0c58aceebc7056290f2c506f8a17240c59c9389d943668fb68ec454`;
- local generated summary SHA-256: `5562c19bbf279b436feb1a90d249d80017bf9254a215a1f32e89b0d5c00dc02b`;
- minimum-pair source audit SHA-256: `c3c60f04785f8aad02417803be8bd381a306c08afc387eb5abd2757ad8290501`;
- minimum-pair SVG SHA-256: `ece25c034dcb946b0ea23b5ef9bcdd5b27a0c71b92f5590b810b42d648d777dc`.

The raw gripper-mesh `mj_geomDistance` signal disagreed with contacts in 16/50 states and is advisory only. The primary controlled metric is the calibrated EEF-sphere/oriented-box signed distance evaluated from simulator transforms at every substep.
