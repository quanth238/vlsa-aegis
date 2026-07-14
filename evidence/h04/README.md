# H04 kinematics and clearance-transfer evidence

Decision: **pass and advance to H05**.

Slurm job `27123` ran from commit `dd58fd9a` on `worker-mig-3g40gb-0` with MuJoCo 3.2.3. Sixty signed axis probes across 10 saved states fit the world-frame response matrix; 20 random five-action prefixes were held out.

- median endpoint prediction error: 0.0009712704 m;
- 95th-percentile absolute `D_opt` / `D_sim` clearance error: 0.0010816360 m;
- false-safe cases at a 0.01 m predicted margin: 0.

Full artifact: `/mnt/data/quanth/experiments/crfs-oracle/h04-calibration-20260714a.json`, SHA-256 `528aa5f7b90034c6ac6b228f89bc198e01ec0d42f07e2d54e049bfb45aec8534`.
