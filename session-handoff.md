# Session handoff

Objective: test whether exact safe action corrections causally steer the frozen π0.5 flow before training a learned probe.

Branch: `agent/crfs-oracle-harness`, derived from baseline `57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b`.

Current gate: H03 measurement audit. Read `PROGRESS.md`, then run `./init.sh`.

The next allocation should convert and hash the cached JAX π0.5 LIBERO checkpoint, import the modified server and SafeLIBERO client in their existing remote environments, and execute exactly one manifest case. Do not submit the validation array until the one-case artifact validates and the scientific deviations in its provenance are reviewed.

Never treat the synthetic fixture as research evidence. Never tune using `oracle_test.json`.
