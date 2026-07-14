# 0002: Separate evidence tiers

Status: accepted — 2026-07-14

Dependency-free geometry and synthetic end-to-end tests verify harness behavior only. They never support a research claim. Artifact provenance uses explicit tiers:

- `synthetic`: implementation fixture;
- `real_safelibero_preliminary`: allocation-backed baseline experiment with disclosed deviations;
- `safelibero_validation` and `safelibero_test`: permitted only after all preceding gates pass and the protocol is frozen.

The initial real runner is preliminary because its optimizer queries simulator geometry directly and released obstacles are movable. Those limitations are stored in every result, not only in prose.
