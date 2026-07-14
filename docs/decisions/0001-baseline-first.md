# 0001: Start from the VLSA-Aegis baseline

Status: accepted — 2026-07-14

The repository history and worktree start at THU-RCSCT/VLSA-Aegis commit
`57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b`. CRFS is implemented as an additive experiment layer around the released SafeLIBERO task construction, action execution, and OpenPI policy stack.

The baseline paths remain the reference behavior:

- `main/main_aegis.py` and `main/main_aegis_translational.py` for the released AEGIS evaluation;
- `safelibero/` for benchmark tasks, states, objects, and environment wrappers;
- `openpi/` for the released policy client/server and π0.5 sampler.

Opt-in CRFS code may expose traces, inject residuals, or add more precise measurement. With no CRFS controls, it must not change the baseline sampler output or runner API.
