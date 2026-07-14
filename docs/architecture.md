# Architecture

## Process boundary

The baseline has two incompatible dependency environments and already communicates over WebSocket:

```text
SafeLIBERO simulator process                 OpenPI policy process
Python client environment                    Python 3.11 / PyTorch

fixed state + rendered observation  ----->   transforms + frozen π0.5
fixed 10x32 Gaussian noise          ----->   trace x_t, v_t, predicted clean
physical 5x3 correction             ----->   scale-only normalization + padding
                                      <-----  physical 10x7 action chunk + trace

reset + settle + execute first five actions
measure every hidden physics substep
```

CRFS reuses this boundary. It does not combine the simulator and large model into one Python environment.

## Component ownership

| Component | Baseline source | CRFS responsibility |
|---|---|---|
| Task/state construction | `safelibero/libero` | Select immutable task, episode, and environment seed |
| Controller/action semantics | robosuite OSC_POSE | Retain seven-dimensional baseline actions and execute first five |
| Physics loop | `ControlEnv.step` | Add callback-enabled equivalent for substep measurement |
| Policy transforms | `openpi.policies.Policy` | Remove `__crfs__`, scale physical displacement without offset, pad |
| Flow integration | `PI0Pytorch.sample_actions` | Trace and opt-in residual/bridge arms; default unchanged |
| Oracle runner | new `main/crfs_oracle` | Branch, repair, execute, measure, and write atomic paired artifacts |
| Harness | new `src/crfs_harness` | Manifests, schemas, atomic output, aggregation, synthetic fixture |

## Safety and evidence boundaries

- `D_opt` is the clearance model used by offline label optimization.
- `D_sim` is the independently reported EEF-sphere/oriented-box distance computed from MuJoCo substep transforms, with physical contact as a one-way conservatism check.
- Raw mesh--box `mj_geomDistance` is stored for diagnosis only after H03 found false negative values without contacts in 16/50 states.
- The current preliminary runner queries `D_sim` during optimization and therefore cannot pass the D_opt/D_sim independence gate. This is explicit provenance, not hidden technical debt.
- The direct repaired action is a teacher upper bound, not a flow-steering result.
- Equal-norm random and oracle interventions use the same observation, noise, step, and correction norm.
