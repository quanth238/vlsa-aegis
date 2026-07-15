# R05A evidence

`ift00-synthetic.json` records the passed local IFT-00 implementation gate.
It is deliberately labeled synthetic: neither pi0.5 nor SafeLIBERO was run, so
it cannot support an efficacy, safety, latency, generalization, or novelty
claim.

The useful sentinel is narrow. On a deterministic nonlinear field, repeating
the final-action delta as the old constant residual missed the terminal target
by `0.048832`, while the constrained time-dependent solver reached the fixed
synthetic fidelity gate with path `0.067193 <= B=0.1`. Duplicate solves were
bit-identical. A separate coupled-field test proved that the solver can discover
a necessary control coordinate initialized at zero even when called beneath
the sampler's outer `torch.no_grad()` context.

IFT-00 also tests exact iterative float32 sampler times, exact target pairing
outside first-five XYZ (including padded dimensions and signed zero), the two
physical fidelity gates, hard timing/mask/path constraints, explicit
nonconvergence/nonfinite statuses, graph-free results, reverse-order controls,
and absence of parameter-gradient contamination.

The next scientific evidence must come from a valid one-case allocation
integration canary. ADR-0035 permits one intervening shell-only cgroup
capability artifact, but that is apparatus evidence only. Until a canary is
accepted, no accepted real inverse-flow transport result or simulator-efficacy
result exists.

`adr0031-apparatus-cpu-a.json` records terminal CPU-only apparatus job
`27797_0`. All registered suites passed 17/8/10/12 with zero skips, then the
task failed closed because its exact cgroup-v2 task leaf did not expose a
readable `memory.peak`. Synthetic unit-test solver calls ran, but no checkpoint,
policy server, real teacher observation, simulator action, or training ran.
This is apparatus evidence only and does not authorize retry C.

`cgroup-v2-current-capability-a.json` records shell-only task `27820_0`. It
completed on worker-1 with a positive finite 256 MiB job-scope `memory.max`, 20
strictly increasing `memory.current` samples, a 120 ms maximum gap, a 6,045,696
byte sampled high-water, and zero new hierarchical `max`/`oom`/`oom_kill`
events. The sampled high-water is a lower bound, never an exact peak. This
passes only the telemetry-capability gate and cannot authorize retry C by
itself, IFT-01, efficacy claims, or training.

`ift00a-sampled-current-launch-a.json` records the first exact ADR-0037 release:
run `r05a-inverse-flow-sampled-current-canary-20260715a`, task `27928_0`, and
release commit `223667c91b05be9ab403e4d92d0cd1a96b45246f`. A shell token parser
falsely rejected the valid held Slurm record before any receipt, CPU validator,
GPU release, allocation, checkpoint load, pi0.5 call, teacher search, or
simulator action. The exact task was cancelled with zero runtime. This is
apparatus-inconclusive; the run ID is consumed and cannot support a transport
or efficacy claim.

`ift00a-attempt-a.json` records the first allocation attempt and its CPU Slurm
diagnostic. Attempt `20260715a` stopped in synthetic focused tests before pi0.5
startup: 17/17 inverse-control tests passed, while one sampler test imposed an
extra platform-sensitive `1e-5` equality assertion despite all four frozen
fidelity gates passing. Its SHA-256 is
`bc1707688a16c54a8facfedfae874d17c9c123b4c57f5f18260d493cbd2cc998`.
No real-sampler transport or simulator outcome exists.

`ift00a-attempt-b.json` records the earlier real-pi0.5 retry B at jobs
`27726_0`/`27727`. Two finite deterministic 128-update searches ran, but host
memory finalization failed and no accepted result was published. The payload is
apparatus diagnostic only; it neither proves infeasibility nor authorizes a
solver change, IFT-01, or training.

`ift00a-sampled-current-launch-b.json` records exact release commit
`06b365b5899c2cb31db12187350cce48a3a0ea20`, completed GPU task `27962_0`,
failed CPU publisher `27963`, every immutable receipt/log/raw-output digest,
the diagnostic deterministic finite-nonconvergence values, and the absent
publication. The CPU wrapper compared parent `JobIDRaw=27962` to exact task
`27962_0`, so Python publication never ran. ADR-0039 consumes the identity and
permits only the exact-task accounting/provenance repair plus a zero-GPU live
regression before any separately reviewed H100 release.
