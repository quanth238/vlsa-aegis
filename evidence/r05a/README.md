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

`exact-array-task-afterany-regression-a.json` records the terminal shell-only
repair check at source task `27975_0` and validator `27976`. Both completed
`0:0` on worker-1; exact display `JobID=27975_0` was observed, all nine bound
files matched commit `5e595a3`, and the published result SHA-256 is
`0de4b8b736bd750a82e7439cf737b9d16e248d67ee0d2f741d82f09d9bcd6b74`.
Each job requested one CPU, 256 MiB, and zero GPUs. This is apparatus evidence
only and cannot authorize an H100 retry, IFT-01, a method claim, or training.

`cfs00a-same-budget-launch-a.json` records exact CFS release `77bf9f6`, GPU
task `28021_0`, CPU `afterany` validator `28022`, all immutable receipt and log
digests, and the terminal allocation-contract failure. The runner rejected the
canonical OpenPI virtual-environment Python path because it is an intentional
symlink to an existing executable. The GPU task failed before allocation tests,
telemetry, model loading, or Arms A/B/C; the CPU job correctly refused
publication. ADR-0042 consumes the identity and classifies the run as
apparatus-inconclusive. It contains no transport, efficacy, infeasibility, or
training evidence.

`runtime-identity-regression-20260716a.json` records shell-only, zero-GPU
worker-1 job `28043`. It completed `0:0` and matched both production Python
launcher paths, direct link targets, fully resolved executable files, and
resolved-binary SHA-256 values without invoking either interpreter. It
requested one CPU and 256 MiB; Slurm reported two allocated logical CPUs and
256 MiB, with no GPU TRES. This validates only the standalone identity
validator. The full CFS integration, transport hypothesis, simulator efficacy,
IFT-01, and probe/MLP training remain untested or unauthorized.

`cfs00a-same-budget-launch-b.json` records exact release `12a7da6`, completed
GPU task `28048_0`, completed CPU publisher `28049`, every immutable receipt,
log, telemetry, raw-payload, and published-result digest, and the precise
partial scientific boundary. Real pi0.5 computed the zero-control baseline,
35-by-75 autograd Jacobian, and all three directions by two epsilons, but the
registered finite-difference gate failed before FISTA. No Arm-B candidate,
nonlinear replay, Arm C, or four-gate result exists; Arm A was not duplicated
or independently validated. The allocation used about 15.0 GiB sampled host
RAM and 8.5 GiB sampled device memory with zero OOM events, and publication
passed. The run is therefore apparatus-inconclusive because its generic
exception discarded the failed numeric comparisons. ADR-0047 permits only a
new immutable diagnostic that records those values and still stops before
FISTA; it does not authorize a method, efficacy, infeasibility, or training
claim.

`cfs00a-fd-diagnostic-20260716a.json` records exact release `3d44b2c`,
completed GPU task `28212_0`, completed CPU publisher `28213`, all immutable
artifact and log hashes, 99 allocation tests with zero skips, sampled host and
device high-waters, zero OOM events, and zero simulator-generated actions. The
adapter reached its typed one-key `__crfs_terminal__` response, after which the
unchanged WebSocket server appended standard `server_timing`. The paired
client rejected that two-key transport mapping before extracting or
serializing the terminal diagnostic. No numeric Jacobian or finite-difference
values were preserved, and no FISTA, candidate, replay, Arm C, or four-gate
result exists. ADR-0048 classifies the run as apparatus-inconclusive and
permits only strict client-side timing normalization followed by review and a
separately released immutable diagnostic retry. It provides no teacher-
transport, efficacy, infeasibility, or training evidence.

`cfs00a-fd-diagnostic-20260716b.json` records the corrected transport run at
exact release `1e2d36d`, completed GPU task `28222_0`, and completed CPU
publisher `28223`. The client normalization succeeded, so the immutable
payload preserves the full float32 `35 x 75` autograd Jacobian and all six
registered finite-difference comparisons. Every comparison failed: relative
errors were `0.626`–`0.993` against the frozen `0.10` limit. More importantly,
halving the fixed perturbation changed the three central-derivative vectors by
`0.819`, `0.937`, and `1.187` relative L2, so the actual forward response did
not supply a stable local derivative at those scales. The mixed-precision
`bfloat16` cast is a plausible source mechanism, but this run does not isolate
it from other numerical effects. All 99 allocation tests passed, host/GPU use
was about 15.0 GiB/8.5 GiB with no OOM, and zero generated actions entered the
simulator. ADR-0049 retires only the autograd-linearized CFS-00A submethod and
keeps the broader actual-forward flow-transport hypothesis open. No FISTA,
candidate, replay, Arm C, efficacy result, probe, or MLP exists.
