# Archived R05A sampled-current launch B

This record preserves the detailed terminal history of the second
ADR-0037-released sampled-current IFT-00A canary. It is provenance, not an
instruction to republish, resume, reuse, or resubmit anything.

## Registered identity

- Run: `r05a-inverse-flow-sampled-current-canary-20260715b`
- Release commit: `06b365b5899c2cb31db12187350cce48a3a0ea20`
- Accepted implementation: `7d15c2c7921d9d1201638cb68ee48cc06a94ded1`
- Source-contract SHA-256:
  `b9a9e253c9a5e27815018c13c839805c97752545189085caac016db4ee81e1e6`
- Submission-receipt SHA-256:
  `d6b6effb9761770858e591f1ad6f793c5b931c1f3aab25e2ce83e3dcbe6b437b`
- Case: `crfs-1069f29a8d76463a`
- Source node: `worker-1`
- GPU task: `27962_0`
- CPU `afterany` publisher: `27963`

## Terminal accounting

GPU task `27962_0` was `COMPLETED`, exit `0:0`, on worker-1. It started at
`2026-07-15T13:48:21Z`, ended at `13:51:30Z`, and used `00:03:09` with
`billing=8,cpu=8,gres/gpu=1,mem=64G,node=1`.

CPU job `27963` was `FAILED`, exit `3:0`, on worker-1. It started one second
after the GPU task ended, ran `00:00:31`, and used
`billing=2,cpu=2,mem=8G,node=1`. The exact log says:
`source H100 task did not complete successfully: state=missing`.

The CPU wrapper queried parent `27962` with `JobIDRaw` but compared the result
to task identity `27962_0`. Exact accounting shows that VinUni reports display
`JobID=27962_0` and raw allocation `JobIDRaw=27962` for this singleton task.
Waiting longer could not make the impossible identity comparison pass.

## Raw GPU diagnostic

The raw payload SHA-256 is
`4c85603c62446d74259820dc7f86451ffe53723b79e0c866d2e38d7a026622cd`.
It is 7,014,377 bytes. Exact source and target pairing passed, including the
float32 path budget `3.6398398876190186`, scale-only displacement conversion,
and bitwise equality outside the first-five XYZ mask.

Both teacher calls ran 128 updates, stayed finite, and returned status code 3.
Their registered schedules share SHA-256
`714ac52e791248e7590196b5dfe47cc5f4a4966dce0a0237092a9cd939b72fc7`;
their returned actions share SHA-256
`23c855c3109ffcb78dce3c2671cae52a3fafdd820d2c4ab1d0c248da51fe48c1`.
The schedule used the exact path budget but missed target fidelity: XYZ maximum
absolute error `0.8631911873817444` and RMS `0.3312218487262726`. This is
finite nonconvergence, not infeasibility. The failed schedule was not applied,
and returned actions remained frozen.

All allocation suites passed: 17 inverse-control, 8 sampler, 10 policy, and 12
canary tests, for 47/47 with zero skips. Host sampled-current telemetry observed
1,537 samples and a 16,070,856,704-byte high-water with zero `max`, `oom`, or
`oom_kill` deltas. It is a sampled lower bound, not an exact peak. Periodic GPU
samples observed 8,504 MiB process and 8,513 MiB device high-water; these are
also not continuous peaks.

No policy-generated or teacher-generated action was executed in the simulator,
and no efficacy rollout occurred.

## Publication boundary

The Python publisher never ran. No `.results.candidate.json` or `results.json`
exists. The CPU failure receipt SHA-256 is
`3295cf9458623d3a9a275a2ee70fa64dc5fc5e1762e8640dd0c4ae8813b0cbdf`;
it records `passed=false`, `published=false`, and no result digest. Therefore
the raw payload cannot be promoted after observing its outcome.

Allowed interpretation: the raw GPU payload diagnostically reproduced
deterministic finite nonconvergence, but publication failed, so no accepted
IFT-00A transport conclusion exists. ADR-0039 permits only the exact-task
accounting/provenance repair and a zero-GPU source-plus-`afterany` regression
before any separately reviewed H100 release. IFT-01 and training remain
blocked.

## Accounting repair validation

The preregistered shell-only regression later ran from clean commit
`5e595a366cb95d50ee86776f5de701627bf09669`. Singleton source `27975_0` and
CPU `afterany` validator `27976` both completed `0:0` on worker-1. The
production helper observed exact display task `27975_0`, not parent
`JobIDRaw=27975`, and the validator published result SHA-256
`0de4b8b736bd750a82e7439cf737b9d16e248d67ee0d2f741d82f09d9bcd6b74`.
All nine bound files matched; each job requested one CPU, 256 MiB, and zero
GPUs. No Python, model, simulator, search, or training ran. This closes the
publication-path apparatus defect only and grants no H100 or scientific
authority.
