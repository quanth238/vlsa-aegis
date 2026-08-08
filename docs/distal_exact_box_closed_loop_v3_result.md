# Distal-only 8 mm closed-loop v3 result

Clean H100 job `37189` completed all 300 registered actions on worker-1 in
`1m58s` from commit `c28bb7ac9cc8e496d40a57940de9e7c24ff1b0b9`; all
47 tests and independent validation passed. Every execution matched its exact
clone, and there was no robot contact, protected contact, CAR, or obstacle
motion. The task did not complete, so `primary_problem_solved=false`.

The seven-row filter never activated: minimum distal clearance was
`88.478 mm`. This fresh live pi0.5 trajectory therefore did not enter the
primary Table-1 failure region and cannot test collision recovery, despite
being safe. The original successful AEGIS failure prefix must be retained
until the first distal intervention before evaluating live recovery.

Result/validation/preflight/MP4/JPG SHA-256 values are
`e0ff795b27f23a06ceb76946ced1985955a2b9d388a549dd5354381774d909df`,
`33b0940e41fcba5768add43385b11734b81f9c27f83bb813a6f5d15934b08351`,
`cb9e5c15ffd9d6b051d2cddfdb3cbccd1ff026a90a7d1ef01f9f96acab18189a`,
`b3b0e753646f708791d094f11e9f4fbeeb73fa6e07b40dcea3d88e163031230b`,
and `f375477034bd0b03d178a885e32170b7fdf0b17c3376adaf37c732018758ead8`.
Payload SHA-256 is
`63414c2bba374521f8be68cfb2db663a06fb7cfb55b08da42421ee2339e2a9aa`.
