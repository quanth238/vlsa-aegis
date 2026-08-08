# Paired distal-prefix recovery v4 result

H100 producer job `37190` ran on worker-1 from clean commit
`40cde909e64b6a880892057d37c6f46d2c945c1a`. It reproduced the paired AEGIS
prefix through action 185 with exact clone agreement, no contact, and no CAR.
At action 186 the first distal `8 mm` intervention had no feasible candidate,
so the method failed closed before live recovery or unsafe execution.

The prior action ended with a minimum exact distal gap of `13.127 mm`. On the
next full OSC transition, the nominal minimum L5 part gaps were
`-3.442/3.656 mm`. All 81 registered candidates were raw-contact-free, but
none met all seven margins. Even the best candidate reached only
`-0.907 mm` for L5 part 1 and `7.400 mm` for L5 part 2. The QP was primal
infeasible (`7.276 ms`). This shows that an `8 mm` one-step trigger is inside
the discrete OSC braking distance for this paired state; it does not show a
solver-runtime or ellipsoid-count failure.

The first validator incorrectly required at least one live-policy query even
for a valid pre-recovery method failure, causing Slurm exit 1 after the atomic
scientific result was complete. The result remained immutable and was then
submitted for validator-only recheck under the corrected contract.

Validator-only H100 job `37191` completed from clean validator commit
`39a12f9fa86594807953981301d831c11dcb2466` and passed all 19 tests plus every
result, producer-commit, target-vector, exact-box, and Table-1 immutability
check. Its receipt SHA-256 is
`d6fcf39d173435c54bd784f7b70ebf6521a36a3742c13e74142564e24358dd43`.

Producer result/preflight/MP4/JPG SHA-256 values are
`26522e4ee8ba7d78d7ecfa4021affbb309a25dc849a297b540e4f77c3f2904dd`,
`531f0e58b9b6706b7c695942720295c7100d844ea8ad5863515590b669b4fb65`,
`d4319a38bd5c6c3d23766b8eeca36fcbdb28a8168c30c88c40a3c361ab7e2d43`,
and `99b92e22d775a0842d525bcaaa1e063f8ee7a534e5f14d70cc6b9b2fff4cdf7b`.
Payload SHA-256 is
`e9859884e6e5e7e8d60dea4d6c4de99d71e48158297d2be84f1ac1bae5540793`.
