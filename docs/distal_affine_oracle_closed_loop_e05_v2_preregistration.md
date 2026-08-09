# Eight-constraint affine-oracle E05 closed-loop preregistration

## Registered correction

H100 job `37283` is retained as a validated scientific NO-GO for the frozen
seven-row closed-loop method. At action 15, the two-step nominal released-AEGIS
EE proxy margin was `-1.535 mm`, while all distal margins were at least
`90.866 mm`. The exact grid contained 201 all-eight-safe candidates, but the
seven distal affine envelopes had zero gradients because none of their
constraints was active. The seven-row QP therefore returned the nominal action
within `1.95e-9` action units, and the fresh EE veto correctly stopped it.

V2 changes one item only: the exact released-AEGIS EE margin becomes the
eighth affine lower envelope and eighth hard QP row. The accepted seven L5--L7
slabs, exact obstacle boxes, 512 actions, action bounds, padding, objective,
two-step cloned OSC transition, immutable Table 1 nominal sequence,
execute-first/replan schedule, and every exact execution gate remain fixed.
This is the direct correction for the observed missing constraint; no margin,
trust region, candidate order, fallback, or success criterion is relaxed.

## Decision

The QP must report exactly eight input rows at every intervention, and its
candidate must pass a fresh two-step clone with all eight margins nonnegative,
zero protected contact, and at most 0.1 mm within-step obstacle motion. Only
the first action executes and its next-state hash must match the clone.

Closed-loop GO still requires all executed substeps all-eight safe, zero raw
L5--L7 contact, zero paper CAR, exact clone fidelity, and native task success.
The result remains a privileged feasibility diagnostic, not learned
generalization, deployable control, or formal whole-body safety. Neural
training remains unauthorized.

V2 config SHA-256 is
`3ad64a16e75852873e7f5b837d7dc7eba85a82fd042ff5d2d261aabb3a3175f0`.

