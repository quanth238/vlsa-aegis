# Exact-contact ranker E05 feasibility diagnostic

This experiment asks whether offline cloned OSC rollouts contain enough signal
for a small state-conditioned MLP to distinguish safe from unsafe Cartesian
corrections near the E05 L5/L6 failure. It deliberately does not claim to
learn signed mesh clearance: the installed MuJoCo binding exposes exact
contacts during the transition, but not a continuous arbitrary-mesh distance.

The main environment follows the immutable, task-successful simulator-oracle
trajectory. At states 184--192, the query is the recorded released-AEGIS
action. The registered candidates are central perturbations, local 0.25/0.5
offsets, stop, reverse, the global {-1,0,1} lattice, and the task-valid action
that the successful oracle executed. Every candidate is passed once through a
cloned full OSC `env.step`; every internal MuJoCo substep is checked separately
for L5, L6, and L7 contact. Ellipsoid clearances remain diagnostic `D_opt` and
cannot replace the raw `D_sim` label.

Complete states are split before training: 184/185/186/188/189/191 train,
192 validation, and 187/190 test. The validation state sets a conservative
threshold immediately below its least-risk unsafe candidate. The frozen gates
are (1) zero false-safe candidates on both test states, (2) the least-change
predicted-safe candidate from each test state passes a fresh exact cloned OSC
transition, and separately (3) at least one bounded negative-risk-gradient
candidate passes the same exact test. Passing (1)--(2) supports only an MLP
Best-of-N ranking mechanism. Passing (3) additionally supports local gradient
steering. Neither result demonstrates closed-loop task completion or unseen
episode generalization.
