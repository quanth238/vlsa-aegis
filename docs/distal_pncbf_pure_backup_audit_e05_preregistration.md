# Pure backup-policy representative-state audit

The validated jobs 39422/39424 establish exact finite-horizon values for an
augmented deterministic policy that includes a frozen pi0.5 proposal ledger.
That is legitimate policy evaluation on the augmented state, but it is not the
pure ledger-independent backup policy required by the new system
specification.

This gate changes exactly that component. The action ledger from job 39422 is
used only to reconstruct three immutable physical/controller states: the first
warning state (187), a later closest-clearance warning state (222), and the
terminal state (288). It is forbidden from the backup-policy action and score.

At each state, the backup evaluates a fixed candidate family:

- zero-motion hold;
- positive and negative world XYZ actions;
- positive and negative state-derived obstacle normal, up-tangent and
  side-tangent actions;
- amplitudes 0.5 and 1.0 normalized action units;
- zero rotation and neutral gripper command.

Each candidate executes one action followed by a 25-action zero-motion hold.
All 26 actions are measured at every internal MuJoCo substep. A candidate is
eligible only when all seven L5--L7 proxy gaps remain at least +1 mm, protected
MuJoCo contact is zero, and paper CAR is at most 1 mm. The policy selects the
largest verified future clearance with fixed semantic tie breaking.

Pass requires at all three states:

- exact replay determinism within `1e-12`;
- candidate-order invariance within `1e-12`;
- no mutation of the source state during cloned evaluation;
- construction without VLA chunk or proposal-ledger arguments;
- at least one eligible candidate;
- a selected candidate whose registered hold tail passes the complete gate.

This is a representative-state backup-mechanism audit, not a complete backup
policy, neural model, QP, CBF, invariant-set, task-completion, or generalization
claim. Passing permits the next experiment to roll this pure backup over a
broader state population before learning direct action risk.
