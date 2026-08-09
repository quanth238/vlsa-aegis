# E05 ellipsoid candidate-support prerequisite

Per the user direction, this diagnostic uses only the registered ellipsoid
model: seven L5--L7 robot ellipsoids against the unchanged released-AEGIS
obstacle MVEE. It does not calculate mesh distance and does not train a neural
network yet.

The immutable successful simulator-oracle trajectory reconstructs recoverable
states 184--192. At each state, the query is the recorded released-AEGIS
action. The frozen candidate family contains central perturbations, local
0.25/0.5 offsets, stop, reverse, the global {-1,0,1} lattice, and the action
executed by the successful oracle. Each candidate is run through one complete
cloned OSC `env.step`; all seven support gaps are minimized over every internal
MuJoCo substep.

The prerequisite passes only if test states 187 and 190 each contain at least
one candidate with all seven margins nonnegative. If either safe set is empty,
an MLP trained on these ellipsoid labels cannot produce a valid safe action
inside this candidate family; training must stop. Raw contact is retained only
as a diagnostic and never replaces the ellipsoid decision.
