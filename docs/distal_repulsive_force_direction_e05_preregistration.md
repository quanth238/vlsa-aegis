# E05 learned repulsive-force direction gate

This is a deliberately local mechanism experiment at the primary E05
failure.  It does not claim population generalization, real-time deployment,
or SafeLIBERO task completion.

The accepted seven L5--L7 ellipsoid slabs define future two-action margins
`h`.  Paired cloned-OSC rollouts identify the local pullback `G = dh / da`
for XYZ action changes.  A positive-weight monotone MLP learns a scalar
safety potential from quantitative exact rollout margins.  Its correction is

\[
u_{\rm learned} = \frac{\nabla_a f_\theta(h+G\,\delta a)}
{\|\nabla_a f_\theta(h+G\,\delta a)\|_2}.
\]

Because every network path from clearance to safety has nonnegative weight,
the learned gradient can reweight physical push-away directions but cannot
invent a direction that deliberately decreases every modeled clearance.

Whole simulator states are grouped: steps 182--183 train, step 184 validates
and selects the checkpoint, and step 185 is not evaluated until the model is
frozen and hashed.  Candidate supervision uses paired unit directions at
normalized-action radii 0.10 and 0.25.  The target is the exact two-action
soft minimum of the seven future ellipsoid margins, not a contact label.

At step 185, three directions are compared at exactly equal action norm:

1. the learned monotone-potential gradient;
2. the fixed analytical affine soft-min gradient;
3. 256 preregistered paired random unit directions.

At both radii the learned correction must improve exact clearance by at least
0.5 mm, exceed the fixed direction by at least 0.1 mm, and achieve an
add-one matched-random p-value no greater than 0.05.  It must also provide an
exactly safe candidate, produce no raw L5--L7 contact, and move the obstacle
by at most 1 mm.

Only after that direction gate passes may the controller execute a corrected
action, re-identify the local pullback from the measured next state, and
execute one more corrected action.  Both executed transitions must exactly
match their accepted clones.  Useful motion requires at least half the
end-effector progress of the unmodified two-action VLA prefix.

This test uses cloned OSC online and is therefore an oracle mechanism test.
If the learned direction passes, a later experiment may learn the pullback.
If fixed repulsion performs equally well or better, the MLP has not added
evidence beyond the physical oracle.
