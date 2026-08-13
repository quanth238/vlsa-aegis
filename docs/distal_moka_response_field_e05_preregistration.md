# E05 Moka controller-conditioned response-field gate

This is a one-task feasibility pilot. It tests whether an MLP can reproduce
the useful physical action response already demonstrated by exact
counterfactual OSC rollouts without repeating the rejected binary classifier,
joint-trajectory predictor, or monotone scalar-weight model.

The frozen case is `vlsa-t1-goal-ii-t0-e05`. Completed Table 1 actions are
read-only. The frozen VLA, OSC, seven L5--L7 robot ellipsoids, and action
bounds are unchanged. The released single obstacle MVEE is excluded from this
pilot. The Moka pot is represented by all 15 compiled collision boxes, each
with its own tight enclosing ellipsoid; raw protected MuJoCo contact and exact
box-union overlap remain separate physical diagnostics.

At each E05 warning state, 32 smooth five-action directions are executed with
paired `+0.05/-0.05` corrections through the same 20-action archived
continuation. For every future action and each of the seven robot ellipsoids,
the label is the minimum support gap over the 15 Moka primitives. The network
is conditioned on the complete cloned OSC state, the nominal 20-action
continuation, time, and robot-row identity. It predicts the quantitative
witness value and one 15D action-response row. Training uses the paired
directional equation

\[
v^\top \hat g_j(x,A)
\approx
\frac{h_j(A+\epsilon v)-h_j(A-\epsilon v)}{2\epsilon}.
\]

Whole state groups are split before training:

- train: actions 178--182;
- validation: actions 183--184;
- untouched test: actions 185--186.

At both test states, compare the frozen learned direction with fixed
multi-primitive repulsion, a privileged exact local-secant ceiling, and 64
matched random directions at correction radii `0.1/0.25/0.5`. The learning
gate requires learned-versus-exact direction cosine at least `0.8`, held-out
directional sign accuracy at least `0.75`, exact proxy gain at least `0.5 mm`,
gain over fixed repulsion at least `0.1 mm`, matched-random `p<=0.05`, and no
new raw protected contacts at both test states.

Passing supports only this statement:

> A controller-conditioned multi-witness MLP can reproduce a useful local
> Moka avoidance response across nearby held-out states of one task.

It does not establish episode generalization, live VLA task completion,
online deployment, certified safety, a CBF, or a deployable perception system.
If it fails, do not add a QP or run closed loop; audit value versus response
prediction and the test-state support first.
