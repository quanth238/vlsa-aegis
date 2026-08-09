# E05 paired directional contact model

Job `37441` ruled out a sign, normalization, or XYZ-indexing error. This
experiment changes only the missing supervision: the contact classifier is
trained to order already-executed opposite action perturbations.

## Data and split

The immutable job-`37416` dataset already contains cloned-OSC executions of
symmetric central and local perturbations. Opposite offsets are paired only
within a state, with L2 radius at most `0.45`. Complete state groups remain:

- train: 184, 185, 186, 188, 189, 191;
- validation: 192;
- test: 187, 190.

No test-state record enters normalization, training, early stopping, or model
selection. A pair is informative when its complete OSC traces have different
raw L5--L7 contact-event counts. Ties are reported and excluded from the
directional loss.

## Objective

The network and per-link weighted BCE remain unchanged. For each informative
pair, lower contact count defines `u_better`. With
`R=max_l sigmoid(z_l)`, add

\[
L_{dir}=\max(0,0.05-[R(u_{worse})-R(u_{better})]),
\]

and optimize `L_BCE + 10 L_dir`. This directly trains the ordering required by
the action gradient while preserving contact classification.

## Unseen-state matched-random test

At states 187 and 190, evaluate the revised negative-risk gradient at the
small radius `0.1`. Re-execute the same 256 job-`37427` random directions at
the same radius to add penetration measurements while requiring their action,
contact-count, and next-state receipts to reproduce exactly. Rank each action
by raw safety, contact count, then summed MuJoCo penetration.

The add-one empirical value is the fraction of random directions at least as
good as the learned direction. A mechanism GO requires validation informative
pair accuracy at least 0.7 and empirical value at most 0.05 on both unseen
states. This is deliberately a small-radius directional test; it does not
claim collision avoidance or task completion.

Closed-loop E05 is forbidden regardless of outcome.

