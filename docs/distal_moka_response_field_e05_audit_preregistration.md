# Frozen E05 Moka response-model audit

This audit diagnoses the strict NO-GO from producer `39230`. It does not
train or modify the model, enlarge a correction, solve a QP, or execute a
corrected action.

The immutable model and dataset protocol are replayed from complete cloned
OSC states. The model payload hash, state split, action continuation,
directions, perturbation, and geometry must match the accepted producer.

The audit answers three questions in order:

1. **Fit:** Does the frozen MLP reproduce paired action responses on training
   and validation states, including eight directions excluded from each
   state's local ridge fit?
2. **Support:** Are actions 185--186 outside the train distribution in the
   dynamic state, controller memory, nominal action chunk, or obstacle
   geometry groups?
3. **Witnesses:** Do errors concentrate on near-active link/time/compiled-box
   witnesses, weak geometric-mode separation, or late horizons?

For every split, report response cosine, sign accuracy, RMSE, value error,
row-gradient cosine, and smooth-field direction cosine. State support is
measured groupwise using training-only means and scales. For each base
rollout, preserve the closest compiled Moka primitive for every one of the
`20 x 7` witnesses and report rows within 2 mm and 5 mm of the worst value.

Interpretation is diagnostic:

- weak train and validation response means objective/model underfit;
- train passes but validation/test fail with large state distance means
  coverage/generalization failure;
- errors isolated to near-active witnesses despite otherwise good fit means
  witness representation or weighting failure;
- a combination must be reported as a combination, not forced into one cause.

This audit cannot authorize retraining or control. Its output only selects the
next preregistered intervention.
