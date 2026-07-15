# R05A evidence

`ift00-synthetic.json` records the passed local IFT-00 implementation gate.
It is deliberately labeled synthetic: neither pi0.5 nor SafeLIBERO was run, so
it cannot support an efficacy, safety, latency, generalization, or novelty
claim.

The useful sentinel is narrow. On a deterministic nonlinear field, repeating
the final-action delta as the old constant residual missed the terminal target
by `0.048832`, while the constrained time-dependent solver reached the fixed
synthetic fidelity gate with path `0.067193 <= B=0.1`. Duplicate solves were
bit-identical. A separate coupled-field test proved that the solver can discover
a necessary control coordinate initialized at zero even when called beneath
the sampler's outer `torch.no_grad()` context.

IFT-00 also tests exact iterative float32 sampler times, exact target pairing
outside first-five XYZ (including padded dimensions and signed zero), the two
physical fidelity gates, hard timing/mask/path constraints, explicit
nonconvergence/nonfinite statuses, graph-free results, reverse-order controls,
and absence of parameter-gradient contamination.

The next evidence must come from the one-case allocation integration canary.
Until that passes, no real inverse-flow action or research result exists.
