# 0019 — Freeze R04B exact resume parity as the final apparatus gate

Status: accepted on 2026-07-14 before the R04B allocation smoke, before any
perturbation-label collection, learned-probe training, guidance outcome, or R04
claim.

## Context

The research question is whether a learned scalar continuation-clearance probe
provides a useful steering direction for the frozen VLA. The decisive outcome
is paired simulator Safe-Progress Success against frozen, equal-norm random,
strong analytic, privileged-oracle, and direct-planner references. Resume
plumbing is not that outcome.

It is nevertheless the last necessary apparatus check. A probe trained on a
synthetic or pre-edit feature/label pair would not estimate the clearance of
the continuation actually produced after guidance. R04A exposed real eager
states but did not resume or edit them.

Independent review of the first R04B implementation found three defects before
submission: reset/settle was mislabeled as no simulator execution, zero-edit
validation compared only final actions rather than the post-edit feature, and
the modified sampler lacked a current-commit ordinary-baseline regression.
Those defects are closed in this contract.

## Decision

Freeze `configs/experiments/r04_resume_parity.json` at SHA-256
`a0edb7edac86d2abd887218002d14e28e072cd472d631797d18d540f5140be33`.
The only allowed run is one apparatus case, with the following 32 policy calls:

1. one ordinary compiled no-trace call with explicit paired float32 noise;
2. at each eager step 1--5, two source calls, two zero-edit resumes, and two
   nonzero-edit resumes; and
3. one identical ordinary compiled call after the 30 eager/resume calls.

The distinct `latent_resume_edit` mode consumes the absolute saved float32
latent and the captured float32 time. The captured time is authoritative; it
is never reconstructed as `1-step/10`. A direct model-coordinate edit is
applied once, the velocity is recomputed at the edited latent, and ordinary
Euler integration continues. Unedited coordinates use a byte-preserving
selection so a zero edit cannot change the sign bit of `-0.0`.

For every zero resume, require exact dtype, shape, and contiguous-array bytes
for all of:

- post-edit latent versus source latent;
- recomputed velocity versus source velocity;
- normalized predicted-clean feature versus source predicted-clean feature;
- inverse-transformed predicted-clean feature versus its source counterpart;
- normalized final action versus the source final; and
- physical `10 x 7` final action versus the source final.

Duplicate source, zero-resume, nonzero-resume, and compiled calls must also be
byte-exact. The one-coordinate nonzero edit is only an implementation sentinel;
it is never perturbation support, a causal dose, or scientific evidence.

The current eager observation, noise, five trace hashes, and physical final
must equal the independently validated R04A golden artifact. The two current
compiled calls must be byte-identical to each other and must remain within the
unchanged ADR-0010/0011 physical compiled/eager bounds. The ordinary policy API
does not expose the normalized compiled final, so R02 remains the authority for
that diagnostic; R04B makes no stronger claim.

The apparatus performs one simulator reset plus 20 dummy settle-control steps
to obtain the observation. It executes zero sampled-policy action steps and
zero efficacy rollouts. The final artifact records that boundary explicitly,
is validated before atomic publication, and must be checked again by a
separate CPU Slurm job bound to all three source Slurm identifiers.

## Stop and transition

- Submit exactly one R04B smoke after a clean reviewed commit and live
  preflight. Do not tune an equality rule or tolerance after observing it.
- A failed exact resume or baseline regression is retained as a failed
  apparatus result and blocks labels and training.
- A pass establishes only that exact continuation labels can be collected. It
  does not make R04 passing and does not show that a probe predicts clearance
  or steers safely.
- R04B is the final apparatus-only subgate. After it passes, work moves to the
  independent source-state estimand and the direct prediction, gradient-causal,
  and paired Safe-Progress Success experiments. Do not add unrelated sampler
  variants before those experiments.

## Consequences

The project remains focused on the learned-probe hypothesis while preventing a
feature/label mismatch or a changed baseline from masquerading as probe
efficacy. Claim-bearing work still requires independent source groups,
support-matched perturbations, powered boundary/false-safe coverage, strong
matched controls, and runtime reporting.
