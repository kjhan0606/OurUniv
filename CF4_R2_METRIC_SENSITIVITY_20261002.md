# R2 matched-metric sensitivity — 2026-10-02

## Stage and question

R2 (actual-data present z=0 field inference) remains open. N256 gives a
1.5 cMpc/h numerical grid in a 384 cMpc/h box; it is not the final <=0.3
cMpc/h LG field. This bounded experiment asks whether changing the sampler's
inverse-Laplacian metric from600 to6000 caused the low acceptance in jobs
409763/409764, holding the initial state, exact target, step size and four-step
trajectory fixed. It does not test posterior convergence or astrophysical
structure.

## Evidence entering the control

The unchanged exact GL2 target and Gaussian prior were used in all three runs.
`step_size=.08`, nuisance inverse mass `1e-5`.

| Jobs | Fundamental-mass parameter | Integrations/proposal | A/B acceptance | Interpretation |
| --- | ---: | ---: | ---: | --- |
| 409590/409591 | 6000 | 1 | 8/8, 5/8; mean probability .907/.886 | Eight short transitions, not stationary |
| 409763/409764 | 600 | 4 | 0/1, 0/1; probability .00148, 4.90e-9 | Two valid rejections; no accepted field movement |
| Existing force-pair exact-GL2 controls | 6000 | 8 | .480/.551 | Different saved states/momenta; not a matched metric control |

The first two rows do not isolate metric because trajectory length also changed.
The eight-step force-pair reference is not a same-state/same-momentum
substitute. Fable's usable read-only audit considered the code's transition
mechanics and the observed endpoint checks valid; the driver independently
verified the four force calls in each rejected proposal and the reported
energies. It also identified no astrophysical discovery; the driver agrees.

The 409763/409764 JSON trace hardcoded `integration_steps=1`; this did not alter
the HMC calls, which used four integrations (`force_evaluations=4`) and matched
their declared settings. The trace writer now records the requested count.
The runner previously ignored its checkpointed RNG state when continuing; it
now restores that state by default. Setting `CF4_R2_CHAIN_SEED` deliberately
overrides the restored stream and records the explicit seed, enabling exact
matched replay. Nine focused HMC tests pass, and Python compilation, Slurm
script syntax and `git diff --check` pass. The old result files are preserved.

## Next bounded control

Use the same accepted state and momentum as each 600/four-step proposal, but
restore metric6000:

- Chain A input: `/gpfs/kjhan/CF4/z0_density/r2_n256_gl2_metric600_a_v1/chain_a_accepted_checkpoint.npz`, seed `2026100201`.
- Chain B input: `/gpfs/kjhan/CF4/z0_density/r2_n256_gl2_metric600_b_v1/chain_b_accepted_checkpoint.npz`, seed `2026100202`.
- Exact GL2 target; fundamental-mass parameter6000; step.08; four integrations;
  one proposal; no warmup/adaptation; unchanged nuisance metric and MH test.
- H100 x2 independent Slurm jobs;4 CPUs,24GiB host and4h wall cap each.
  This is about2.8 GPU-hours based on the previous one-proposal runs, with the
  prior measured device peak30.56GiB under the69.81GiB device limit. Host RSS
  in those jobs was12.2/15.7GiB.

This paired contrast isolates the metric at the fixed tested trajectory
settings. If the matched high-mass proposals have small energy errors and
move, reject600 for this configuration; if they also fail, do not attribute
failure to mass alone. Either way, no long chain or wider tuning sweep starts
automatically. If this contrast cannot distinguish a viable kernel, close the
current metric tuning path and plan a sampler redesign. No heldout values,
field selection, gravity evolution, training, final map or posterior claim.

Submitted via Slurm on2026-10-02: job410010 (A) and410011 (B), both on
H100/syn08, each requesting1GPU,4CPUs,24GiB host and4h wall. At first state
check both were RUNNING. Output directories are
`/gpfs/kjhan/CF4/z0_density/r2_n256_gl2_metric6000_control_a_v1/` and
`/gpfs/kjhan/CF4/z0_density/r2_n256_gl2_metric6000_control_b_v1/`; each uses
the corresponding accepted checkpoint from the600/four-step pilot and the
same explicit replay seed2026100201/2026100202.

Q-GOAL: this is a narrow R2 sampler-mechanics check; it is necessary only to
determine whether the current exact-target transition can support later
present-field inference. It does not deliver the z=0 density/velocity map.
Q-LEAN: reuse two preserved accepted states, exact oracle and already measured
600/four-step proposals; add only the matched6000/four-step control, with no
extra trajectory sweep or instrumentation ladder.

MW/M31 remain role-ambiguous and M33 unresolved. No candidate is selected by
native truth identity; subsequent MW/M31/M33 observables must constrain these
latent roles on the same NEW evolved field. R3 identification, R4 precision
evolution and R5 zoom IC construction have not been advanced by this bundle.
