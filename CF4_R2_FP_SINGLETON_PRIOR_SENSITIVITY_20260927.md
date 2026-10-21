# R2/5 — overlap-free FP singleton radial-prior sensitivity

The CF4+galaxy-conditioned z=0 density/velocity posterior remains **NO-GO**.
This is a train-only, one-state observation-law diagnostic, not a fit or an
independent test. No email was sent.

The graph-closed v5 split contains 3,535 training FP source groups with exactly
one FP row, no counted 2M++ member and no raw distance anchor. There are 441
sky-heldout groups with the same geometry, but their marks were **not read**.
The source Q257 geometry was bound to one archived, unconditioned N128/384 PM
state with native origin zero. The conditional FP mark log ratio was evaluated
under three radial weights: selected `d²ρ`, geometrical `d²`, and `1/d`
(flat in log distance). The redshift kernel, FP source PDF and field state were
identical across the three evaluations. No count likelihood, group-selection
probability, source mock, nuisance fit or sampler was included.

Typed-H100 Slurm **406598 COMPLETED/exit0** in 13 seconds, with 2.17 GiB
peak host memory under an 8 GiB request. The preserved
[result](/gpfs/kjhan/CF4/z0_density/r2_fp_singleton_prior_sensitivity_v1/result.json)
records input hashes and all scores. The 3,535 training-group log-ratio sums
were 15.806 (`d²ρ`), 17.183 (`d²`), and 13.354 (`1/d`). Relative to `d²ρ`,
the absolute per-group score difference has median 0.00355/0.00368 and 99th
percentile 0.0670/0.0694 for the two alternatives. The signed *sum*
differences are -1.377 and +2.451. These are one-state score changes, **not**
Bayes factors, uncertainty estimates or evidence that the selected FP law is
calibrated. They cannot be compared to scores from a different archived field
as an information fraction.

Driver decision: do not replace the full grouped CF4/FP model with only these
easy singleton rows. Their modest one-state modulation does not establish
enough constraining power, and omitting linked groups would discard relevant
CF4 information. The next substantive observation-law task is a source-bound
selected-group/marked-point model that owns each 2M++ count and its linked FP
mark **once**, with explicit treatment of FP inclusion, shared covariance and
unmatched/multi-member groups. Its assumptions must be stress-tested with
source-backed mocks and the untouched sky split before any R2 posterior claim.
The existing official selected SDSS FP mock checks individual source PDFs but
does not contain the parent or recovered Tempel membership needed to calibrate
real-group inclusion by itself. Do not import simulation truth identities into
new-field candidate selection.

Q-GOAL: tests whether a less overlap-dependent FP subset might support the
first science delivery without uncalibrated double counting; the answer is
insufficient for promotion. Q-LEAN: one frozen state, one train-only score
calculation, no new simulation, model fit, broad calibration framework or
external review. MW/M31/M33 remain latent roles to identify from each **new**
field, with M33 allowed to remain unresolved; their observables must constrain
that same field, and native truth labels may only evaluate/calibrate.
