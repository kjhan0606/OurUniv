# R2/5 — bounded same-field count+FP+TF equilibration test

The all-method factor has a correct same-IC derivative, but the two old
count+FP chains (64 warmup/64 retained each) drift badly. The new TF-only
source groups are mostly real `Ngal=1` groups, and cross-method source
comparison does not independently calibrate their selection/velocity law.
This bundle tests **sampler equilibration only** on the explicitly partial
N128/384 target. Do not promote its parameter values as calibration.

Start both chains from the two archived count+FP terminal states, with no
selection by TF score or LG appearance. Include all57,238 eligible galaxy
counts, original conditional FP group marks and6,745 disjoint TF-only
training marks from the *same* PMWD-evolved current field. Preserve the old
seven white tracer-nuisance coordinates. Add exactly two provisional standard-
normal white nuisances: TF selected-density exponent `exp(0.5*w)` and
singleton-dominated redshift width `150*exp(0.7*w)` km/s. They expose model
sensitivity, not calibrated source knowledge. The shared TF modulus offset
remains the existing FP-anchor parameter, not a second independent zero.
TF-group redshifts enter only inside the conditional TF mark ratio.

Use 192 warmup and128 retained HMC transitions per chain, with the existing
4–8-step randomized trajectory and target acceptance0.8. Save scalar traces,
accepted terminal white states and one blocked density map per endpoint, not
128 full maps. Inspect between-chain and split-chain drift/ESS/Rhat before any
extension. Preserve finite accepted states and explicitly count rejected
nonfinite proposal energies. One H100/4CPU/30GiB/3h allocation with a bounded
application deadline; the previous HMC process peaked10.16GiB and the
all-method derivative control5.51GiB, so30GiB exceeds their conservative
sum by~90%. GPU-memory use is separately watched by Slurm exit/status.

Q-GOAL: directly tests whether the new all-sky CF4 distance information and
galaxy counts can be sampled as a coherent current-field target. A numerical
pass alone does not deliver R2; selected-group inclusion, source covariance,
tracer response and heldout prediction remain. Q-LEAN: one predeclared pair
of starts, one target, no MAP-selected seed, large map stream or parameter
search. If nonstationarity remains, diagnose mass/temperature geometry
instead of blindly extending the same HMC kernel.

MW/M31/M33 remain R3 latent identities from each NEW state. No native truth
labels or fixed known components seed/select candidates; role ambiguity and
unresolved M33 must survive, and their observables must constrain the same
evolved field. This N128/3-cMpc/h calculation cannot separate those members.

## Execution and result

H100 **406491 was cancelled by the driver after2m18s**, while the source
overlap audit found2,761 TF training groups securely linked to counted
2M++ points. This audit does not prove double counting; it exposes an
unverified count-conditioned group-cz/selection relation in the partial
target. Preserve the job's partial result and logs. No retained samples or
equilibration verdict were produced. Revisit this HMC design only after the
conditional observation relationship is specified or convincingly bounded.
