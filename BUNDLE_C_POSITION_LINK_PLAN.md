# Autonomous C: observations to the SAME fine-field position likelihood

2026-09-12. User grants autonomous in-goal continuation, replacing repeated
bundle approval waits. Driver still uses bounded experiments, Fable5 planning
advice, Slurm execution and recorded judgments. Source528af87 closes the
population location comparison347085, not a physical density posterior.

## Evidence and deliverable

Frozen21-coefficient q(W,A,T|F,O,Earchive) on64^3/.1875cMpc/h improves test
joint NLL11.2234->9.9064, with M333.8976->3.4396 and shared-cell3.7562.
Train/calibration/test observers1388/87/127 use voxel-disjoint supported slabs
in one TNG100 box. Alpha=1. Broad autonomous positional uncertainty remains;
cell size does not establish position accuracy. Freeze this fit, no retraining.

Deliver ONE observation-space operator and native heldout/actual-observation
field-sensitivity product. It marginalizes distance errors and evaluates
observational positions against probabilities inferred from the SAME total
field, with no true catalogue parents/candidate list passed into q. This is
the first connection to make; do not pretend it supplies the missing field
prior, observer-selection probability, masses/COM velocities or an LG map.
No repeated generic QA series, neural repair, new snapshots or simulation.

## Concrete likelihood

Retain the saved ICRS sky directions and distance moduli/sigmas in
config/cf4_lg_observation_contract_v1.json (M31/M33). Use only its2x2 distance
covariance, including its shared ladder term; no LOS/PM/mass constraints here.
The existing covariance is an EXPLICIT development approximation, not complete
measurement/systematic calibration. Native h=.6774 and z=0 units conversion.
Use fixed tabulated solar position relative to the MW and ZERO additional
stellar-to-halo position offset for this development computation. The latter
is an approximation, not a measured offset law; production remains disabled.

Continuous position law: uniform within each .1875 cell, joint density
qW(cW) qA(cA|cW) qT(cT|cW,cA)/dx^9. MW halo position is a fixed point
observation for this development operator, not inferred native identity.
M31 and M33 sky directions are exact observed directions at tabulated
precision. Evaluate their JOINT predictive density per solid angle and per
distance modulus, not a conditional-direction density with a missing
field-dependent normalization. For fixed observer O and each sampled true
distance modulus pair u, points are O+D_i(u_i) n_i, and the volume Jacobian
per companion is D_i^2*dD_i/du_i = (ln10/5)*D_i^3.

Lpos = E_{u~N(d,Cdistance)}[
 qW(c(xW)) qA(c(O+D_A n_A)|cW)
 qT(c(O+D_T n_T)|cW,cA) /dx^9
 * (ln10/5)^2 D_A^3 D_T^3 ].

All positions/distances in cMpc/h, so units/constants explicit; MW contributes
a density in Cartesian position. Correlated Gauss-Hermite quadrature order16
and32 checks numerical stability (no broadening observations or best order
chosen by the science score). Sum over ALL quadrature-induced parent cells,
including shared M31/M33 cells. Cell labels arise from integration of the
observation model, not truth-seeded free generation. Previous unconditional
autoregressive draw evidence remains separately reported; this is a likelihood.
No native halo mass floor or artificial mass is inserted in empty cells.

Existing model conditioned on Earchive does NOT define p(Earchive|F,O) or
an actual MW-selected field prior. Report Lpos|Earchive and refuse scientific
posterior status. Do NOT multiply the earlier data-conditioned mark posterior
as a second likelihood. Missing offset/selection corrections cannot be hidden
as a Gaussian epsilon or declared a new observational error.

## One bounded computation

Reuse the same8 frozen test observer fields from347085. Re-read ONLY their
72^3 buffered seven-moment cubes and needed existing SubhaloPos catalogue
rows. No new particles/profiles/full400^3 feature construction.

1. Implement differentiable five scalar features with identical periodic halo
   support and cold/empty conventions to the accepted numpy model. Freeze
   theta and alpha. Verify native feature agreement and the quadrature against
   a small analytic uniform-field/measurement control inside the same job.
2. For each8 fields, form native mock observations for ALL archived triples
   with the existing hierarchical weights. Native exact centers are used only
   as mock measurements; no arbitrary candidate selection or new fit. Score
   own-field versus the other7 fields with a common observer/core convention.
   Report conditional-position cross scores without inventing independent
   coverage or requiring every field rank first.
3. Score the actual saved LG distance/sky data in all8 native fields using
   fixed solar reference. Fields are NOT CF4-aligned local-universe candidates;
   these8 actual-data scores are interface diagnostics, not posterior weights
   and not field/seed selection. Compare16/32 quadrature; report discrepancies
   rather than threshold-chasing when a ray crosses a coarse voxel boundary.
4. On the first frozen field, compute the actual Lpos gradient through the
   native seven moments/features. Compare analytic derivatives to centered
   finite differences along three convex subcell-permutation directions:
   F(eps)=(1-eps)F+eps P_axis(F), permutation within each8^3 parent block.
   Such positive mixtures conserve each parent M,Pxyz,Qdiag and preserve
   moment realizability, but are NOT assumed LCDM field draws. At eps=0 use
   a forward finite difference (negative eps need not be realizable).
   Save compact projected sensitivity maps and derivative/conservation values.
   No density optimization/painting or ad hoc local posterior is authorized
   by this diagnostic; its purpose is a verified same-field likelihood.

Minimum operational checks: finite normalized probabilities; native feature
agreement; measurement-Jacobian/covariance and shared-cell handling; derivatives
and conservative direction construction. Reuse existing tests where possible.
No full audit framework; numerical checks run in this same Slurm allocation.
Scientific result may be "likelihood implemented, quadrature limited"; never
turn scores on8 native fields into an accepted cosmological field prior.

## Resources and next autonomous decision

OneGPU/2CPU/6GiB/30min, a40,a100,h100,h200 excluding syn06. Estimated host
peak<=4.5GiB +20% rounded6GiB; GPU envelope8GiB. Stream8 small haloed fields,
do not cache full probability maps across all observers/parents. Application
cap25min, no optimizer. Source-pinned Slurm, new output position_link_v1.

After completion the driver evaluates whether this is usable as a likelihood
component. Next substantive bottleneck is a calibrated joint fine-field/
mass/COM distribution and observation-selection treatment, not more standalone
location scores. Before any new field training, design a concrete bounded
replacement using the enlarged spatial data population and stable physical
coordinates; do not revive failed13-patch epsilon-diffusion/flow objectives
without new evidence. The z=0 posterior -> IC -> forward route is unchanged.

Fable5 advice requested: explicitly address Q-GOAL, Q-LEAN, mathematical
likelihood/Jacobian and observational conditioning, MW/M31/M33 identification
including shared cells, feasibility, essential/deferred scope, and whether
this is a proportionate bridge to the final CF4/LG field goal. Give actual
verdict and essential corrections; do not request an inspection/audit ladder.
