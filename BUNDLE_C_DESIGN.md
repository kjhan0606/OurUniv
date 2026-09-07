# Bundle C — observation-driven multiresolution present-day fields

User authorized entry after Bundle B, 2026-09-07. Driver Astra; external
pre/closure audits waived. C is IN PROGRESS, not a high-resolution delivery.
Authority: CF4_MASTER_PLAN.md. No Bundle D/zoom production is authorized.

## Result to work toward

Infer z=0 density and mean velocity using CF4, disjoint galaxy observations,
and MW/M31/M33 observations, at surrounding-cell size1.5 cMpc/h and LG-cell
size0.1875 cMpc/h. Numerical cell size is NOT information resolution or halo
force resolution. Keep observation-constrained, model-prior and unresolved
components distinct. Do not restart the superseded direct-CF4/peak-IC bank.

Use the existing384 cMpc/h domain. The native12 cMpc/h diagnostic parent
centres are(i+.25)*12; its cell faces are12*i-3. An8x child grid therefore
has centres-3+(j+.5)*1.5, NOT the standard(j+.5)*1.5 or PM(j+.25)*1.5.
The LG patch initially covers parent cells[15,17) along all axes, physical
faces[177,201) or SG offsets[-15,9) cMpc/h. Another8x refinement gives
128^3 local cells at0.1875. This asymmetry preserves the existing native
cell boundaries and still buffers the LG by at least9 cMpc/h.
Count-cell coordinates remain their own ordinary zero-origin partition.

## Substantive work within C

1. Retain the *native actual point data*, with exactly the corrected magnitude
   convention, row exclusions and three-way mark split already used in A.
   Assign sparse1.5 cMpc/h count cells; prove integer aggregation reproduces
   the old12 cMpc/h counts. Preserve CF4 positions/radial observations/errors
   and individual holdout identities. Report how many direct observations
   reach the LG, rather than asserting the parent already constrains it.
   Keep selection/response calculation separate; galaxy counts are not matter.
2. Implement a positive, exactly mass/momentum-conserving child field parameter
   map and its native coordinates. Test the actual diagnostic parent as well
   as a tiny algebraic example. A zero-detail refinement is only a baseline;
   **do not publish it as a reconstructed fine map or add arbitrary peaks**.
3. Construct the fine z=0 observational model. Galaxy likelihood must use
   native data plus newly integrated selection at the working resolution,
   not interpolate old exposure or fine density priors. LG must use the
   identity-preserving observational contract with a physically defined
   halo/subhalo operator and explicitly separated mass/profile/COM priors.
   Covariance and PM-reduction-distance systematics in B are not solved by
   declaring the old80 km/s screening width an observational uncertainty.
4. Execute one bounded multiresolution inference/control comparison once the
   operator and scale-dependent prior are specified. Compare LG constraints
   enabled/disabled with the SAME prior/initialization policy; retain all
   selected fields. Show native observation predictions, spatial maps and
   information changes attributable to LG data. Coarse fields may update in
   the joint model: B is not an exact validated boundary to freeze forever.
5. Close C with achieved numerical/information resolution, model sensitivity,
   actual LG/environment consistency and the remaining dynamically compatible
   IC/zoom requirements. Stop before D for approval.

The first implementation now covers1–2. These are necessary data/operator
work, NOT completion of3–5. No additional A sampler extension or B mock
optimizer run is required. Do not hide an unspecified high-resolution prior
behind a generic validation framework or call sparse count binning a posterior.

## Probability and feasibility restrictions

- Final target is joint z=0 coarse/fine + selection + LG nuisance inference.
  If old posterior samples are reused as proposals, account for the old
  likelihood/prior exactly; do not multiply old and new data factors twice.
- A conservative log-density detail map preserves positivity and coarse
  integrals, but does not by itself establish an LCDM/nonlinear field prior.
  A Gaussian at IC cannot be labelled Gaussian z=0 density. Old N32 spectral
  covariance is not calibrated on1.5/0.1875 scales; extrapolation is not a fix.
- A0.1875 cell does not resolve a virial halo/subhalo force structure. The
  resolved catalogue operator may need finer internal particles or a tested
  subgrid model. Never pass `resolved_halos=True` for three imposed grid peaks.
- Fine matter templates and unknown mass/velocity discrepancy require explicit
  assumptions and tests before use. No claim of actual LG conditioning until
  observations update this model rather than merely label a displayed map.
- Parent CF4 uses cz1500..18000 km/s and counts r5..180 cMpc/h. Check actual
  retained rows for direct local overlap. Shared distance/photometric calibration
  can remain even if MW/M31/M33 themselves are absent from those rows.

Initial native-data build and coupling tests: Slurm2 CPUs, estimated3000 MiB
+20%=3600 MiB, at most20min; <0.1 GiB new outputs. No GPU, new simulation,
RAMSES output, filesystem investigation or process-monitor loop for this work.
Record later inference resource estimates before submission; do not silently
launch a uniform2048^3 box or an unbounded training ensemble under C approval.

Next within C3: stream N256 angular/LF/shell selection in x-slabs of8,
4^3 Gauss points per cell, float32 compressed HDF5 (<3 GiB new product).
Reuse the unchanged source completeness masks/LF conventions; verify radial
tabulation error <=1e-6 and rotation before integration. CPU2, estimated
3000 MiB+20%=3600 MiB,30min cap. No old-exposure interpolation, count-driven
support insertion or promotion to calibrated selection. Report unresolved
positive-count support if present. The N256 density/count origin offset is
exactly2 whole cells: use a cyclic field permutation, not interpolating either
the field or survey mask. A numerical integration finish is not field inference.

Completed implementation amendment: order4 missed one occupied footprint.
The geometry-only v2 zero-support repair and preservation checks are complete;
see BUNDLE_C_RUN.md. The repaired selection is an input for likelihood
development, not calibrated selection: finite thin-boundary cubature error
remains explicitly measured in the controls. Do not remove observed rows,
insert a probability floor, or label this selection product a matter map.

## Resolved physical operator implementation (approved C continuation)

The next implemented component uses existing TNG100-1 z0 particles and native
SUBFIND catalogue, not retraining Hong networks or inventing LG grid peaks.
Native metadata/selected catalogue fields and one bounded FoF particle set
are copied from syntax-local /scratch to the project output directory by
I/O-only staging. All numerical selection, projection and tests use Slurm.
No new download, simulation, storage investigation, or full snapshot copy.

Native positions are ckpc/h, particle velocities need sqrt(a), whereas
SubhaloVel is already peculiar km/s. Bound SubhaloMass is NOT host M200c;
only a FoF primary exposes Group_M_Crit200 as its host mass. Definitions:
https://www.tng-project.org/data/docs/specifications/ . Explicit identity
assignments and frame rotations are required; no best anonymous pair search.

Use one predeclared MW-mass-range FoF group with three particle-resolved
subhalos as an engineering fixture, not an LG analogue or observational prior.
Check native member masses/COM velocities against the catalogue. Deposit
actual FoF particle mass, momentum and diagonal second moments on a24 cMpc/h
cube at0.1875; aggregate to1.5 and compare direct particle deposition.
Save density, mean velocity, physical sigma_v and an empty-cell validity mask.
Physical sigma_v is not posterior uncertainty. The product lacks external
diffuse matter and is therefore explicitly a FoF component, not a total
matter map or an actual CF4-conditioned map. No component is pasted into CF4.

This implements a particle-backed readout and moment representation, NOT a
calibrated fine-field prior. The conditional ensemble distribution of these
components plus diffuse matter, cosmology dependence, stellar/halo COM offsets
and joint CF4/LG conditioning remain to be implemented. A single fixture
cannot establish that prior. The LG-on/off inference cannot be claimed from
this check. Two bounded CPU jobs:2 cores, estimated3000 MiB+20%=3600 MiB,
10min each; catalogue staging plus selected particles/products <2 GiB estimate.

The fixed native-catalogue readout belongs to the original particle
realization. Do NOT alter fine density/momentum with the generic conservative
map and continue treating the old SUBFIND catalogue as its resolved halos.
A changing field needs a jointly defined halo state or recomputed physical
readout. Passing this fixture settles implementation units/moments, not the
conditional prior or its cosmological/statistical validity.

## Total-matter conditional-prior development

User authorized the next implementation after the native operator check.
Implement a finite whole-patch mixture, not independent shuffled cells or
arbitrary NFW/point-peak insertion. Each component carries unchanged density,
momentum, second moments and the linked native SUBFIND catalogue. A diagonal
Gaussian kernel on8 parent log masses and24 parent bulk-velocity components
defines p(component | coarse summaries); its bandwidth is a declared PRIOR
hyperparameter (training SD), not an invented observational uncertainty.
New observation factors reweight that distribution only once. Report ESS and
zero support; do not force a seed or interpret a collapsed bank as a posterior.

The first support baseline has27 disjoint24-cMpc/h patches from one75-cMpc/h
TNG box (18 training,9 spatially nonoverlap checks). These are not independent
universes, a continuous LG prior, or exact conditioning on every parent cell.
Actual CF4/LG weighting remains disabled until its joint field/observer model
is specified. This finite model is a support baseline, not C completion.

The required new physical source includes ALL gas, DM, stars/winds and BH
dynamical mass, including unbound/diffuse and external-halo matter. Stream
existing snapshot arrays once from syntax-local storage into a Slurm CPU job;
source process performs I/O only, with one SSH connection. No whole raw
snapshot is copied and no new simulation/download is requested. Deposit at
400^3 (0.1875) and conservatively restrict to50^3 (1.5), cross-check against
direct deposition. Velocity moments describe bulk particle/cell kinematics,
not gas thermal broadening or posterior uncertainty. Partial file streams
are timing checks ONLY and cannot train a spatial prior.

First run a bounded2-file timing/resource check, then at most one full448-file
source pass if feasible. Slurm2 CPUs, estimated8000 MiB peak (including a
full absolute-moment reduction temporary) +20%=9600 MiB; timing cap15min,
full cap4h, estimated<8 GiB permanent new products. The full pass reads
several hundred GiB of existing raw fields; runtime must be estimated from
the timing check, with source-order/spatial-occupancy caveats. No GPU, IC or
RAMSES job, phase rescaling, GPFS investigation or process-scan loop.

## One bounded translation/isotropy support comparison

The completed full source passes physical/moment checks but the18-component
coarse-summary prior has heldout ESS1.00–2.06. User authorized continuation.
Reuse the50^3 coarse moments and ONE fine patch; do not reread raw particles.
Freeze original9 validation targets,32 conditioning features and bandwidth.
Compare original18, rotations-only, dense native origins and dense+rotations
in a single run. Origins: x0..18,y/z0..34 in coarse cells; each24-Mpc/h patch
ends before heldout x=34. No training/validation native voxels overlap.

23275 native anchors ×24 proper cube rotations yield558600 component
hypotheses, NOT558600 independent simulations. Rotate mass, polar momentum,
diagonal second moments and catalogue coordinates consistently, by exact
axis permutations/signs. No interpolation, amplitude changes or phase draws.
Report raw ESS, rotation-collapsed anchor ESS and concentration grouped by
native patch-centre24-Mpc/h tiles. Spatial groups can still be correlated or
overlap; grouped ESS is not an independence certificate.

Retain the old support threshold on the grouped diagnostic: all9 require
ESS>=4 and max source-group weight<=0.5 to count as improved finite support.
Neither outcome is an actual LG GO. Also report standardized nearest/mean
coarse-feature residuals. If insufficient, STOP this finite-bank repair;
design continuous coupled matter/halo states rather than expanding the bank
again or widening the kernel. CPU2, estimated1500 MiB+20%=1800 MiB,10min,
<0.1 GiB new outputs. No new simulation, actual-data inference or Bundle D.

Closure:335875 returned NO_GO_FINITE_BANK_SUPPORT_CLOSE_THIS_REPAIR. Grouped
support fails4/9 targets despite improved coarse-feature matching. The
single bank-enrichment allowance is exhausted; preserve the full TNG source
for continuous joint matter/halo-model calibration instead of another bank
variant. The next model must couple changing halo marks and diffuse density/
momentum/second moments; the old catalogue cannot remain fixed while a new
field is freely changed. Priors and discrepancy calibration are outstanding,
not solved by the current support test or by declaring three imposed peaks
resolved. No automatic new fit follows this closure.

## Continuous coupled-state prototype — authorized continuation

Following user approval to proceed, implement ONE kinematic inverse control
on the existing total-matter source, not another template-support repair.
Subtract the three explicitly identified, disjoint native SUBFIND member
components from an integer-aligned24-cMpc/h total-matter patch. The remainder
includes diffuse matter AND every other structure; do not call it pure diffuse
matter. Separately summed near-empty subtraction residuals may be removed
only at1e-10 relative roundoff, with the removed moments reported.

Each component has7 continuous marks:3 translations, log member-mass scale,
and3 COM-velocity boosts. Exact overlap remapping of piecewise-constant cells
changes mass, momentum and second moments together. Transformed catalogue
positions/COM/member masses derive from that same state. Do not carry over
host M200c, unrelated native catalogue marks, or `resolved_halos=True`.
Native memberships/profile shapes are fixed; force resolution, gravitational
binding and a newly identified halo are not established by this operation.

Let the remaining mass fraction be alpha=(M_patch-sum M_components)/M_rem.
Reject alpha<=0. A common remainder bulk-velocity change preserves total
patch momentum. This positive reservoir construction is a declared algebraic
response, NOT calibrated halo accretion or diffuse response. Individual parent
cells may change; report exact fine/coarse restriction. Second moments transform
with bulk boosts and preserve realizable component dispersions; kinetic energy
need not remain constant when changing the state. Mean-velocity posterior
uncertainty remains unavailable because this experiment is not a posterior.

The bounded test injects21 dimensionless marks inside[-1,1], with unit scales
0.1875 cMpc/h,0.2 log mass,30 km/s. These are numerical test amplitudes/bounds,
NOT astrophysical priors or observational uncertainties. Fit only mass/momentum
on alternating fine support cells, reserve complementary cells and all second
moments for prediction. Require parameter error<0.02 in these units and all
heldout moment errors<2% of injected changes, in addition to1e-8 conservation
and restriction. A noiseless same-generator recovery is implementation evidence
only; it cannot settle the failed nine-condition support test or physical prior.

After this ONE control, the next substantive requirement is physical response/
profile/COM-discrepancy and prior calibration, not additional toy-control
variations. Do not activate actual LG likelihood until those choices exist.
No new full snapshot pass, GPU job, simulation, IC, or Bundle D. Slurm2 CPU,
estimated3000 MiB peak +20%=3600 MiB,15min cap, <0.5 GiB new fields/reports in
`bundle_c_v1/continuous_transport_v1`. This is not an approved global extension
of the N32 prior to fine scales or a density reconstruction delivery.

## Native population calibration and actual LG mark conditioning

User approved progressing toward a physical continuous prior and real LG-on/off
comparison. Next implement the marked population and observational link, not
another inverse toy. This is an intermediate scientific product: full angular
remainder/profile fields are still absent, and the result MUST NOT be named a
0.1875 density posterior. No painted-halo or stale-catalogue map is delivered.

Use existing full TNG catalogue and50^3 total-matter moments. Population policy
is fixed before outcomes: two distinct FoF primaries with M200c2e11..5e12 Msun,
>=1000 DM particles, separation200..3000 physical kpc. Third valid subhalo has
bound mass1e9..1e12, <=half second-host M200c, >=1000 DM and100 type4 members,
and separation50..1500 physical kpc from the second host. Treat third-object
membership as TWO conditional alternatives: satellite of second FoF, or a
different FoF primary. Satellites of a fourth host and shared-FoF primary pairs
are outside this pilot's population definition, not proven impossible for LG.
No observed mass likelihood, isolation/binding/first-infall cut or best-pair
choice. Broad eligibility is an explicit astrophysical prior assumption.

Probability sampling measure: uniform retained MW-role observer, uniform its
eligible M31-role companions, then uniform eligible M33-role candidates. Do
not overweight rich hosts by counting every triple equally. Fit branches
separately. Training uses all three native x<50 cMpc/h, heldout all x>=50;
omit straddling triples. Native identities cannot overlap; long modes and
shell-field voxels may still overlap, so this is NOT independent-universe
validation. Require>=30 distinct observers in each split and>=100/50 rows.

Continuous coordinates:3 log masses, log r31/log r32/atanh triangle cosine,
six asinh relative-velocity components in the oriented triangle frame, and
six native matter-shell summaries at2..4 and4..8 cMpc/h (log density/mean,
radial flow/100km/s, log physical dispersion/100km/s). One weighted Gaussian
in these18 coordinates, frozen5% diagonal covariance shrinkage, truncated to
declared mass/distance eligibility. This is NOT Gaussian fine z0 density.
Report heldout score against diagonal with both truncation normalizers
estimated using131072 proposals each. No tuning or scientific GO by sampler
success alone. Shell summaries do not replace32 full environmental features.

Calibrate a stellar-minus-native-halo COM proxy on fixed-seed32 training
centrals and32 satellites. Read at most20m requested type4 rows through ONE
SSH I/O-only source process on syntax; all calculations stay in Slurm. Source
ordering/units: https://www.tng-project.org/data/docs/specifications/ . Use
native stellar half-mass radius, exclude formation-time<=0 wind particles,
and measure star COM inside1x and2x half-mass radii. Keep position AND velocity
offsets and their cross-covariance, isotropized with zero vector mean. This
is a pooled central/satellite proxy, not a calibrated disk/Gaia fit, gas LOS,
mass-dependent COM relation or a new observational error. Physical particle
dispersion and error in mean velocity are not interchangeable.

Use actual B-contract eight distance/LOS/PM values and conditioned stellar
sky directions. Explicitly opt into its partial covariance and fixed solar
reference; do not change the saved contract or falsely set resolved_halos=True.
Marginalize joint6D position/velocity COM offsets for all three identities;
share the SAME MW draw between M31 and M33, retaining induced correlations.
Other cross-halo offsets remain independent by assumption. Apply Hubble and
solar reflex at each sampled distance, not a constant PM velocity conversion.
Missing Gaia cross-covariances, LMC/solar and PM reduction-distance systematics
remain limits, not solved by the TNG proxy. Compare both stellar apertures.

Within each branch/aperture use65536 proposals from the observational Gaussian
and COM proxy, integrate masses/shells via Gaussian conditionals, enforce
selection support, and retain importance weights and4096 resampled states.
The invariant-to-Cartesian and fixed-sky DM/PM Jacobians MUST be present:
variable log factor5sum(logD)-3(logr31+logr32)-log(1-cos^2)-sum(logcosh(v_asinh)).
Report proposal ESS, max weight, unique resamples, actual observable predictions,
prior/posterior mark/shell quantiles. ESS<1000 is proposal NO-GO. No posterior
branch odds without population/model selection normalizers. No CF4 factor is
used at this stage and no old CF4 posterior is multiplied again.

Two focused coordinate/conditional checks plus ONE native population/calibration/
actual-data run, Slurm2 CPUs, estimated6000 MiB+20%=7200 MiB,30min cap,<0.2 GiB
outputs in `bundle_c_v1/lg_population_v1`. It advances actual LG observational
conditioning of a physical reference population, not full-field reconstruction.
Profile/remainder-field calibration and joint CF4/environment coupling remain
required before the promised spatial LG information comparison can be claimed.
