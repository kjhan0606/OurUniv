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
