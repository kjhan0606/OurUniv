# C-spatial execution record

User authorized entry2026-09-08. This remains within master C; no D/IC/zoom
entry. Design: BUNDLE_C_SPATIAL_DESIGN.md.

Implemented first native spatial calibration job:16 training and16 heldout
MW-role selections from the unchanged satellite population, one companion
pair per observer, unique source members streamed once. All member gas/DM/
stars+wind/BH dynamical masses participate. Native profiles, total velocity
covariance and sparse fine moments link to exact total-matter remainders.
Native mass/COM, residual realizability, reconstruction and coarse restriction
are checked in the calculation using existing moment code, not a new generic
audit framework. No sphericalization, frame rotation or actual field inference.

Output `bundle_c_v1/spatial_calibration_v1`; resource request2 CPUs/4800 MiB/
30min, estimated4000 MiB peak+20%, <=50m source rows and<8 GiB outputs.
This job prepares the missing spatial calibration INPUT. It does not fit the
joint spatial probability law or deliver the CF4/LG-on/off posterior map.

Submitted Slurm335916, sourceee6dd13 (committed/pushed). Logs:
`/gpfs/kjhan/CF4/logs/cf4_C_spatial_335916.{out,err}`. No automatic downstream
fit or simulation is queued. Only final result.json and the complete HDF5
status establish completion of the calibration input, not this bundle.

335916 COMPLETED, exit0, elapsed4m08s; sourceee6dd13. Two reused moment/
transport tests passed. Read16638729 selected native massive particle/cell
rows from93 distinct subhalos. All32 patches completed native-member mass/COM,
nonnegative realizable remainder, component+remainder reconstruction and
fine-to-coarse checks. The code requires native mass agreement2e-4, COM0.2
km/s, full/restricted reconstruction1e-8; no tolerances or source cases were
changed after outcomes. Recorded batch MaxRSS1022340K is below the4800 MiB
request, but is sampled accounting, not an exact memory bound.

Products in `bundle_c_v1/spatial_calibration_v1`:
- `request.json`: source IDs, member ranges, native cosmology and case policy.
- `spatial_components.h5`:93 sparse native halo moment components/profile
  metadata,32 full128^3 remainders at0.1875, and total1.5 coarse moments.
- `result.json`: all profiles, native unit/mass/COM checks and patch reports.

Status NATIVE_SPATIAL_CALIBRATION_INPUT_READY_NOT_BUNDLE_COMPLETION. This is
the first substantive input of C-spatial, not an observed field or a fitted
continuous spatial prior. No three arbitrary peaks, profile sphericalization,
halo movement, CF4/LG likelihood, original full-source re-read or new simulation
was used. No calculation remains active or queued from this first step.

Native primary/satellite object IDs are disjoint across the16/16 split, but
whole patches can share native voxels. The next remainder-field fit must
enforce source-voxel exclusion or use a separated subset for heldout scoring;
these object labels alone cannot certify full-field generalization. Member
profile checks and spatial-field checks have different leakage conditions.

Next authorized work within C-spatial: implement the joint spatial probability
law for member profiles and remaining matter, then actual CF4/LG-on/off maps
with measured uncertainty. Do not close this bundle on the successful source
decomposition or substitute another catalogue-only fit. D remains unentered.

Next within-bundle calculation implemented: `cf4_bundle_c_spatial_model.py`
and `cf4_spatial_copula.py`, with two focused tests and a Slurm runner. Fits a
conditional remainder distribution and generates heldout0.1875 fields under
fixed native1.5 moment/halo conditions; profile regression saved separately.
This does not yet implement the FULL joint prior or actual observation fit.
Predeclared split, morphology criteria, limitations and stop rule are in the
design. Slurm request2 CPUs/7200 MiB/30min (6000 MiB estimated+20%), <4 GiB
output, no source re-read. Implementation awaits the single test/fit job.

Submitted Slurm335968, sourcefc7235e (committed/pushed), with the two tests
followed by the conditional spatial fit. Logs:
`/gpfs/kjhan/CF4/logs/cf4_C_spatial_model_335968.{out,err}`.
No downstream observation fit or simulation is queued. Interpret result.json
separately from the scheduler exit; morphology rejection is a scientific
NO-GO even if the calculation exits normally.

335968 COMPLETED on syn05 via Slurm, exit0, elapsed2m46s; sampled batch
MaxRSS1237416K, below7200 MiB requested. Both focused regression tests passed.
Geometry-only selection retained13 training patches and three mutually
nonoverlapping heldout patches16/17/20, with no shared training/test voxels.
All12 conditional draws were generated and passed realizability and exact
coarse7-moment conservation (maximum channel-normalized error1.1472e-15).

Scientific result: NO_GO_CONDITIONAL_REMAINDER_MORPHOLOGY. All12 draws fail
the predeclared small-scale power criterion in BOTH total and remainder.
Total-field fine-band power ratios span0.1351–2.3332 (permitted0.5–2);
11/12 have insufficient power in the first fine band, while the other has
excess power in the two highest bands. The first fine band starts at the
1.5-grid Nyquist, k=2pi/3 h/cMpc. Top1%mass fractions are0.7761–0.8469
times native and hot-component fractions0.7378–0.8829; these broad two
morphology checks pass. Thus the observed rejection is specifically the
power criterion, NOT a demonstrated failure of every topology statistic.
Fixed native member components barely change the remainder-only verdict.

Products `bundle_c_v1/spatial_conditional_v1/{model.h5,conditional_fields.h5,
result.json}` preserve the fitted candidate, native references, all12 full
7-moment generated fields, four-draw diagnostic mass means/variances, and
separate profile regression. No draw is a CF4/LG reconstruction. No physical
velocity-distribution validation or calibrated uncertainty claim follows
from the successful moment conservation. Profile regression is not coupled.

Close this ONE Gaussian-copula conditional candidate without amplitude
repair, best-seed selection or a variant series. The result demonstrates
that this implementation's stationary empirical copula plus conservative
conditioning is inadequate under oracle coarse conditions. It does not
isolate a single cause (nonlinear spatial phases, marginal tails, environmental
conditioning and the conservative mapping all participate), nor prove the
project impossible. Do not attribute the outcome solely to random phases.

C-spatial remains unfinished. Its missing deliverable is still a defensible
joint environment/member/remainder distribution connected ONCE to native
CF4/count and LG likelihoods, with LG-on/off maps and information gain. The
failed candidate must not become the prior for that inference. A replacement
needs explicit nonlinear environment/halo–matter spatial dependence; source
profiles and a mark Gaussian alone are insufficient. No automatic replacement
fit, observation inference, IC or simulation is queued. Next substantive
design must address that missing dependence, not add generic validation code.
