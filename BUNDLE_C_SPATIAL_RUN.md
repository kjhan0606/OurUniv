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
