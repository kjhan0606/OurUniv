# R2/5 official selected-mock bridge

Q-GOAL: replace unsupported scalar-scatter assumptions with source-defined
host/member observables where the public product genuinely provides them.
Q-LEAN: inspect metadata/docs, stream only the first regular mock member with
a64MiB compressed/128MiB member cap; do not acquire the10.643GB archive or
284MB random catalogue before establishing what they can calibrate. No PM,
new training, repeated derivative test or production posterior.

Official source: https://doi.org/10.5281/zenodo.6824749 . API inspected:
`https://zenodo.org/api/records/6824749`. The mock description distinguishes
z_true, z_obs, z_obs_cen (halo-centre redshift), cen_flag, parent/subhalo mass
and member/host velocities. It does NOT document Tempel recovered memberships
or a pre-selection parent denominator. Do not call z_obs_cen an observed
Tempel mean or infer a whole-group inclusion probability from selected galaxies.
256 simulation boxes each have8 observers; catalogues are not2048 independent
simulation boxes. This one fixture is not an ensemble covariance calibration.

The source mock's eta uses INDIVIDUAL z_obs, unlike the real FP group-redshift
numerator. Implement explicit conversion of both mock truth and fitted eta:
eta_group = eta_individual + log10[D(z_obs_cen)/D(z_obs)]. This does not refit
the FP, reproduce richness corrections, or make mock host IDs observed groups.
Preserve original fields; check residual invariance and cosmological distance
convention. Read internal member/host velocity separately from host COM/PM
discrepancy. The latter cannot be obtained without the matching matter field.

MW/M31/M33 remain identified from each NEW field, retaining MW/M31 ambiguity
and unresolved M33; their observables constrain that SAME state. Mock truth
columns are calibration/evaluation inputs only, never generated-field seeds.

One typed-H200 Slurm job,1CPU,2GiB (<=1.6GiB estimate plus20%, rounded),10min.
Choose after checking H200/H100/A100. Source archive checksum cannot be verified
for a partial download; record this explicitly and hash the extracted member.
Extract no archive-named paths. Preserve existing data and refuse overwrite.

Success provides reusable physical host/member inputs, not a group-selection
law, actual R2 posterior or permission to replace the real likelihood blindly.

## Result and limits

406142 acquired the first fixed source member, then failed on a driver error:
the CSV delimiter was omitted. Corrected406148 reused the saved bytes and
completed1s.406149 completed1s with the direct3D-velocity projection added,
retaining the redshift-offset proxy separately. No repeated download or changed
sample. Very short scheduler MaxRSS samples are not reliable memory peaks.

Final artifacts: `/gpfs/kjhan/CF4/z0_density/r2_sdss_mock_bridge_v3/result.json`
and `host_member_inputs.npz`; original file/source hash remain in v1.
Read5,304,320 compressed bytes; saved12,992,578-byte member
`mocks/MOCK_HAMHOD_SDSS_v5_R19051.5_err_corr` with33,881 galaxies. Source SHA256:
`e5b7ddebcf0a8c514dcf71acf093b174347d4b91c0658f66eae6bc8ac27e6630`.
No full-archive MD5 verification is claimed.

14,775 centrals have identical member/host velocities by construction;19,106
satellites have direct LOS relative-velocity RMS397.5927 km/s and absolute
50/90/99 percentiles232.902/634.837/1202.333 km/s. Observer5's documented
Cartesian origin is(1150,260,1200) cMpc/h; use the source member-observer
direction, not untransformed RA/Dec against simulation velocities.
The redshift-offset velocity proxy has RMS398.0186 and differs from the
direct velocity by RMS21.7677 km/s. Preserve both, do not conflate them.
Eta reference conversion preserves measurement-minus-truth residual to
5.55e-17; supplied rounded redshifts reproduce source truth eta within
7.071e-6. These are data/convention checks, not calibrated coverage tests.

Accept the bounded physical host/member input bridge. Do NOT replace the
real-data provisional150km/s with398km/s: this fixture is a selected satellite
mixture, not the error on a recovered Tempel group COM or a PM discrepancy.
No Tempel membership/parent-before-selection table is in the delivered schema;
the full mock archive's repetition of this schema is not sufficient reason
to download10.6GB or declare group inclusion solved. SDSS nbar/randoms describe
galaxy selection, not automatically the selected group-distance law.

Next scientific model decision must retain distinct individual/host/recovered-
group velocities and observational selection. A source-matched group mock
requires the parent galaxy population and the actual group-finding selection;
without them, any real-data group law remains explicitly approximate. The
current inputs permit a bounded host/member measurement-likelihood check,
not an automatic actual-field sampler. No new gravity run or R2 posterior.
