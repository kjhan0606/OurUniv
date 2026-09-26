# R2/5 source semantics and observation assembly

Goal: remove false missing-ID/calibration blockers using the actual source,
not another covariance fit. Howlett+2022 sec.2.2 explicitly states that418
full-sample objects have no Tempel counterpart and are assigned no group.
It also states that group redshifts were recalculated to avoid Tempel's
additive approximation. Verify these semantics against existing frozen
tables; never infer a counterpart by nearest neighbour or call source
ungrouped objects physically isolated. Preserve all10,020 FP observations,
the source group eta numerator, existing same-field link and split.

Implement a small membership classifier plus an aligned observation artifact
including original redshifts/errors, exact integer IDs and catalogue-present/
absent/ambiguous membership status. Compare the individual redshift conventions
once without fitting an offset or changing observations. Reuse the existing
map-zero/crossmatch evidence rather than repeat HEALPix neighbourhood scans.

Q-GOAL: correct source data semantics before the joint present-field likelihood.
Q-LEAN: existing tables only, two targeted classifier tests, one <=5min Slurm
job; no download, new PM run, covariance sweep or sampling. Estimated host
peak <=0.8GiB plus20%, rounded to1GiB. H200/H100/A100 available/mixed;
choose one typed H200 allocation. Numerical CPU work inside Slurm only.

MW/M31/M33 identification remains a NEW-state R3 readout, with ambiguous
MW/M31 assignments and unresolved M33 retained. Their observed positions,
masses and velocities must constrain the SAME evolving state; source labels
here never supply truth identities to a generated field.

The literal point selection at recno67100 stays unresolved. Its coarsened
count is valid under the existing development count representation. If it
has no relevant distance mark, its exact position need not be retained in
that count-only part of a future mixed-resolution likelihood. However, a
mark-dependent observation/coarsening law is still required; do not silently
drop its position because it contradicts a map or proclaim the full joint
law calibrated. Group COM discrepancy and selected-group distance priors also
remain open. No posterior promotion is authorized by a successful assembly.

Source: https://arxiv.org/html/2201.03112 (sec.2.2 and footnote2).

## Completed execution and decision

H200 Slurm406020 completed3s/exit0; both focused tests pass. Scheduler MaxRSS
is not a reliable peak estimate for this three-second job. Result:
`/gpfs/kjhan/CF4/z0_density/r2_source_observation_assembly_v1/result.json`.
Reusable aligned input: `observations.npz` in that directory, containing all
original source-link arrays plus source identities, membership states and
original individual redshifts/errors. No original artifact was overwritten.

Full34,059-row catalogue:23,953 grouped matches,9,688 catalogue-present
ungrouped rows and418 catalogue-absent ungrouped rows. The418 reproduces the
paper exactly. Selected10,020 rows:7,621 grouped,2,324 present ungrouped and75
absent ungrouped; unresolved membership conflicts0. Thus close the75-row
identity concern: these are explicitly source-ungrouped observations, not
failed distance measurements or missing grouped members. Do not turn this
into proof of physical isolation or assign them measured group dispersions.
Their group-versus-individual cz differences are <=1.499 km/s, consistent
with differently rounded source columns.

For9,945 matched individual galaxies, the original cz mismatch has absolute
median9.084,p90 16.143,max30.935 km/s. The fixed additive-convention comparison
gives1.591,2.909,19.500 km/s. This supports the published convention explanation
but does not reproduce every spectrum or validate a group-level correction.
Keep the published FP zgroup/eta numerator unchanged. In particular, do not
calibrate the provisional50-km/s group-catalogue noise from the original
uncorrected offsets.

The single zero-selection recno67100 has no CF4 crossmatch edge. Its exact
position is not a shared FP redshift in this assembly; the inclusive count
retains it. This narrows where a position-level model is necessary, but the
literal full point-process support problem is unchanged. Any future mixed
count/mark likelihood must define its observation/selection process for all
such objects, not hide this point with an after-the-fact exception.

Driver accepts the source-semantic assembly and closes this identity check.
No new covariance fit, sampler, gravity run or posterior exists. Next bundle
must advance the joint field-dependent group-distance/selection law using
these inputs; do not repeat catalogue matching or redshift residual sweeps.
Unknown COM discrepancy and selected-group radial prior remain the scientific
limits, distinct from the now-resolved absent-catalogue semantics.
