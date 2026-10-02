# R2 — v6 training group/redshift overlap readout (2026-10-03)

## Driver plan review

The identity census established that only 167/272 v6 multi-link training
groups have one shared assignment in both the CF4 and 2M++ catalogues. This
small source-only follow-up uses those exact 272 groups and the existing
secure member links to summarize individual and group `Vcmb` relationships
within each identity-relation class.

- Q-GOAL: yes. A same-field shared count/mark factor must not count repeated
  redshift information as independent when member links connect its catalogues.
- Q-LEAN: one relation-stratified readout of existing source fields is enough;
  no new empirical fit, synthetic covariance tuning, likelihood evaluation,
  heldout measurement use, field read, or simulation is authorized by this
  bundle.
- MW/M31 remain role-ambiguous and M33 unresolved. These catalog identities
  do not label generated candidates; later observables must constrain those
  roles on the same NEW evolved field at LG resolution <=0.3 cMpc/h.

The broad source comparisons already present in
`CF4_R2_GROUP_MEMBER_SCORE_STRESS_20260926.md`,
`CF4_R2_CONDITIONAL_GROUP_MARK_20260926.md`, and the CF4–2M++ group bridge
are reused as prior evidence; their all-sample comparisons and Student-t fit
are not repeated. The incremental question here is restricted to the frozen
v6 272-group cohort and its newly measured catalogue-relation classes. In
line with the user's routing rule, this is directly reviewed by the driver:
an Astra/Fable round immediately following the preceding driver audit would
be consecutive and add a duplicate reviewer layer.

## Calculation contract

`scripts/cf4_r2_v6_redshift_overlap.py` reconstructs the frozen v6 training
cohort and secure PGC/2M++-recno links from the frozen identity census (which
records the prior graph-census digest), verifies the FP grouping hash and all
catalogue hashes, then reads only the following velocity/ID
columns: CF4 member `Vcmb`, CF4 group `Vcmb`, 2M++ point `Vcmb`/`GID`, and
2M++ group `Vcmb`. It reports paired member-velocity differences, linked
group-to-member offsets, and CF4-to-2M++ group-velocity offsets, summarized
by the pre-existing two-catalogue relation classes. `Vcmb` is used on both
sides; no `V3k`/CMB frame mixture is introduced.

These are descriptive matched-source associations, not independent noise
samples, a physical group-membership claim, or a fitted covariance law. In
particular, the published CF4 systemic-velocity member process is not
identified by the CF4 distance-contributor membership table. The calculation
does not use heldout measurement values, FP distance marks, likelihood
scores, or field states, and performs no PM evolution. The source files are
hash-checked, while velocity columns are retained only for the selected
training links/groups.

Group-velocity comparisons are deduplicated by catalogue group pair within
each relation class; member-pair offsets remain link-weighted. Neither sample
is an independent error sample.

The single typed-GPU Slurm job uses H100, one CPU, 2 GiB host memory (over
20% above the conservative <1 GiB CSV/array estimate), and a 10-minute cap.
H200/H100/A100 modes are checked before choosing one compatible allocation.

## Result and next decision

Pending the frozen Slurm result. R2 remains incomplete and NO-GO for a
production posterior or IC. No next covariance fit is implied: the driver
will compare this cohort readout with the already preserved source audits and
decide whether it adds enough information to specify any shared factor.
