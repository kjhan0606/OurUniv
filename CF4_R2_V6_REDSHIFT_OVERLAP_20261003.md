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

Typed-H100 job 410985 failed in the test phase after four seconds (MaxRSS
1.4 MiB): a stale test import name. No source rows were reached; log preserved
at `/gpfs/kjhan/CF4/logs/cf4_R2_v6_zoverlap_410985.err`. Retry 410986 passed
all three tests, then caught a real selection omission before velocity
summaries (5 s, MaxRSS 3.44 MiB): the reconstruction had not restricted secure
links to the frozen 2M++ count-point manifest. Its log is preserved at
`/gpfs/kjhan/CF4/logs/cf4_R2_v6_zoverlap_410986.err`; no velocity comparison
or output directory was produced. The point-manifest hash/ID filter and a
focused exclusion regression are now included. Retry uses fresh output path
`r2_v6_redshift_overlap_20261003_v3`.

Corrected typed-H100 job 410988 completed in five seconds, MaxRSS 3.29 MiB;
all four focused regressions passed. It reproduced the frozen 272-group,
828-secure-link cohort and all six relation-class counts exactly. Grouped
velocity residuals (absolute km/s; unique CF4/2M++ group pairs for the final
column) are:

| CF4 / 2M++ relation | Groups | Member-pair Vcmb Δ median (p90) | CF4 group / linked CF4 member Δ median (p90) | CF4 / 2M++ group Vcmb Δ: pairs, median (p90) |
|---|---:|---:|---:|---:|
| shared / shared | 167 | 7 (51) | 205 (679) | 148, 79 (285) |
| shared / partly unassigned | 32 | 7 (39) | 451 (1,398) | 30, 100 (1,724) |
| shared / all unassigned | 54 | 7 (33) | 139 (535) | unavailable |
| shared / distinct | 11 | 4 (33) | 408 (1,140) | 21, 551 (1,422) |
| distinct / shared | 4 | 7 (9) | 178 (587) | 8, 194 (749) |
| distinct / partly unassigned | 4 | 5 (40) | 209 (370) | 4, 22 (43) |

All 828 crossmatch group IDs agree with the CF4 member table. The direct
individual velocity pairs are repeated catalogue measurements of matched
objects; the large and relation-dependent group/member offsets do not estimate
independent redshift noise. The driver therefore answers the planned question
NO: these fields do not specify a covariance law or calibrated shared-group
factor. Q-GOAL is satisfied as a prerequisite diagnosis; Q-LEAN is satisfied
because this one frozen-cohort readout reuses prior all-sample source audits
and adds no fit, gate, simulation, heldout outcome, or posterior claim.

The official EDD [CF4 All Groups definition](https://edd.ifa.hawaii.edu/describe_columns.php?table=kcf4allgroup)
distinguishes `Vcmb` (group velocity averaged over Tully 2015 2MASS group
members) from the [FP Groups table](https://edd.ifa.hawaii.edu/describe_columns.php?table=kcf4fpgroup)'s
`gVcmb` (average over all group members), `Nest`, and `Ng`. The v6 relation census above uses
CF4 `1PGC` and local 2M++ `GID`, not the Tully member identity. The next
bounded source step is a v6-subset crosswalk to the already archived Tully
member list: test exact matched-PGC membership in the Tully nest associated
with each CF4 `1PGC`, and report unresolved/multiple associations separately.
This may clarify source ownership but still cannot calibrate the later EDD
revision's covariance or selection law. No posterior or IC is promoted; R2
remains NO-GO. MW/M31 role ambiguity and unresolved M33 remain unchanged.
