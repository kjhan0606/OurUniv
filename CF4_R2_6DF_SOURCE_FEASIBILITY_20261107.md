# R2: one 6dFGS source-feasibility decision — 2026-11-07

The full objective is unchanged: CF4+galaxy-conditioned z=0 density/velocity
posterior, 384 cMpc/h box, 1.5 cMpc/h grid, uncertainty, held-out prediction
and observation/sampler calibration. This investigates an alternative source
component, NOT a replacement target, posterior claim or R2 exit.

## New evidence and the precise opportunity

The [CDS release description](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/MNRAS/443/1231?format=html&tex=true)
lists a 11,102-row raw FP table with source/2MASS identities, JHK measurements
and errors, morphology, native groups, six per-band selection flags and
separate PV-membership flags. It also lists spectrum and photometry tables.
The [author paper](https://arxiv.org/html/1406.4867), especially sections 2–3
and 5.3, places this catalogue before the final FP selection and distinguishes
it from the 8,885-object PV sample. Earlier spectroscopy/photometry exclusions
remain, including image-processing and small-radius failures. Thus this is
not a complete targeting/failure roster.

Driver inference: write B for entry into the released raw table and R for
the authors' later FP/PV acceptance. If both R outcomes retain their raw
measurements, a new analysis of eligible B=1 data need not impose R=1. It
can avoid that **particular** residual-based selection instead of pretending
it cancels. This does not remove B, calibrate its response dependence, or
justify treating rejected galaxies as a single clean Gaussian population.
Using the PV flag/sixth-bit pass as a new filter restores the very problem.
These flags are provenance, not extra independent likelihood observations.

## Bounded access result, before any numerical FP inspection

CDS archive requests timed out; Harvard VizieR ASU did not expose the requested
table/catalogue. This is transport/service availability, not scientific
absence. The DOI resolves to the [official journal landing page](https://academic.oup.com/mnras/article/443/2/1231/1060810),
whose Supplementary Data link supplied the author ZIP. After correcting a
copied-URL typo, that exact public link downloaded successfully with a 10 MB
limit; no credentials, email, broad source census or new mirror loop was used.

Archive:
`/gpfs/kjhan/CF4/external/6dfgs_2014_source_probe_20261107/stu1198_Supplementary_Data.zip`

SHA256: `474c82f5f1338d1ed457ee7e86806388960e6b0b7c79f5cbebfb1343cb074283`.
Its directory contains individual `campbell_table2.zip`, `campbell_table4.zip`,
`campbell_table8.zip`, `campbell_table9.zip`, and two combined bundles. Retain
one original download; do not unpack duplicate bundles or execute archive content.

One CPU Slurm job inspects only the table-8 archive/schema and identifier
strings, recording the exact table hash and row/key counts. Numerical FP,
velocity, distance, environment and selection-flag values remain unparsed.
Even an FP-rejection bit is a derived outcome; do not summarize held-out bits.
Expected peak <=100 MiB, one CPU, 128 MiB request (>20% headroom), two-minute
cap. No GPU, population fit, field evolution, sampler or generic test ladder.

## Astra mid-course advice and driver disposition

Astra returned **CONDITIONAL GO for this one feasibility/identity probe;
NO-GO for replacement, field fitting, production or R2 exit**. It confirmed
the specific potential to remove R while warning that initial availability B
still depends on quantities relevant to an angular-size response. The driver
adopts this distinction after checking the primary source and successful
archive retrieval. This is not an exit audit or an email authorization.

Adopted requirements:

- Verify actual release identity and availability of pre-final-cut rows; a
  paper/schema claim is not a row-level demonstration of retained rejections.
- Establish source/group/CF4/count ownership before parsing FP outcomes. The
  existing CF4 table has PGC/group/sky coordinates but no 2MASS-name column.
  Exact names through 2M++ may help link some objects but may not define the
  science cohort. Keep unmatched/conflicting objects; never call them failures.
- Preserve the frozen v6 split and target. Any new source needs its own
  conservative identity/group closure; no accepted-only CF4 or unique-count
  requirement may be silently imposed on the raw source sample.
- Derive the native measurement transformation, errors and selection law.
  In particular, the optical–NIR aperture mapping prevents copying SDSS's
  transform unchanged; confirm angular/physical radius and group-redshift
  conventions. Do not mix preliminary 2012 fits with revised 2014 data.
- Supply same-field source-distance weights, population/contamination model,
  shared group/zero-point covariance and absolute calibration. Published fits
  to overlapping objects cannot become independent priors. Distinguish a
  defensible conditional experiment from the full joint catalogue posterior.

Stop this source candidate if only accepted/PV rows are available, raw release
identity is unresolved, usable rejected measurements cannot be established,
or split isolation cannot be maintained. A successful identity probe advances
the source-specific likelihood decision, not automatic target implementation.
Do not repeat the closed SDSS parent search or its completed algebra tests.

Q-GOAL: the alternative must constrain the same NEW evolved field, not a
separate map. Subsequent MW/M31 role ambiguity and unresolved M33 must be
constrained there at <=0.3 cMpc/h; truth IDs remain calibration/evaluation only.
Q-LEAN: one exact released archive and a falsifiable source decision. No large
simulation, new sampler, broad catalogue campaign or stage-by-stage approvals.

## Identity-only execution and bounded training-status extension

CPU job453857 (a10/syn01,128 MiB,3 s,exit0) recovered 11,102 distinct 6dFGS
and 2MASS identities. All rows have31 whitespace-delimited columns, matching
the supplied schema. The nested source is `campbell_table8.ascii`,2,433,077
bytes, SHA256 `c93783da45e924215b35cc6fdfb4243768a4d30301886ab787d70cf1aef8741e`.
Measured Python peak31.64 MiB. No flags or numerical FP/velocity/distance
values were parsed. Results/identities are under
`/gpfs/kjhan/CF4/z0_density/r2_6df_source_probe_20261107/453857/`.

Complete the same feasibility question by using exact 2MASS names and the
existing secure CF4/count crossmatch. Read J selection/PV flags ONLY when
the existing point and CF4-group roles are both training and the entire
native group is safe. Unknown, ambiguous and non-training connections seal
their whole native group; propagate that seal through shared CF4 groups
with the existing closure routine. No flag chooses the inspected subset.
All unlinked/conflicting rows remain in the identity artifact and stay sealed.
This deliberately restricted diagnostic must NOT become the native science
cohort or a selection-probability denominator. No numerical FP values are
parsed even for these training rows. The frozen split/target is untouched.

This metadata extension uses NumPy and the existing graph routine; expected
peak<=200 MiB, request256 MiB (>20% headroom), one CPU and the same two-minute
cap. Record whether any final residual-cut failures actually survive in these
training-linked raw rows. A null result in this restricted subset cannot prove
the full raw table lacks rejected measurements.

### Completed source decision — job453861

Source commit `24a2adc9fb4a761ad682c35aee28de60c90a9238`; a10/syn01,
one CPU,256 MiB,COMPLETED/0:0,3 s. The metadata operation took0.575 s with
80.91 MiB process peak. Its pinned inputs matched the recorded archive,
v6 split and existing crossmatch/catalogue hashes.

Of11,102 identities,4,007 had direct unambiguous training links. Closing the
new native group connections conservatively left3,001 inspected rows in
2,834 native groups;8,101 rows remained sealed. The inspected rows contain
2,973 J-PV inclusions and28 exclusions. Exactly one inspected row has
`sampcodeJ=111110`: its final residual criterion failed while its other five
FP-fit criteria passed. The raw record remains in the released table.

This verifies **at least one** training-linked residual reject's record
retention, not complete rejection history, numerical measurement usability,
selection probability or representativeness of the3,001-row subset. Neither
the28/3,001 fraction nor the single residual rejection estimates the native
source's rejection rate. Numerical FP values/errors are still unparsed;
their finite/usable status has not been asserted. No held-out selection
flags or outcomes were summarized. No old split or live target changed.

Artifacts:
`/gpfs/kjhan/CF4/z0_density/r2_6df_source_probe_20261107/453861/result.json`
and `identities.csv`. The JSON's `published_reject_retention_is_row_verified`
means this existence check only, not all rejected rows. The original table
header refers to “Table6” despite the official table8 filename and31-column
schema; retain that literal metadata rather than silently repairing it.

Driver decision: the bounded source-access question has a positive answer
and is closed. Continue a **source-specific model decision** using this
actual pre-final-cut release; do not launch another access/census loop or
automatically adopt the diagnostic crossmatched subset as the science sample.
The next required bundle must resolve identity/held-out ownership for the
native (including rejected/unlinked) population, bind the2014 native
measurement/error/coordinate conventions, and determine which earlier
selection stages can be conditioned on or normalized with source evidence.
If those dependencies remain unsupported, do not call the resulting model
selection-calibrated. The source-distance kernel, shared/absolute calibration,
field inference and validation still have to be delivered. R2 is incomplete.
