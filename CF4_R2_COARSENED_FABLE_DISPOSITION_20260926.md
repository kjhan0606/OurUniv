# Fable5 advice and driver disposition — R2 coarsened observation

Read-only Fable5 CLI (`--model fable`, plan mode; Read/Glob/Grep only)
returned **CONDITIONAL PASS** for the *development direction*, not for a
joint likelihood or posterior. It considered Q-GOAL aligned and Q-LEAN
proportionate: retain the voxel-count factor from the ARES/BORG source,
preserve individual redshifts for later conditional CF4 marks, and stop
count-only score variations. The driver independently checked the advice
against source and numerical results; it is not automatic authority.

Adopt:

- The source paper uses redshift-space voxel counts with voxel-integrated
  target/angular selection, so an individual-point process is not essential
  to an N128/N256 density factor. The count factor is a normalized Poisson
  PMF for supplied nonnegative expected counts; Slurm405676 establishes
  finite support for one homogeneous actual-data control, not field adequacy.
- The *observed crossmatch table* is reproducibly constructed from catalogue
  identifiers/angles/redshifts, so its deterministic edge bookkeeping can
  be kept fixed as an observed design **conditional on both catalogues**.
  But this is not a proof that CF4 distance-group inclusion is ignorable.
- Explicitly state the redshift-space mapping: observed 2M++ `Vcmb` becomes
  a cosmology-dependent radial coordinate in
  `scripts/cf4_r2_common_catalogue.py`; the field-to-count development
  operator in `src/cf4_r2_continuous_tracer.py` applies spherical RSD and
  a separate FoG/redshift scatter. Its calibration remains missing.
- N256/1.5 cMpc/h is the R2 surroundings target, after the observation
  model is defensible; N128/3 is only feasibility. A point inside a
  positive-exposure cell does not create 0.3-cMpc/h LG information.

Reject or qualify:

- Fable claimed the CF4 group redshift is a deterministic function of the
  secure matched 2M++ individual redshifts and advised not modelling or
  scoring it. This is **contradicted** by the source audits: published CF4
  group `Vcmb` comes from a different velocity-member process than the
  CF4 distance contributors; the archived Tully(2015) group velocities
  differ from current CF4, and the simple 2M++-member conditional failed
  heldout coverage. One may condition on observed group redshift as a
  covariate in a *narrower* distance likelihood, but cannot call it a known
  deterministic function or drop its uncertainty without a bias analysis.
- Fable suggested calibrating the CF4 group-inclusion/Malmquist term on the
  native CF4 holdout. A selected-only train/holdout split does **not**
  identify the probability that a potential group receives a distance
  measurement; the denominator/parent population or an external/mock
  selection model is needed. Holdout can test predictions *among selected
  groups*, not identify missing groups.
- The 15,239 edges split into 14,878 secure, 14,360 native-CF4 and 14,026
  secure-and-native edges; these are distinct predicates, not inconsistent
  totals. Twenty-eight 2M++ points have two CF4 edges, seven involving
  different group IDs. The eventual mark law must marginalize ambiguous
  links or use an explicit fixed subset with selection consequences; no
  first-edge collapse.
- Fable called all 442 catalogue/map disagreements deferrable at the cell
  level. Their pointwise zero no longer blocks the *count factor*, but
  mismatched source products remain a selection-model sensitivity before
  scientific posterior promotion.
- Fable's MW-at-observer and smooth-aperture M33 suggestions are not an
  identification law. Each **new** evolved field must generate MW/M31/M33
  hypotheses without native truth IDs; keep MW/M31 ambiguity and an
  unresolved M33 branch. A coarse field cannot certify bound M33 or its
  motion. Its observed direction, distance, mass and velocity must constrain
  that same locally resolved state through a calibrated role/observation law.

Decision: accept the coarsened-count *development* route, not the Fable
recommendation to discard the CF4 group-redshift process or use selected
holdout as an inclusion denominator. Next work must tackle source-backed
CF4 group inclusion and conditional distance/velocity marks under the same
forward state. No new N256 sampling, posterior or zoom is licensed by this
advice. Q-GOAL remains direct; Q-LEAN forbids more homogeneous-count tests.
