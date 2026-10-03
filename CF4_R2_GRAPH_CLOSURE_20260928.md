# R2 source-graph closure — 2026-09-28

## Purpose and scope

Build a prospective train/heldout split closed under observed links among
2M++ count points, count voxels, 2M++ GIDs, Tempel parents, CF4 groups and
SDSS-PV source groups. The old v5 geometry octant is retained as the initial
holdout. Any connected component that mixes train and heldout, or touches an
existing boundary buffer, is buffered in full. Since counts are aggregated
by population/voxel, a buffered point also buffers its entire count key. This
is source-identity leakage control, not a new likelihood.

The exact NYU `lss_combmask.dr72.ply` was not recovered from a trusted source.
The original NYU-VAGC endpoint redirects HTTP to HTTPS, where strict TLS
verification fails on this node. Exact-name searches found no verifiably
identical mirror. No TLS bypass or approximate random-neighbour mask is used.
The Tempel denominator route is closed descriptively; this does **not** close
R2 or validate the field-selection law.

## Correction to the previous mixed-parent count

The full saved Tempel/2M++ role array has six parents with linked training and
heldout members. Five happen to lie inside the now-withdrawn proxy-mask and
redshift window used for descriptive strata; the sixth lies outside that
proxy window. Thus “five” was not the full source-graph count. This does not
rescue v5: an expanded Tempel-parent likelihood is not graph-closed under any
of the six cross-role parents.

## Frozen inputs and algorithm

The implementation reads only the preserved v5 role arrays and source
identities: the 2M++ point catalogue, reciprocal Tempel-member/2M++ links,
secure CF4↔2M++ links, CF4↔FP anchors, SDSS-PV/Tempel member IDs, and existing
point-to-CF4 edges. It does not read heldout distance marks, evaluate field
scores, choose a field, or access native MW/M31/M33 truth labels.

Each source object is a graph node. A disjoint-set closure assigns components:

- train-only components remain train;
- heldout-only components remain heldout;
- train/heldout mixed components become buffer;
- any component touching an existing buffer becomes buffer.

The final count keys must partition all 57,238 observed parent points exactly
into train, heldout and buffer. `train_window_excluded_keys` and
`heldout_window_excluded_keys` record where buffered observations must also
be removed from the corresponding Poisson exposure window. No threshold is
tuned to preserve a preferred sample size. The reciprocal Tempel↔2M++ bridge
is only an observed catalogue association used conservatively for leakage
control; it is not a physical-membership or selection probability.

The code has four focused disjoint-set regressions and static compilation
checks. The single pinned-source H100 Slurm job requests one CPU, 6 GiB RAM
and 15 minutes; no gravity, sampler, field fit, or N256 calculation is part
of this bundle.

## Science questions and limits

- **Q-GOAL:** this repairs prospective heldout separation needed to assess the
  CF4+galaxy present-field target. It does not infer density/velocity or add
  observational information.
- **Q-LEAN:** one graph closure over existing IDs, no new mask estimator,
  random threshold sweep, mock archive, gravity run or sampler.
- **MW/M31/M33:** no component identities seed or select candidates. On a new
  evolved field, MW/M31 remain ambiguous latent roles and M33 remains
  explicitly unresolved when unsupported; their observables must constrain
  that same field. Native identities remain evaluation-only.

The R2 posterior remains NO-GO until source/selection, tracer and shared
covariance assumptions are defensible, and an actual field fit has stationary
uncertainty and untouched heldout predictions.

## Execution result

Typed-H100 Slurm **407069** completed in 6 seconds and wrote
`/gpfs/kjhan/CF4/z0_density/r2_sky_closed_split_v6/`. The output projects all
57,238 source count points exactly into 47,121 train, 8,475 heldout and 1,642
buffered rows. There are 37,951 occupied training keys, 6,946 heldout keys,
and 879 buffered keys. The 1,642 buffered points comprise 421 formerly
training and 1,221 formerly heldout rows; 243 of the buffered keys came from
the old training side and 636 from the heldout octant. Thus the original
9,696-point count holdout is reduced, not silently replenished.

The closure saw 148,577 graph nodes and 136,561 edges, yielding 42,257
components; 27 components directly mixed train/heldout labels, and 74
components were buffered after propagating either a mixed role or an existing
boundary buffer. This also moves 131 CF4 and 36 FP source-group roles to
buffer. Exact Tempel source-object matching has zero group-ID conflicts, and
the full count projection assertion passed. The six mixed Tempel parents
were independently reproduced; five lie in the withdrawn proxy-mask/redshift
window and one does not.

Jobs 407066, 407067 and 407068 failed before output creation on, respectively,
a short submitted commit pin, an unjustified hard-coded total-row expectation
for the complete local 2M++ catalogue, and a misnamed archived NPZ field.
Those operational/code errors were corrected; their failures and logs are
preserved. They are not scientific evidence. The successful source and all
input SHA-256 hashes are frozen in the v6 result JSON.

This is a conservative split under the *observed association graph*, not proof
that the 10-arcsec/300-km-s-1 Tempel↔2M++ matches are physical identities, and
not a new likelihood. The 1,642-point buffer costs data and could be
conservative. No heldout values or scores were read. No field fit, sampler,
posterior, map, gravity run or N256 calculation ran. Use v6—not v5—for any
prospective fit that includes this expanded graph; adapt its Poisson exposure
window to exclude `train_window_excluded_keys` and
`heldout_window_excluded_keys`.

**Next R2 work:** reuse the frozen v6 training roles for the existing scalable
linked-source factor and then complete the missing normalized point/group
selection and shared-covariance ownership. The conditional factor results are
still only source/forward mechanics until those terms are calibrated and a
stationary actual-field fit is evaluated on the untouched v6 holdout.
