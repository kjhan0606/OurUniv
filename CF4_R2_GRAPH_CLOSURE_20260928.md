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
