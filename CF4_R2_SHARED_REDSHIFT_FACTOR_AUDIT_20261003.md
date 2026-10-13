# R2 shared-redshift factor audit — 2026-10-03

Status: the current likelihood code is a development collection of valid
numerical components, not the configured real-catalog joint observation law.
No field fit, posterior, production claim, or gravity simulation was run.

## Finding

`config/cf4_2mpp_joint_likelihood_v1.json` declares that the 2M++ count and
CF4 group-mark factors share redshift latents. The current NumPy/JAX
`joint_log_likelihood` entrypoints instead add a Poisson score at a supplied
intensity to a separately marginalized group-redshift Gaussian score. The
intensity predictors accept no group IDs or shared-redshift realization.
The source-bound manifest proves object/group identity and blocks a second
standalone 2M++ redshift factor; it does not prove the probabilistic dependence
or conditional factorization.

Astra's read-only audit confirmed a genuine contract/implementation gap, with
an important qualification: the additive sum is valid for a model whose count
and mark factors are conditionally independent, or when the mark term is
already the correctly normalized conditional given count-owned observations.
The current interfaces do not establish either condition for the configured
2M++/CF4 observation law. Therefore this is not evidence that the Gaussian
covariance calculation itself is wrong, nor proof of a measured-data double
count; it is a missing derivation and coupling.

The bounded secure source graph has 14,878 eligible edges in 10,393 CF4 groups;
no eligible 2M++ recno occurs in more than one CF4 group. This supports
group-local blocks for this subset, but does not establish physical membership,
spatial independence, covariance calibration, or completeness. The existing
group-conditional sample has 9,754 groups. Student-t4 improves heldout mean
log score over a Gaussian in both one-member (-5.755 vs -6.150) and
multiple-member (-6.372 vs -6.610) strata, but singleton 90%/95% coverage is
only 0.820/0.861. This conditional is selected-subset diagnostics, not a
calibration of the shared-latent variance used by the likelihood.

## Minimal correction made

The API/config now explicitly label the additive kernel as development-only
and state that identity ownership does not establish probabilistic
factorization. A small NumPy reference primitive evaluates a mark conditional
on count-owned data under one shared latent:

```text
log p(mark | count, field)
  = logsumexp(log p(latent) + log L_count + log L_mark)
    - logsumexp(log p(latent) + log L_count)
```

A two-member fixture compares that expression with direct finite integration,
demonstrates that separately marginalizing then multiplying can differ, and
checks the no-information identity. These tests verify the math primitive
only; they do not supply a real count model, selection function, latent
covariance calibration, or production likelihood.

## Next R2 action and limits

The next required implementation is a source-aware training control in which
the same group latent affects both the count-owned observation kernel and its
two-member marks, with the normalized conditioning denominator explicit.
Use the actual 2M++ count/source kernel and declared selection; do not replace
missing empty-sky angular completeness, redshift/FoG covariance, or mock
calibration with trial values. If those inputs remain absent, keep this as a
blocked design/control and do not fit or promote an R2 posterior.

Q-GOAL: this closes a necessary statistical definition for CF4-to-z=0 field
inference, but does not produce or validate the field.

Q-LEAN: one formula and one two-member reference fixture; no repeated
simulation or parameter sweep.

MW/M31 roles remain ambiguous and M33 unresolved. Their observables must
ultimately constrain those same latent roles in the NEW evolved field at LG
resolution `<=0.3 cMpc/h`; native truth IDs remain calibration/evaluation-only.
