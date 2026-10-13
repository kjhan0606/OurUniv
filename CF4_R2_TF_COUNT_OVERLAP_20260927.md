# R2/5 — TF-only group and 2M++ count-point ownership

The all-method likelihood multiplies a redshift-space galaxy-count factor
by conditional TF distance marks. Before interpreting this as a properly
factored joint observation model, measure how many TF-only CF4 groups have
**secure observed** links to eligible 2M++ count points. Reuse the existing
source-bound crossmatch edge manifest, without inventing membership for
unmatched groups. This identifies direct observed-data dependence, not a
parent-selection denominator, physical independence, or a covariance fit.

One small typed-H100/2CPU/4GiB/10min Slurm source check while the bounded
all-method sampler runs. Q-GOAL: tests a direct dependency of the same-field
CF4+count likelihood needed for R2. Q-LEAN: a single exact crossmatch
intersection and ownership report; no new galaxy matching, gravity or fit.

MW/M31/M33 remain unresolved R3 roles extracted from each NEW generated
field. They cannot be seeded from native truth or assumed inside g(F), and
their observed quantities must eventually constrain that same field.

## Result and decision

Typed-H100 **406492 COMPLETED/exit0**. The pinned v1 result is
`/gpfs/kjhan/CF4/z0_density/r2_tf_count_overlap_v1/result.json`.
Of8,502 TF-only groups,3,608 have any eligible 2M++ point edge, and3,458
have a secure edge. Secure training groups number2,761 of6,745;697 are
heldout. There are3,686 secure edges to3,686 distinct eligible count points.
Thus at least40.9% of TF training groups have explicit count-datum
associations. Unlinked groups are not proven disjoint from the redshift
survey, nor a missing-distance denominator.

This overlap does **not by itself prove double counting**. A normalized TF
mark conditional on a known observed point can factor from a Poisson count
under explicit within-cell/mark-selection assumptions. The current factor
conditions on CF4 group cz, not the matched point and its possibly different
cz, and uses a provisional selected-density radial law. Its equivalence to a
count-conditioned group mark has not been shown. A v2 source check measures
the actual matched redshift offsets before deciding whether a simple
singleton conditional is tenable.

Typed-H100 **406493 COMPLETED/exit0**; pinned v2 result is
`/gpfs/kjhan/CF4/z0_density/r2_tf_count_overlap_v2/result.json`.
Across3,686 secure TF-only edges, the absolute CF4-group minus 2M++-point
Vcmb difference has median9, p90=115 and p99=420.45 km/s;61.53% are within
20 km/s and92.67% within150 km/s. A unique matched singleton therefore
often has a near-identical observed redshift, but the tail makes literal
equality or an exact one-to-one velocity-data identity unjustified. This
source comparison is not physical peculiar-velocity/FoG calibration.

The bounded all-method HMC job406491 was **cancelled by the driver after
2m18s** while still an unassessed partial-target computation. The saved
partial result/logs are preserved; no HMC result or posterior is claimed.
This was a Q-LEAN decision to address the explicit observation dependency
before spending up to3h on a target that may need to change, not evidence
that a count×TF likelihood is intrinsically invalid.
