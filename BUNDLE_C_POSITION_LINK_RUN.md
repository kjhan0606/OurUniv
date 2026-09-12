# Same-field position-likelihood bridge

2026-09-12 autonomous continuation after user approval. Plan and driver
disposition: BUNDLE_C_POSITION_LINK_{PLAN,REVIEW}.md. Fable5 returned PROCEED;
the driver replaces alias-prone GH quadrature with cell-face segmentation,
analytic Gaussian exponential tilting and deterministic rectangle integration.
No added trained model or new data. The frozen21 coefficients/alpha are from
population_locations_v1/Slurm347085/sourceaa024b4.

Implement src/cf4_position_link.py, the matching runner/config and3 focused
regressions, all numerical work IN one Slurm allocation. Checks cover native
scalar-feature parity, all seven parent moment sums, Gaussian rectangle/
Jacobian controls, shared cells, and interior conservative-direction gradients.
This is not another location-model fit. Eight already-frozen test observers,
all their archived alternatives under hierarchical weights, actual LG distance/
sky data with stated development approximations, and one projected field
sensitivity map. Saved native density is NOT reconstructed local density.

Lpos is a joint density per MW Cartesian volume, both solid angles and both
moduli, conditional on archive eligibility. M31/M33 correlated distance errors
are integrated; contributions MW, marginal M31|MW and incremental M33|MW/M31
data are reported separately. Uniform within-cell positions, fixed solar
position and zero additional stellar/halo offset are development assumptions.
No mass/COM velocity or observer/existence selection law is supplied, and no
already-data-conditioned mark posterior is multiplied again.

Slurm1GPU/2CPU/6GiB/30min; estimated host4.5GiB+20% rounded6GiB. Application
25min. No manual syn101, process/storage probes, new snapshots, particle
extraction, density painting/optimization or IC. Output new
/gpfs/kjhan/CF4/z0_density/bundle_c_v1/position_link_v1/.

Source1a7af09 committed/pushed. Slurm347086 COMPLETED/exit0 in14s at19:55:17
KST on2026-09-12; three tests pass,26 native triples/208 field cross-scores,
all8 actual observation interfaces evaluated. All native feature comparisons
agree to<5.96e-8 scaled error. Three conservative derivatives pass against
both finite differences; parent moment errors<=5.91e-14. No learning step.
Application8.78s, host1.269GiB, GPU reserved.463GiB. Nevertheless retained
INCONCLUSIVE_POSITION_LINK_NUMERICS because the original error estimate used
sum(rectangle error) / sum(qA*qT*rectangle probability), replacing all qA*qT
by the worst-case1 in the numerator. This valid but very loose worst-case
estimate is not evidence of an inaccurate likelihood; actual16/32-replacement
tolerance comparisons had identical logL in all8 cases.

Correction: propagate each adaptive-integration error with its OWN frozen
qA*qT factor. Keep the unweighted worst-case estimate visible and the genuine
clipped-tail bound separate; quad error estimates are NOT formal certified
bounds. Same1e-4 estimated-error threshold, model, measurements and tolerance.
Add an analytic constant-cell-probability error-weighting regression to an
existing test. Run the same tiny comparison once, new position_link_v2 output,
preserving v1; compare all actual/mock logL to v1 and reject changes>1e-10.
No refit, scientific gate relaxation or additional audit stage.

Planning correction from direct old-source inspection: the old FIELD diffusion
already used dense translated native crops; it was the failed MEMBER learners
that had13 fields. More observations alone is not new evidence sufficient to
revive the closed diffusion line. Any next field-prior design must address
its actual objective/representation/conditioning limitations explicitly.
