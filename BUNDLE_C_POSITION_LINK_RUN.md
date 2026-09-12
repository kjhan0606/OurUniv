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

Status: implementation prepared; static checks and source-pinned submission
next. No numerical success claimed before execution. Autonomous approval
allows a substantive next design after outcome review, not false promotion.
