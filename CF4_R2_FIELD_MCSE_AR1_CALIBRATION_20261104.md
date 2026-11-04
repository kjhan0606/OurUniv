# R2 field-MCSE AR(1) calibration — 2026-11-04

## Scope

This is a synthetic calibration of the batch-means estimate for a posterior
mean, not a CF4 target evaluation, chain-convergence result, or spatial-field
test. It uses four independent stationary Gaussian AR(1) chains with 512
retained draws each. For each coefficient, 5,000 independent four-chain
replicates were generated with seed `20261104`. The estimator pools each
chain's non-overlapping batch means and divides their sample standard
deviation by the square root of the total number of batches; the denominator
is the pooled sample SD of all retained scalar draws. The exact reference is
the finite-length stationary AR(1) variance of the pooled mean.

The candidate scales are 32 and 64 states per batch (16 and 8 batches per
chain, respectively) with a candidate threshold of MCSE/posterior-SD `<=0.10`.
This explores the maximum 512 retained draws in the conditional-v6 proposal;
it does not select production settings. `scripts/cf4_r2_mcse_ar1_calibration.py`
reproduces the table.

## Results

| AR(1) \(\phi\) | Exact pooled ratio | 32-state estimate median / p95 | 64-state estimate median / p95 | Both point estimates `<=0.10` for a scalar replicate |
|---:|---:|---:|---:|---:|
| 0.00 | 0.0221 | 0.0220 / 0.0252 | 0.0219 / 0.0266 | 100.0% |
| 0.50 | 0.0382 | 0.0372 / 0.0424 | 0.0375 / 0.0452 | 100.0% |
| 0.70 | 0.0525 | 0.0501 / 0.0565 | 0.0510 / 0.0613 | 100.0% |
| 0.90 | 0.0954 | 0.0813 / 0.0881 | 0.0885 / 0.1025 | 90.84% |
| 0.95 | 0.1353 | 0.0984 / 0.1041 | 0.1158 / 0.1294 | 5.56% |
| 0.98 | 0.2090 | 0.1129 / 0.1169 | 0.1457 / 0.1563 | 0.0% |

The 64-state scale is more sensitive to slow correlation than the 32-state
scale. For \(\phi=0.95\), the exact pooled ratio is 0.1353; the 64-state
estimate has median 0.1158, and only 5.56% of scalar replicates put *both*
point estimates below 0.10. At \(\phi=0.90\), the exact ratio is 0.0954 but
the 64-state estimate's replicate p95 is 0.1025. Thus the candidate p95 rule
would conservatively reject some chains near the threshold, while using only
the smaller batch scale would hide more of the slow correlation.

The reported p95 values are quantiles over scalar simulation replicates. They
can approximate cellwise tail behavior only for a hypothetical field of many
independent, identically mixed cells; actual map cells are spatially
correlated and have heterogeneous autocorrelation. The single-cell pass
fraction is not the pass probability of the production regional-p95 gate.
The simulation is also stationary, unimodal, and Gaussian, unlike an
unqualified R2 chain. It therefore cannot certify a field-level threshold.

`MultiScaleFieldMCSEAccumulator.merge_independent_chains_inplace` now combines
complete equal-length chain sufficient statistics in-place, including the
between-chain contribution to pooled batch-mean variance. Sixteen focused
tests passed; the fixed-seed AR(1) test matches the pooled mean-MCSE formula at
both scales. The helper has no file-based runner/report integration, and no
N256 resource or I/O profile has been measured. Chain-specific disagreement
must still be recorded separately before the destructive merge.
Keep 32/64 and eight batches per chain as test candidates only; do not freeze
them as the R2 production contract. The candidate `0.10` threshold remains
unaccepted pending the pooled implementation, spatially correlated
calibration, actual chain diagnostics, and a defensible precision/cost rule.

## Disposition

This closes a narrow estimator-feasibility question: at the 512-state cap,
using a larger batch scale is materially more informative than 16/32 alone,
but even 32/64 is not a universal MCSE calibration. It does not authorize a
sampler, held-out read, or production GPU calculation. Observation
selection/group calibration remains a separate blocker; R2 remains NO-GO.

**Q-GOAL:** this supports the uncertainty methodology needed for the
conditional z=0 density/velocity posterior but creates no posterior or field.
MW/M31 remain role-ambiguous and M33 unresolved; their observables must
constrain those same roles on the same NEW evolved LG field at
`<=0.3 cMpc/h`, with native truth IDs reserved for calibration/evaluation.

**Q-LEAN:** one CPU-only seeded simulation and a compact reproducibility
script; no CF4 target evaluation, held-out access, new data, mock archive,
GPU calculation, or expanded diagnostic ladder.

## Next technical work

Wire the in-memory merge to persisted/replayed per-chain sufficient statistics
and a file-based four-chain report, after costing the storage-versus-replay
tradeoff and preserving independent-chain disagreement separately. Calibrate
the production path on spatially correlated synthetic maps with heterogeneous
autocorrelation before choosing batch lengths or promoting the `0.10`
candidate. Keep all target/held-out data sealed and R2 NO-GO until observation
scope and the full resource/MCSE contract are defensible.
