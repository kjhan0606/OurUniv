# R1: isolated consistent TSC force trial — 2026-09-14

User approves continuation after the cause diagnosis. This implements the
driver-checked Fable recommendation in CF4_R1_PLANAR_DIAGNOSIS.md, not a new
external review, sampler extension or installed/production PMWD patch.

The trial uses quadratic TSC weights on27 neighbors, identical for deposit
and force gather, with JAX differentiation of these actual weights. CIC's
8-neighbor stencil/custom backward is not reused. FFT Poisson, zero/Nyquist
conventions, physical modes and KDK remain unchanged; no deconvolution,
gain adjustment, interlacing or permanent high-k suppression. TSC is a
standard assignment scheme, not a new cosmological prior; see
[Abacus documentation](https://abacusutils.readthedocs.io/en/latest/tutorials/analysis/tsc.html).
Unlike a density power estimator, this trial must test the entire force pair.

Six frozen evolutions: original nodal CIC128/256 reproduce359203, TSC128/256
at node and half-cell alignment.128³ particles,12 cMpc/h, single mode4,
final linear amplitude.3, a1/64→1, max delta-a1/256. Compare all TSC results
with the corresponding archived CIC359206 values, including the tiny initial
force/gain/shape residuals, not only integrated aperture averages. Reuse the
existing diagnostic driver/config interface; compact histories only.

Three focused tests run FIRST in the SAME Slurm allocation: weight/mass/
periodicity and deposit-gather transpose identity; self/net force and mesh
translation; full-force position derivative versus finite differences at
nodes AND half-cell stencil boundaries. This is not a trajectory-adjoint or
generic3D accuracy certificate. Numerical failure aborts, not a threshold
relaxation. Interpretation: quantify improvement and any degradation in
every alignment; no automatic production promotion based on a plane result.
Remaining spatial error must be reported even if the CIC kink disappears.

Resources:1GPU/4CPU/10GiB/20min, a100_pcie,a40,a100,h100,h200, exclude syn06.
The previous diagnostic's host peak was1.68GiB. Conservative8GiB expected
ceiling allows compiler workspaces, three256³ force channels (.375GiB),
FFT buffers, particle states and bounded32768-particle27-neighbor chunks;
+20% rounded to10GiB. This is a sizing estimate, not measured new usage.
Application18min plus test/startup headroom; R1 allocation5178/14400s before
this job. Even a full1200s allocation remains within the cap. No new snapshots.

Q-GOAL: accurate differentiable gravity is needed to connect actual CF4/LG
observations to the same evolving state. Q-LEAN: one assignment-order trial,
not a backend search or another HMC campaign. MW/M31/M33 identification from
generated state, role ambiguity, bound M33 and observation/model calibration
remain unsolved; planar labels are analytic controls, never oracle halo IDs.
This trial cannot by itself produce an actual present-field posterior or
close R1. Any production correction also needs generic3D/trajectory-gradient
and independent nonlinear accuracy evidence.

Files: config/cf4_r1_tsc_trial_v5.json, src/cf4_r1_tsc_diagnostic.py,
tests/test_cf4_r1_tsc_diagnostic.py, scripts/run_cf4_r1_tsc_trial.sbatch.
Outputs: /gpfs/kjhan/CF4/z0_density/r1_tsc_trial_v5/job_JOBID/
and /gpfs/kjhan/CF4/logs/cf4_R1_tsc_JOBID.{out,err}.

Source bc9dede committed/pushed. Slurm359341 submitted and RUNNING on
syn103/a100_pcie2026-09-14. Numerical tests/evolution outcomes pending at
submission; inspect the fixed359341 artifacts, not a process-scanning loop.

## Completed result and driver judgment

Slurm359341 COMPLETED/exit0 on syn103/a100_pcie,2026-09-14
18:13:53–18:15:01 KST,68s allocation. Three tests pass in5.781s; diagnostic
58.81s. Full-force derivative scaled finite-difference error max4.10e-7
(denominator max(1,|AD|,|FD|), not a universally relative error). Original
two CIC endpoints reproduce. All six evolutions complete, no numerical failure.

| Force mesh / lattice alignment | CIC velocity relative RMS | TSC velocity relative RMS |
| --- | ---: | ---: |
|128³ / nodal |13.5679% |2.9335% |
|256³ / nodal |82.6752% |6.3113% |
|128³ / half force cell |4.3961% |2.4238% |
|256³ / half force cell |25.6351% |14.1480% |

CIC half-cell values are preserved job359206 references, not rerun in359341.
These are particle trajectory/velocity errors, not Gaussian-aperture scores.
At tiny initial amplitude, TSC128 nodal force shape residual is5.61e-8 versus
CIC .09518, while the TSC gain is.988814 (not1). TSC256 nodal shape residual
is1.28e-5 versus CIC .61694, gain1.018081. This confirms that a consistent
smooth assignment pair removes the identified leading CIC corner artifact,
without rescaling the cosmological power or deleting modes.

**Partial correction, not accuracy closure.** Every tested final alignment
improves over CIC, but the force256 results still depend strongly on alignment
and remain worse than force128. In particular14.15% cannot be hidden by
reporting only the best6.31% case. Remaining finite-displacement lattice/mesh
errors and assignment-window response have not been uniquely decomposed or
calibrated. Neither the two meshes nor these two shifts bound generic3D error.

Driver accepts the isolated TSC implementation as a candidate for further
evaluation, NOT as a production inference kernel. No need for another routine
external audit of this expected, already-advised comparison. Do not start R2,
lengthen HMC, arbitrarily choose the best lattice phase or blindly refine the
force grid. Next substantive evidence should be a fixed3D same-state comparison
and an independent force/evolution reference, including the used spatial bands
and trajectory gradients, before choosing a science backend. This is a next
priority, not a new job or an implemented independent solver. The previous
particle64→128 aperture sensitivity15.81% is also not closed by this plane trial.
No generated-state MW/M31/M33 identification or actual CF4/LG posterior yet.

Measured diagnostic resource peak: process-reported host1.749GiB, device peak
1959983360 bytes; Slurm batch MaxRSS1243536K (different collection method).
R1 cumulative allocation5246/14400 GPU-seconds; remaining9154s (2h32m34s).
Results: `/gpfs/kjhan/CF4/z0_density/r1_tsc_trial_v5/job_359341/result.json`
and six compact history files; fixed log paths use job359341. No active
numerical job remains from this trial; production PMWD was not changed.
