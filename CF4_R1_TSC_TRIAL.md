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
