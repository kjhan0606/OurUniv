# R2 external native-K response bundle — 2026-10-02

R1 complete; R2 actual z=0 density/velocity posterior remains incomplete at
N256/1.5 cMpc/h in384 cMpc/h. R3 LG <=0.3, R4 precise forward validation and
R5 zoom IC follow. Previous finite-LF regression410445 passed36/36 and cost
profile410446 completed; those successes do not supply tracer calibration.

## Purpose and available evidence

Legacy CAMELS stellar-mass sextiles have no valid mapping to five active
true-K responses. Existing TNG100-1 snapshot99 group catalogues contain
`SubhaloStellarPhotometrics[:,3]` (native K), positions, velocities, flags and
stellar particle counts. A one-file schema read confirmed those fields and
the448-file,4,371,211-subhalo catalogue. The existing total-matter source has
7x50^3 native moments at1.5 cMpc/h in the75-cMpc/h box, h=.6774 and Om=.3089.
No raw snapshot scan, download or new simulation is required.

[Official TNG specifications](https://www.tng-project.org/data/docs/specifications/)
define the photometric order U,B,V,K,g,r,i,z and subhalo velocity as all-matter
mass-weighted COM km/s. The physical absolute-magnitude interpretation is
supported by [the TNG team's clarification](https://www.tng-project.org/data/forum/topic/431/stellar-photometrics/).
Native K is not automatically calibrated 2MASS Ks. Passband, dust and aperture
correspondence remain unresolved; this calculation cannot yield an immediately
usable external bias prior. The h convention is explicitly M_h=M-5log10(h).
The [official photometric model documentation](https://temet.tng-project.org/source/temet.util.html)
confirms UBVK are Vega magnitudes and native K uses the IR K filter with
Palomar200 detectors/atmosphere. The unresolved issue is the K-to-Ks
passband/galaxy-light correspondence, not an unknown AB/Vega offset.

## One executable bundle

Read catalogue fields from448 fixed files, selecting physical-flag objects
with stellar particles and finite non-sentinel magnitudes. Preserve stellar
particle counts rather than impose an arbitrary mass threshold. Save compact
native K/position/velocity/calibration-ID arrays. For the five active true-K
proxy bins, include empty cells in a rate-profiled Poisson rho^beta response
fit, normalized over the full native box. Fit only x<45; buffer45–52.5;
evaluate x>=52.5 cMpc/h. Record boundary fits, population support, test
count prediction/score and velocity residuals relative to native cell mean.
Report counts with>=1/100/300 stellar particles as resolution evidence.
One known-response/rate control runs before source extraction.

This is evidence about an external galaxy/matter law. Native NGP density,
hydro physics,75-cMpc/h volume, cosmology and all-matter galaxy COM differ from
the R2 PM/observational law. Spatial test regions share the same universe.
Driver will assess whether those limits permit a useful calibration follow-up
or require a different source. No coefficients enter the active prior here.
The CF4/2M++ heldout outcome is untouched.

Q-GOAL: addresses the missing luminosity-defined density/velocity tracer
response upstream of the first actual z=0 posterior delivery. MW/M31 role
ambiguity and unresolved M33 remain for R3; their observables must constrain
those roles on the SAME NEW field. Native IDs are external-calibration labels,
never generated-field seeds or candidate selections.
Q-LEAN: one native catalogue pass plus the preserved coarse moment grid;
no new gravity, raw snapshot, neural training or validation framework.
Routine bounded source work uses driver review under the active audit policy.

Host peak estimate<=~3GiB from chunk arrays, concatenated selected galaxies
and small grids; request4GiB provides>20% margin. Two CPUs,20min Slurm.
Check H200/H100/A100 modes and submit one compatible typed GRES.
Output: `/gpfs/kjhan/CF4/z0_density/r2_tng_native_k_response_20261002_v1/`.
Scripts: `scripts/cf4_r2_tng_k_response.py` and corresponding `.sbatch`.

Submission and terminal science assessment will be appended from Slurm/artifacts.

Source246f036 submitted as410483 at2026-10-02 15:25:01 KST. Initial
PENDING(Resources); chosen mode A100 after checking h200/h100/a100.
`scontrol` confirms `TresPerNode=gres/gpu:A100:1`,2 CPUs,4GiB,20min.
Static AST, shell syntax and diff checks passed. The numerical known-response
control and real source extraction run inside this Slurm allocation.
No scientific calibration result exists at submission.

410483 FAILED/exit1 after3s before source extraction because the Syntax-visible
`/scratch` source was absent on the allocated node. The known-response control
passed first; no native fit or scientific result exists. The repair copies only
the fixed catalogue fields (~150MB) to one shared HDF5 file, under a120-second
I/O-only Syntax bound; all calibration and numerical field work remain Slurm.
Retry writes a new v2 directory. This repairs input availability without
filesystem diagnostics or changing the scientific model.

The I/O-only stage completed448/448 files,4,371,211 rows,146,432,411 bytes.
Source78c1d83 retry410484 completed/exit0 in3s on A100/syn101. All five proxy
populations have support:119/637/3181/4993/318394. Fitted native betas are
1.792/1.364/1.204/1.136/1.035; residual1D RMS148/184/205/207/217km/s.
These are external native-field conditional estimates, not R2 priors.
Bright bins0–3 are all resolved by>=300 stellar particles, while only40,223
of318,394 faint-tail objects have>=100 (24,764 have>=300). Pooling that entire
tail does not calibrate its actually selected luminosity range. In the faint
heldout cells with expected>=5, Pearson4947.2 over2536 cells combines model
mean errors and scatter; it does not identify a unique stochasticity cause.
The known-response/rate identity passed. Full results are in the v2 directory.

One same-source v3 extension now slices the faint tail into one-magnitude
intervals -21 through -16 plus the remaining tail, comparing>=1 and>=100
stellar particles with the SAME spatial split and response estimator. It
tests luminosity/resolution dependence before treating pooled faint bias as
relevant evidence. No new field, observations or model prior; same20min/4GiB
typed A100 job, new v3 directory. Direct model adequacy/calibration remains open.
