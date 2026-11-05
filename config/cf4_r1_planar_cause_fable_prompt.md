Read-only advisory review of an important OurUniv/CF4 dynamics finding.
Do not edit files, run calculations/commands, scan directories or inspect credentials.
Read ONLY these files if needed:
- /home/kjhan/BACKUP/CF4/CF4_R1_PLANAR_DIAGNOSIS.md
- /home/kjhan/BACKUP/CF4/scripts/cf4_r1_planar_diagnosis.py
- /home/kjhan/BACKUP/CF4/src/cf4_r1_particle_forward.py
- /home/kjhan/BACKUP/CF4/src/cf4_r1_particle_resolution.py
- /gpfs/kjhan/CF4/z0_density/r1_planar_diagnosis_v4/job_359206/result.json
- /home/kjhan/miniconda3/envs/circle/lib/python3.11/site-packages/pmwd/gravity.py
- /home/kjhan/miniconda3/envs/circle/lib/python3.11/site-packages/pmwd/nbody.py
- /home/kjhan/miniconda3/envs/circle/lib/python3.11/site-packages/pmwd/scatter.py
- /home/kjhan/miniconda3/envs/circle/lib/python3.11/site-packages/pmwd/gather.py
- /home/kjhan/miniconda3/envs/circle/lib/python3.11/site-packages/pmwd/pm_util.py

Goal: actual CF4/galaxy/LG-conditioned present density/velocity posterior, surroundings
1–2 cMpc/h, LG map<=.3, compatible LCDM history and phase-consistent zoom IC ensemble
for MW/M31/M33 formation. User ratified latent-IC forward Bayesian joint-state/history
inference, with z=0 marginal first delivery. Real LG identities/assignments, bound M33,
multi-resolution gravity and independent collapsed-structure validation remain missing.
No truth IDs may identify members in generated fields. These named plane controls and
Gaussian apertures are NOT MW/M31/M33 objects or a real-data posterior.

Job359206 completed seven controlled evolutions on Slurm in70s, source977852c.
All share n128 particles,12 cMpc/h, plane k-index4, final linear amplitude .3,
a=1/64 to1, max delta-a1/256. Analytic continuum Jacobian minimum .7.
Independent flat-Lambda growth quadrature vs PMWD D and dD/dlna max relative4.345e-6.
Fourier-mode PMWD2LPT initial displacement/peculiar velocity matches analytic formula
to1.12e-17 cMpc/h /4.99e-15 km/s. Original13.57/82.68% errors reproduce.

Case: initial force relative RMS; final velocity relative RMS:
PM force128: .092713; .135679
PM force256: .577285; .826752
PM128 half-force-cell shifted particle lattice: .007070; .043961
PM256 half-force-cell shifted particle lattice: .047934; .256351
Transversely averaged density + axial PM256: .234911; .301575
Same axial PM with modes>=particle Nyquist removed: .047057; .063148
Exact sheet force, unchanged PMWD KDK factors:1.59e-13;2.88e-14.
Half-cell changes quadrature/mesh alignment while sampling the SAME physical plane.
Exact sheet force (3/2)Om*(x-q-mean(x-q)) uses known q ONLY for this analytic control.
No production code/PMWD installation was changed, and no filters were adopted.

At .001 times initial displacement, PM128 nodal force gain .99358684 but residual
after gain .0951809; halfcell gain .99358685 and residual2.98e-6. For PM256 nodal
gain1.074502/residual .616936; halfcell gain1.030601/residual3.69e-5.
Provisional driver interpretation: spatial particle-CIC/FFT/gather and lattice-phase
alias effects, not growth normalization, reference sign/velocity unit or timestep,
cause this benchmark discrepancy. This is not proof that all PMWD realizations have
83% errors, nor a unique line-level software defect, nor a claim full particle-gravity
continuum calibration is done. Smooth aperture scores masked large trajectory errors.

Possible mechanism to scrutinize, not a proven derivation here: CIC at exact mesh
vertices is directionally nonsmooth. In 1D, deposited first-order perturbation from
displacements u contains a central-difference term in u plus a second-difference term
in |u|; cell-centered particles avoid that |u| term for infinitesimal displacements.
Finer force than particle spacing also resolves lattice aliases. Projection/filter
responses support this interpretation but do not justify a production low-k cutoff.

Please deliver a concise substantive verdict (<900 words):
1. Is the localization justified? Any formula, unit, comparison, implementation or
   causal-inference defect that could invalidate it? Distinguish fact vs hypothesis.
2. Recommend ONE lean next corrective experiment, no production patch yet. Consider
   consistent higher-order assignment/gather, mesh interlacing or a tested alternative
   gravity backend; avoid ad-hoc amplitude fixes, simply longer chains, blindly finer
   mesh, or arbitrary permanent high-k suppression. Identify what must be tested for
   differentiable real three-dimensional LG inference versus what can be deferred.
3. Q-GOAL: does this work enable the stated CF4/LG/zoom goal, including unresolved
   MW/M31/M33 identification? Q-LEAN: is the next experiment proportionate?
Do not demand global .3 resolution or waive M33. Do not assert your advice is executed.
