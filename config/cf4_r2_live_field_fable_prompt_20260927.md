Read-only advisory audit. Do not edit files, build, submit jobs, run Python,
run simulations, or scan processes/filesystems. Read only the specified files.
Current project is OurUniv /home/kjhan/BACKUP/CF4, not another project.

Review CF4_R2_LIVE_FIELD_BUNDLE_20260927.md, src/cf4_r1_particle_forward.py,
src/cf4_z0_physical_field.py (read_centred only),
src/cf4_r2_joint_calibration.py, src/cf4_r2_fp_group_marginal.py,
scripts/cf4_r2_fp_group_marginal_control.py, and installed PMWD
/home/kjhan/miniconda3/envs/circle/lib/python3.11/site-packages/pmwd/scatter.py
and particles.py (from_pos/gen_grid). Do not request further numerical jobs.

Important implementation finding: native scatter has offset0 but historical
source/group likelihood reads used origin_fraction.5. Verify whether this is
a half-cell coordinate error; flag assumptions, not just agreement. The new
implementation will read native PM meshes at origin0 and regenerate the
state-dependent kernel/density measure each time. No global observer shift.

Ultimate goal: actual CF4+galaxy z=0 density/velocity posterior first,
surroundings1–2cMpc/h, LG<=.3; same-state MW/M31/M33 observational constraints,
compatible LCDM histories and phase-consistent zoom ICs. The small current
N128 source-mark-only HMC target is explicitly partial/development: no full
posterior, calibrated group inclusion, or LG claim; count/point dependence
and source/FoG covariance remain unresolved. Previous empirical offsets are
not reused as priors. No true component IDs choose generated candidates.

Answer Q-GOAL and Q-LEAN explicitly; address MW/M31 role ambiguity, unresolved
M33 and same-NEW-field observables. Is the bounded coordinate repair plus
two short IC/nuisance chains justified now, or is there a substantive target/
algorithm error requiring correction? Essential versus deferred corrections
only; do not create a ladder of prerequisite micro-audits. Distinguish
implementational feasibility from scientific calibration. Verdict PASS,
CONDITIONAL PASS, REJECT or NO VERDICT, with concrete evidence. <=900 words.
