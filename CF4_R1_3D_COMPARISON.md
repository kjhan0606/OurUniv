# R1 fixed3D comparison and independent-solver connection — 2026-09-14

User authorizes the next priority after the partial TSC correction. The
existing independent-solver requirement is NOT waived. This bundle first
executes the immediately reusable same3D-state comparison, while a current
verified RAMSES executable/input contract is located. No routine new audit.

## Independent reference: read-only discovery and unresolved contract

The old CF4 RAMSES launchers point to
`/home/kjhan/BACKUP/lagRamses-de-nonstd/build_lb_minimax/ramses_lb_minimax3d`
(2026-07-30); the documented `bin/ramses_final3d` is dated2026-08-17. These
are not confirmed to contain the later run-safety fixes. The inspected
`patch/lagRamses/amr_step.jaehyun.f90` still triggers periodic output at
coarse step0. Do not call this old source the verified current implementation
or launch a historical namelist. No external-project source was modified.
User was asked for the current verified executable path without blocking the
in-scope3D comparison. Rebuilding/patching another project is not silently
substituted for a science comparison.

The inspected GRAFIC reader supports explicit displacement plus independent
velocity, but the existing CF4 exporter does not yet provide a checked exact-
state handoff. RAMSES creates particles at cell centers whereas these PM
states start at grid vertices; the offset must be accounted for, not ignored.
Preserve original128³ displacement/velocity files for that connection. No
RAMSES evolution, independent force accuracy or new source-build is claimed.

## Frozen same-state work executable now

Use all three archived128³ ICs from359203, including their interpolated32³
mode content, unchanged box12 cMpc/h/cosmology/mass. No new high-k or selected
seed. Four arms: CIC force256/max delta-a1/512 (reproduce archived particle
endpoints); TSC force128 and256 at1/256; TSC force256 at1/512. Same PMWD KDK
factors, no amplitude/force-window compensation or phase shifts.

Report particle positions/velocities, Gaussian probe observables, density
readout band power/cross-correlation/difference at common.1875 cells, and
force differences at IDENTICAL initial and archived final particle positions.
Comparisons with CIC are differences, not errors against truth. Fixed readout
bands in h/cMpc: (0,pi/2],(pi/2,pi],(pi,pi/.3],(pi/.3,pi/.1875]; these are
reporting scales, not information-frontier declarations or imposed force cuts.
Report TSC time-halving particle differences separately. Four focused tests,
including a small8³/3D trajectory-gradient finite-difference comparison, run
before these twelve evolutions. Passing that test does not validate a large
production trajectory adjoint or an independent physical solver.

Resource request1GPU/4CPU/10GiB/30min (application25min), current allowed GPU
partitions including a100_pcie, exclude syn06. Estimated host ceiling8GiB
(previous1.75GiB plus paired states, compiler and small-field readouts),
+20% rounded10GiB. Bound particle/force artifacts to about1GiB uncompressed,
not a series of full AMR snapshots. R1 used5246/14400 GPU-seconds before this
job; even the full1800s remains within the approved allocation cap.

Q-GOAL: determine whether the plane artifact materially affects the3D
observation predictions before choosing a dynamics backend for actual CF4/LG.
Q-LEAN: reuse initial states, observables, conservative readout and diagnostic
KDK; no new framework/HMC or another assignment-order sweep. MW/M31/M33 must
ultimately be identified in generated states with role ambiguity and bound M33
handled; these three Gaussian probes do not identify them or supply their
observational likelihood. R2/science promotion remains held.

Outputs: `/gpfs/kjhan/CF4/z0_density/r1_three_dimensional_v6/job_JOBID/`;
logs `/gpfs/kjhan/CF4/logs/cf4_R1_3d_JOBID.{out,err}`.
