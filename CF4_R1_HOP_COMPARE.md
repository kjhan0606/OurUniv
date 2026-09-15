# R1 generated-host comparison — 2026-09-15

User approves the next substantive structure readout. Reuse all four archived
seed2026091301 endpoints, no new simulation/seed selection or external edits.
Original HOP/regroup binaries from user-specified lagRamses source directory;
record executable hashes, use fixed neighbors64/64/4 and density thresholds
80/200/240 in normalized periodic box1 with total particle mass1.
Export all four states in the same initial-ID order to a minimal DMO HOP
particle stream, validate positions/velocities/mass by round-trip. These
files are NOT RAMSES restart snapshots. HOP internally uses float positions;
catalogue moments are recalculated from original double-precision particles.

Report groups>=100 particles with group mass, periodic center and mean
velocity; retain tags for every particle, including smaller groups. Match
AMR9 to CIC/TSC/AMR8 by shared particle identity, with both overlap fractions,
reciprocity, top-two candidate counts and unmatched/nonreciprocal outcomes.
Do not suppress changes by selecting only nice nearest-position matches.
100 particles is an explicit reporting cut, not a convergence guarantee.

Q-GOAL: determine whether stable field probes hide unstable generated host
properties before applying LG observations. Q-LEAN: existing HOP/regroup plus
one readout and overlap match, no new halo finder, mesh sweep or simulation.
MW/M31/M33 are not preassigned. Unconditioned fixture need not contain an LG;
host group mass is not M200c. HOP does not establish bound/deblended M33.
Role ambiguity and unmeasurable M33 must remain explicit in the next LG
operator design; identity matching across solvers is not observational
identification, and no truth catalogue is supplied to candidate selection.

CPU-only Slurm1core/6GiB (estimated5GiB+20%)/100min; HOP20min per arm,
regroup1min each. No GPU, snapshots or new evolution. Four small particle
streams plus HOP diagnostics estimated<1GiB. Preserve previous/new artifacts.
Matching label-permutation regression and input round-trip run in allocation.
Routine driver evaluation; no new external audit or automatic R1 promotion.
