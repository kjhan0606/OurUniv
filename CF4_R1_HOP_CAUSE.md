# Host instability attribution — 2026-09-16

User requests causes, not another resolution escalation. Reuse361459 tags,
per-particle densities and common-ID particle streams. No HOP rerun, new
simulation, selection of seeds or modification of existing data.

For all473 AMR9 hosts, retain unmatched/nonreciprocal and below100-particle
counterparts. Reproduce the previously reported reciprocal>=100 samples.
Bin by AMR9 count100–299/300–999/1000–9999/>=10000; no bin is certification.
Exactly partition N_A-N_B into particles exchanged with ungrouped material
and other groups. Count crossings of the actual outer density80 threshold.
Flag substantial secondary overlap>=20% and nonreciprocity as descriptive
topology signals, not proof of physical mergers. Threshold/peak effects and
physical evolution can jointly change membership; attribution is not unique
identification of a force/integrator defect.

Hold member sets fixed in both solvers. Decompose the observed mean-velocity
vector difference symmetrically into motion and member-selection terms;
verify exact vector closure. Retain fixed-member velocity RMS and physical
sigma. Pure-motion/pure-selection regression controls run in Slurm.

Q-GOAL: distinguish unstable host boundaries/identity from dynamics before
connecting real MW/M31/M33 observations to a generated state. Those identities
remain unknown; HOP host mass is not M200c and a merged satellite is not a
validated bound M33. No automatic rejection of low-mass hosts or selection of
a conveniently stable LG. Q-LEAN: one read-only CPU diagnostic and concise
cause report, no framework or repeated threshold tuning. Resource estimate
<=5GiB plus20%=6GiB,1CPU/15min, JSON only/no new snapshots or GPU allocation.

## Completed362824: attribution and limitations

Sourcefbc70cf, Slurm COMPLETED0:0/69s, peakRSS684512K. Result:
/gpfs/kjhan/CF4/r1_hop_cause/job_362824/result.json. Original pair mass
differences reproduce; exact member-count and velocity-vector identities pass.

For reciprocal pairs with both groups>=100 particles, numbers with >20%
mass difference by AMR9 particle-count bin:

| Reference comparator |100–299|300–999|1000–9999|>=10000|
|---|---:|---:|---:|---:|
| CIC |37/148|22/162|10/104|0/15|
| TSC |25/148|14/161|4/103|1/15|
| AMR8 |20/146|7/164|1/105|0/15|

Thus39/44 TSC and27/28 AMR8 large differences occur below1000 particles.
This association does not prove Poisson noise alone is the cause, nor justify
discarding the objects. Unmatched/nonreciprocal and below100 counterparts
remain in full rows and are not included in the above paired denominator.

Across all reported TSC pairs,169935 member exchanges are with ungrouped
material,26881 with other groups.169508 exchanges cross the outer density80
threshold. In the44 unstable pairs the two exchange totals are13667/13144;
13643 are density crossings. These are gross counts summed over pairs, NOT
unique particles or fractions of net mass error. Most ungrouped exchange is
directly associated with density-boundary crossing, but regroup topology is
important in the unstable subset. No inference that changing80 fixes gravity.

Concrete counterexample to a low-particle-count-only diagnosis: AMR9 group8
has26035 particles,1.8456e12 Msun/h HOP mass, versus TSC group9/19602 particles.
Its +32.82% mass difference is exactly6433 particles:5071 net exchange with
other groups and1362 with ungrouped material. Symmetric COM-velocity
attribution norms are19.07km/s (membership) and4.33km/s (fixed-set dynamics).
These vector norms are not additive. This is a host-scale identity/boundary
instability, not just a100-particle cutoff effect. The descriptive20%
secondary-overlap flag misses this distributed exchange (largest secondary
only11.53%); retain actual exchange counts rather than treating that flag as
a complete split/merge detector.

Dynamics also differs when membership is held fixed: median COM dynamics
term3.40km/s for CIC,1.71 for TSC,.743 for AMR8; in unstable subsets4.92,3.85,
4.02 respectively. Membership terms are1.42,1.13,.837km/s overall and3.44,
4.13,1.92km/s in unstable subsets. Do not attribute all discrepancies to HOP.
No single solver is established as truth and no force/timestep/discreteness
cause is uniquely separated by these archived endpoint comparisons.

Driver conclusion: boundary membership changes plus occasional distributed
group reassignment amplify solver differences, especially at low particle
count; real fixed-member kinematic differences remain. Next useful work is
a common, explicitly defined host-centered enclosed-density/mass/velocity
readout on these same states, alongside—not replacing—HOP membership and
identity ambiguity. It must test whether the high-mass discrepancy is mainly
boundary definition before spending on higher force resolution. No threshold
tuning, automatic particle-count exclusion, MW/M31/M33 identification, M33
boundness claim, R1 closure or new production follows from this diagnosis.
