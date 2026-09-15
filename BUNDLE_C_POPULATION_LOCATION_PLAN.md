# Population-weighted local role law: one approved comparison

2026-09-12. User approved proceeding with the recommended next bundle:
use the existing population/field catalogue, multiple eligible role labels,
a SMALL density-anchored probability model, and separated spatial evaluation.
This is a request to Fable5 for advisory plan review before execution. Please
give an actual verdict with Q-GOAL, Q-LEAN, feasibility and essential/deferred
changes. All necessary facts are below; no tools, filesystem inspection or
promise to read files is needed. Driver verifies and disposes of advice.

## Destination and evidence

Actual CF4+galaxies+LG -> z=0 density/velocity posterior -> compatible LCDM IC
-> forward LG zoom. Surroundings1–2 cMpc/h, LG<=.3 (native target .1875).
No direct-CF4 IC fallback, no global .3 resolution requirement. The observed
high-resolution posterior, complete member mass/COM law q_S and field prior
q_F remain missing. This bundle supplies only a candidate POSITION law.

Prior run347007 completed6500 updates/2h36m33s.13 fixed native triples,
72,417-parameter cubic-equivariant U-Net: training joint NLL .0000387,
development63.0029 vs geometry24.8393/density10.5120. All4 tests and before/
after48-symmetry checks pass. Autonomous positions fail too. Generalization/
overconfidence failure established; data size alone is not a uniquely proven
cause. No extension of that model. Old run/checkpoint/results are preserved.
Driver accepts a design weakness: a13-field profile-extraction subset was
carried into a center-only task which could use the existing full catalogue.

Native source: TNG100-1 z0 total moments7x400^3 FP64, box75 cMpc/h, cell .1875;
existing full SubhaloPos catalogue; archived51,971 eligible triples/5,601
distinct observer-role IDs across satellite and separate-primary branches.
Use ONLY archived branch0 (M33 satellite of M31), not the rejected independent-
primary branch. No new snapshots, particles, profile reads or simulations.
Source cut: two different FoF primaries, M200c2e11..5e12 Msun; primary DM>=1000;
third bound mass1e9..1e12, DM>=1000, stars>=100, <=half M31 host M200c;
MW–M31 separation200kpc..3Mpc and M31–M33 50kpc..1.5Mpc physical at z0.
Archived rows originally exclude triples crossing x=50cMpc/h. Guarded new
slabs below keep their observer cores far enough from this old cut. Keep the
archive eligibility explicit as E_archive, not actual MW-observer selection.

## Data construction and split

Label native CENTER cells, never mass peaks. Load only needed catalogue
positions plus already materialized total moments. No profile-size restriction.
For each eligible MW ID retain ALL distinct eligible M31 IDs, and ALL distinct
M33 IDs for each pair. No uniform weighting over triple rows: each observer
has weight1/N_W; within observer each M31 has1/N_A(W); within pair each M33
has1/N_T(W,A). Duplicate source rows must not duplicate an object's weight.
Mapped-center collisions retain multiplicity/weight, not unique-cell masks.

Local prediction cube64^3 at unchanged .1875, side12cMpc/h. Observer is the
center of its native1.5 coarse cell, at local fine-face coordinates(36,36,36).
Core origin=(floor(x_W/1.5)*8-32) mod400. Min face margin is5.25cMpc/h;
source pair separation bounds plus observer offset fit inside it. Verify ALL
target centers inside, fail rather than silently drop a difficult member.
Different context from old128^3 pilot; compare only simultaneously recomputed
references, never its old NLL values as matched evidence.

Fixed x slab faces in native cells[0,184,272,400]: train, calibration, test.
Retain observer only when its ENTIRE core plus4-native-cell feature halo lies
within one slab, including the periodic x boundary. This includes both field
features and the core-wide density normalization, not just label locations.
Do not choose slabs from scores or retry splits. Training and calibration
cores have their centers safely below the old x50 source cut, test above it.
In TEST also exclude feature footprints overlapping former development cubes
16/17/20 (old lower origins[37.5,48,1.5],[55.5,10.5,12],[55.5,60,31.5], side24).
Require >=100 training, >=30 calibration, >=30 test distinct observers; if
unavailable report data-scope inconclusive without relaxing selection/splits.
Report all counts, dropped-boundary counts, companion multiplicities, and
native IDs shared between splits (must be zero). Within-split overlap is
expected; never call thousands of observers independent universes. These
are new comparison cases in ONE repeatedly used box, not external validation.

## Small field-only model (21 fitted coefficients plus one calibration scalar)

Three normalized factors q_W(c|F,O), q_A(c|F,O,c_W), q_T(c|F,O,c_W,c_A).
All64^3 cells supported; MW/M31/M33 may share cells. At inference neither
catalogue IDs nor true parents/candidate locations enter the model.
For each role q_density proportional to exp(-r_last^2/(2*s_role^2)) *
(1+M_cell/mean_core_mass), widths .75/1.5/1cMpc/h around O/W/A respectively.
This positive reference is a POSITION probability measure, not physical mass
added to F. Never describe its cells as identified bound halos.

Learn q_theta proportional to q_density*exp(theta_role dot phi_role).
Five scalar local fields: M, periodic3^3-mean M, periodic9^3-mean M,
sqrt(trace directional physical variance/3), and |V-V_massweighted_3^3|.
Density features tanh(log1p(M_scale/mean_core_mass)/5); velocity scalars
tanh(value/300km/s). Add tanh(distance/3cMpc/h) to O and each preceding cell
center:6/7/8 bounded features,21 linear coefficients. Derive velocity features
from native momentum/second moments; retain the full native source. These
scalar model features do not replace the physical seven-moment field.
Periodic cubic filters and scalar contractions preserve48 signed permutations;
the velocity features are bulk-boost invariant in populated neighborhoods.
No CNN, convolutions learned from13 patches, absolute xyz feature or catalogue
information in inference. Fresh theta=0 reproduces q_density exactly.

One convex penalized expected conditional-log-score objective, optimized with
Adam lr.02 for12 shuffled observer epochs, batches8, strictFP32. Ridge
.1/2 * sum(theta^2); fixed settings, no sweep/early-stop model selection.
Within EACH observer visit: W label; expected A score over ALL A alternatives;
expected T score over EVERY pair's ALL T alternatives. Apply hierarchical
weights above, mean over3 roles, not -log(sum q over acceptable labels), which
would permit collapse to one alternative. All admissible archived alternatives
contribute every epoch. Normalize by actual last-batch size.
Initial/final theta and optimizer saved; report endpoint and incomplete budget
honestly. Prediction probability normalization and simple weighted-target/
gradient/no-oracle/symmetry regressions run in the SAME allocation.

Calibration fits ONLY one scalar alpha in[0,1] for all roles:
q_cal=(1-alpha)*q_density+alpha*q_theta, each factor normalized. Minimize the
same observer-weighted score on calibration targets, with endpoint0/1 included;
no test use (NLL numerical ties within1e-10 prefer an endpoint,0 first).
This controls reliance on learned deviations; alpha0 explicitly
means no learned contribution, not a passing correction. Also report raw model.

## Fixed endpoint and goal boundary

Score ALL eligible test observers and all their alternatives exactly with the
hierarchical weights. Report per-observer/per-role and aggregate NLL for raw,
calibrated, geometry and density laws. Additional M33 shared-cell reference:
pi*delta(c_T=c_A)+(1-pi)*q_density_T; pi=(N_W*p_shared_train+1)/(N_W+2), where
p_shared_train is observer/pair/third weighted. These are smoothing pseudo-
counts, not independent-triple Bayesian confidence intervals.
Choose eight test observers by a fixed seed BEFORE looking at model scores;
draw32 complete autoregressive triples each for calibrated AND density laws.
Freeze samples before truth comparisons. No native-parent substitution. Use
expected distance to the weighted native candidate set, not nearest-truth
selection, and report per-role maps/spread/shared-cell rates as descriptive
one-box evidence, not coverage. Teacher-forced proper scores are not autonomous
identification. Numerical consistency tested before/after fit on one fixed
training field/all48 signed symmetries, not an orientation ensemble at inference.

Adoption feasibility only if calibrated test joint NLL beats geometry AND
density, and conditional M33 beats density AND shared-cell, with alpha>0;
show each role and observer, no M33 waiver or retuning after a miss. Report raw
model as well; a small score gain never certifies actual LG or independent
coverage. Failure closes this bounded data/model comparison; no auto-extension.

Same-field route if useful: L_pos(d|F,O)=sum_c q(c|F,O,E_archive)
int_cell p(d|x,offsets)p(x|c,F)dx. Marginalize role assignments and subcell/
stellar-halo offsets; never use q(S|F,d) as an independent data likelihood.
Existence/observer selection, mass/COM velocities, q_F and actual-data inference
are still REQUIRED before scientific use. No LG-on/off field map or IC claimed
by this position-only job. Next bundle must address that connection, not an
indefinite identification/score optimization series.

## Resources / Q-LEAN

ONE Slurm job:1 GPU/2 CPUs/10GiB host/2h, a40,a100,h100,h200 excluding syn06.
Estimated host peak<=8GiB +20% rounded to10GiB: five400^3 FP32 feature volumes
1.19GiB; streaming FP64 mass/momentum intermediates and accumulators roughly
3–4GiB plus runtime/catalogue buffers. No full7-moment duplicate plus cache of
all observer cubes. GPU feature volume1.19GiB plus local64^3 maps; envelope8GiB.
Feature preparation and actual fit timing occur in this allocation; learning
cap70min/application110min. No filesystem/inode/rename/process-scan checks.
The110min application cap INCLUDES preparation,70min-capped learning and
all evaluation; these are nested limits, not70+110 minutes. Empty-cell
velocity scalars are zero; periodic footprint exclusion applies in ALL axes.
Fixed split counts are checked before native full-field feature construction.
Persist compact source/weights/metrics,21-parameter checkpoint and256 paired
draw sets; no duplicated raw snapshot or large probability-volume collection.

Please focus your advice on: Q-GOAL contribution toward SAME-field LG
reconstruction, Q-LEAN proportionality, statistical role weights/calibration,
spatial leakage and feasibility. Identify essential errors rather than adding
new audit stages. User approval covers this recommended bundle, not a route
change or automatic observed posterior/IC run.

Advisory verdict and driver corrections: `BUNDLE_C_POPULATION_LOCATION_REVIEW.md`.
