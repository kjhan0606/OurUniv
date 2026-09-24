# CF4 / OurUniv — active master plan

Effective 2026-09-07. The user approved registering the goal-alignment review
as the highest-level project plan and proceeding with short, substantive
bundles. This file supersedes CODEX_PLAN.md, PLAN.md, the Hong restart plan,
and cf4_science_route_v3.json as execution/priority authority. Preserve those
files and previous results as history; do not silently revive their routes.
Direct subsequent user instructions take precedence over this file.

## Latest execution update — 2026-09-23

Seed40349 remains a **TRACE-ONLY zoom candidate**, not a promoted parent.
GalaxyFinder resolves an L9 pair, while the frozen-threshold HOP regrouping
merges its two distinct raw peaks; M33 is unresolved.  The zoom work therefore
tests whether one conditional fine realization can preserve the coarse LG
candidate, not whether the parent or MW/M31/M33 system is already validated.

The old halo-member-only trace was replaced by a spatial environment trace.
Grammar Slurm1108300 selected all1886 parent particles within5 cMpc/h of the
z=0 pair midpoint, retained all143 pair members, traced them to a
14.13x11.21x14.09 cMpc/h initial footprint, and built a buffered L9 sparse
mask with a42-cell enclosing cube.  Seed40349's measured L9 transfer was
continued with an LCDM-shaped EH98 high-k tail: overlap shape scatter0.10%
and Nyquist join ratio1.00315 (Slurm1108316).  These modes above the parent
Nyquist have random conditional phases and are not claimed as CF4-recovered
information.

Syntax Slurm398944 generated the bounded L9-L12 IC hierarchy with an extended
48-byte GRAFIC header, explicit omega_b=0 DMO metadata, finest particle
spacing0.09375 cMpc/h, runtime ceiling L19 and fine seed403495108.  Grammar
reader gate1108348 and no-output two-step runtime gate1108357 both completed:
138632600 particles, Morton mismatch0, base FFT and all fine MG solves pass,
maximum reported fine residual7.437e-5<1e-4, no boundary/OOM/fatal marker and
no full snapshot.  The two-step gate reached a=0.02455 and used48.72 GiB peak
RSS under a64 GiB request; it did not yet refine above L12.

The bounded early-nonlinear sequence is now complete. Grammar1108467 evolved
normally to a=0.05 but the fail-closed wrapper rejected the gate because no
level above L12 was populated; classify the result as epoch-too-early, not
numerical instability. The first restart attempt
1108478 exposed a missing formal INIT_PARAMS block before evolution and wrote
no output. Corrected same-rank restart1108481 then completed a=0.05->0.10 in
4m42s: transient maximum L17, fine residual<=9.706e-5, no boundary/OOM/fatal/MG
failure,48.89 GiB peak RSS, and one20.31e9-byte final dump. Decision:
TRACE_ONLY_ZOOM_L19_EARLY_NONLINEAR_PASS. This does not promote the parent,
resolve M33, identify a z=0 MW/M31/M33 system, or validate high-k phases.

Do not call this a validated zoom or production IC.  A long z=0 L9/L12/L19
evolution has not yet produced a result.  The early-nonlinear
runtime/refinement gate has passed, and the reviewed full-z0 resource, output,
restart, and science-gate contract is now in
`CF4_S40349_ZOOM_Z0_PLAN.md`.  The actual uniform-L9 reference reaches z=0 in
about27 minutes on16 MPI ranks; the older, much larger L8/L12 mask zoom took
22h56m on8 MPI ranks.  The current L12 mask occupies21.6% of that older mask's
volume, so one 32-rank,96-GiB,48-hour restart with exactly one new final dump is
the bounded next calculation.  Runtime success still requires subsequent
GalaxyFinder/HOP and MW/M31/M33/environment/contamination evaluation.

Per the user's raw-output retention instruction, the superseded19-GiB a=0.05
dump and its historical symlink were removed after the a=0.10 successor was
validated; they are not recoverable.  Logs, namelists, hashes, and decision
records remain.  Keep the a=0.10 restart source until z=0 validation and keep
the z=0 dump until all halo catalogues and science diagnostics are sealed.

Full-z0 restart grammar Slurm1109268 ran from2026-09-23 16:20 to2026-09-24
06:14 KST on grammar089 with32 MPI ranks x2 threads,96 GiB, and completed in
13h54m03s/exit0.  The pinned Intel MPI2021.17 environment loaded, the wrapper
verified the checkpoint writer's `ncpu=32`, and RAMSES evolved a=0.10->1.0.
It populated L19, completed285 coarse steps, reported maximum fine residual
9.992e-5<1e-4, no boundary/OOM/fatal/MG-nonconvergence marker, and wrote
exactly one22.011e9-byte final dump.  Peak RSS was60.34 GiB. Decision:
`TRACE_ONLY_ZOOM_L19_Z0_FORWARD_PASS` in
`config/cf4_lg_s40349_zoom_l19_z0_decision_v1.json`.

This is numerical completion only.  The parent remains trace-only, M33 remains
unresolved, and the single random conditional high-k realization is not
CF4-recovered information. The user now directs using the newer
`GalaxyFinder/NewGalFinder` instead of HOP. Run the RAMSES NewDD/opFoF path,
then NewGalFinder's DMO density-peak/watershed/boundedness/tidal-radius
substructure analysis on output_00003 and evaluate MW/M31/M33, environment
drift, and contamination. HOP is deferred unless NewGalFinder has an
implementation-level failure. Do not promote the seed or call it a validated
production zoom before that science decision.

After the z=0 result and hashes were committed and pushed, the superseded
19-GiB a=0.10 checkpoint and its z=0-run symlink were removed under the user's
raw-output cleanup instruction; they are not recoverable.  Preserve the final
z=0 output_00003 until both halo-finder products and the science decision are
sealed.

NewGalFinder source commit586a62e was fetched without overwriting the user's
modified local GalaxyFinder tree, checked out at the separate GPFS worktree
`GalaxyFinder_newgal_586a62e`, and built successfully with the documented
opFoF ABI. The fixed binary hash and one-run resource plan are recorded in
`CF4_S40349_NEWGALFINDER_PLAN.md`. Grammar job1113459 completed the required
mass-preserving NewDD/opFoF conversion:512 DM slabs,146681 hosts, and a largest
host of165853 particles. This justifies the fixed 8-GB-per-worker allocator
build and a150-GB NewGalFinder request. The first NewGalFinder launch exposed a
DMO ABI mismatch (72-byte member records versus a168-byte hydro build) plus
two periodic-unwrapping defects; it was cancelled after54 seconds without a
usable product. GalaxyFinder branch
`agent/fix-newgalfinder-periodic-unwrapping` commit43046a3 fixes both and adds
a compile-time 72-byte DMO ABI assertion. Corrected-ABI job1113504 then passed
the early hosts and produced DMO subhalos, but was deliberately cancelled after
showing that every zero-star host still paid for an empty stellar FFT. Follow-up
commit96560d9 sends such hosts directly to the identical adaptive DMO finder
and zero-initializes the per-halo state. No HOP job is submitted.

Final NewGalFinder job1113522 completed all146681 FoF hosts on2026-09-24
21:38:44 KST/exit0. Validation1113532 failed at startup because its batch
PATH lacked `python`; the absolute-interpreter retry1114528 completed on
2026-09-25 01:52:23 KST. Its frozen LG cutflow gives0 accepted pairs:
5 eligible hosts,2 separation passes,1 mass-ratio/midpoint pass,0 isolation
passes. The closest candidate misses the3 cMpc/h isolation threshold at
2.7388 cMpc/h. Decision: `NEWGAL_DIAGNOSTIC_PAIR_NO_GO`; Virgo and Coma are
still located. The current trace-only zoom is not a validated MW/M31/M33
realization. Detailed evidence is in `CF4_S40349_NEWGALFINDER_PLAN.md` and
the preserved validation JSON.

The bounded member audit1114563 completed2026-09-25/exit0. It confirms the
isolation intruder is a distinct6.1498e12-Msun/h FoF host2.73884 cMpc/h
from the candidate midpoint. A third LG-scale host of2.5309e12 Msun/h is
only0.8532 cMpc/h from that midpoint. Both nearest-pair hosts contain
well-resolved bound M33-scale components, but there are several competing
assignments. Their FoF and bound-member particle masses show no coarse-particle
contamination. The failure is therefore a real frozen-cut local configuration
at the catalogue level; retain the strict NO-GO for this one conditional
fine-phase realization. No threshold or seed selection follows from it.
The observed-frame cross-check in `CF4_S40349_NEWGALFINDER_PLAN.md` also
finds a minimum57.57-degree MW-to-M31 sky-direction mismatch, a1.522-Mpc
pair separation versus0.761-Mpc observed M31 distance, and candidate MW
hosts3.77/4.57 cMpc/h from the fixed observer. This is an independent
geometric diagnostic, not a calibrated likelihood. Stop seed40349 work;
connect actual LG observables and latent roles to the same evolved IC state
before another zoom or parent promotion.
The first bounded observation-operator bridge now exposes
`predict_candidate` in `src/cf4_lg_observation_contract.py`: for a caller-
supplied resolved MW/M31/M33 role hypothesis it returns the eight predicted
observables **and** both angular mismatches instead of rejecting an off-sky
candidate. The original strict `predict` behavior is unchanged. This is an
interface test, not a likelihood, role enumerator, M33 certification, or an
IC-posterior update. Next, candidate support must come from the NEW evolved
state without truth IDs, with unresolved M33 retained explicitly; a calibrated
joint sky/kinematic likelihood and multiresolution forward connection are
required before an LG-on/off IC comparison can be claimed.
`src/cf4_lg_generated_roles.py` now supplies the corresponding identity-free
NewGalFinder bound-component adapter. On a predeclared generated-state local
support it enumerates every ordered MW/M31 component pair (including two
components sharing one FoF host), every distinct M33 component, and an
explicit unresolved-M33 branch; it does not select the most massive component
or best observed match. For an assigned triple it transforms the saved SG
position/peculiar velocity to the observation contract's ICRS frame and
reports the eight predicted quantities plus both sky offsets. A cap fails
without truncating support. The pure synthetic tests pass. Q-GOAL: this
addresses the missing role/observation wiring on one generated state.
Q-LEAN: no new RAMSES run or arbitrary candidate scoring. These hypotheses
have **no calibrated role prior, detection law, sky covariance or posterior
weight**, and the adapter is not differentiable IC inference. Its M33 branch
labels indicate FoF sharing, not demonstrated association with M31. Do not
call enumeration or a close observed-space match a validated LG realization.
Syntax CPU-only Slurm403323 then read the preserved NewGalFinder z=0
`GALCATALOG.LIST.00003` (6s/exit0) with the fixed 5 cMpc/h observer aperture,
without observational or mass ranking. It contains521 local FoF hosts and579
bound components. Exhaustive ordered MW/M31/M33-or-unresolved enumeration
would generate193434636 hypotheses, so the adapter correctly stopped before
materializing them. The compact result is
`config/cf4_lg_newgal_role_support_403323.json`. An initial job403318 failed
in3s because its runner supplied the particle-data file rather than the
catalogue-list file; it produced no science result and was corrected without
changing the aperture or state. This cardinality is **not evidence that the
observed LG is absent**. It establishes that a best-match shortlist or uniform
explicit triple list is not an acceptable route to posterior conditioning.
The next inference design must specify a normalized generated-state role and
detection law (including missing M33), then preserve its probabilities through
factorized summation or an importance-corrected proposal. Only after that and
the calibrated sky/kinematic likelihood can the same-state LG-on/off IC test
be performed. No extra zoom run or mass-cut retuning follows from this count.
Q-GOAL: the readout tests whether MW/M31/M33 role ambiguity can be retained
when actual generated components, rather than truth IDs, feed the observation
operator. Q-LEAN: one 29-MB catalogue read and a scalar count; no new physics
run, candidate ranking, filesystem diagnostic or validation framework.

## Latest planning update — 2026-09-13

User requested an end-to-end feasibility/implementation replan with Fable
consultations. [CF4_END_TO_END_REPLAN_20260913.md](CF4_END_TO_END_REPLAN_20260913.md)
and its [advice dispositions](CF4_END_TO_END_ADVICE_20260913.md) record that
work. Three Fable5 consultations were checked against source and literature;
their unsupported claims were not adopted wholesale.

First-scale specialist352623 completed8m21s:16 valid draws, individual1/16,
ensemble2/8; its physical failure persists. CLOSE this ML repair line. No
actual fine CF4/LG posterior has been delivered and no new calculation was
launched for this replan. Historical queued states below are superseded by
[the completion record](BUNDLE_C_FIRST_SCALE_EXPERT.md).

The new recommendation uses a joint present-state/history posterior with
latent ICs and gravity. It is explicitly forward Bayesian IC inference,
with the z=0 marginal delivered first, not a standalone z=0-then-invert
method. **User2026-09-13 explicitly APPROVED this route and R1 execution.**
The R1–R5 replan now supersedes the historical A–D implementation order and
independent-z0-first restriction below. Actual-data z=0 posterior remains
the first science delivery; no best-seed or arbitrary fine-mode substitution.
R1 starts with the bounded implementation in [CF4_R1_RUN.md](CF4_R1_RUN.md).
Its cumulative numerical budget is4 GPU-hours; no new large production.
Preserve all old results, source and failed-model limitations.

R1 entry implemented and submitted as Slurm354568 on2026-09-13 21:34:05 KST,
source8f2aa0d,1GPU/4CPU/10GiB/1h. Initial PENDING(Resources), no numerical
outcome yet. One small periodic PM/aperture/adjoint/HMC mechanics job, not
actual LG inference, nested-gravity validation, R1 closure or R2 entry.
See CF4_R1_RUN.md for the bounded scope and fixed output/log paths.
User2026-09-14 requests a100_pcie. Added it to the SAME pending354568 job's
eligible partitions without cancel/resubmit or altering its source/resources.
Include a100_pcie in subsequent applicable GPU submissions.

Update2026-09-14:354568 COMPLETED5m52s on syn103/a100_pcie,9 tests pass,
likelihood derivative error<=0.0292%. Short-chain mixing remains unestablished
(max unsplit Rhat1.1504/min ESS6.36); mesh32→64 aperture-mass sensitivity up
to40.85%, versus subsequent timestep sensitivity<=3.35%. No R1 closure.
User directs resolving these important issues. Driver implements ONE bounded
same-data HMC comparison (4 chains, fixed8 versus random16–48 trajectories,
matched expected retained gradient work) and a three-seed same-particle force/
time ladder through128^3. CF4_R1_RUN.md records interpretation/resources.
1GPU/4CPU/16GiB/2h within the remaining R1 cumulative4GPU-hour budget. No
mass/noise rescaling, threshold relaxation, actual LG inference or R2 launch.
Physical MW/M31/M33 identification and verified local/coarse dynamics remain
required; successful aperture numerics do not supply them.
Follow-up source32ae8c5 pushed; Slurm358369 started2026-09-14 11:26:08 KST
on syn103/a100_pcie with the above bounded resources. Numerical outcomes
pending; follow the fixed result/logs in CF4_R1_RUN.md, not historical jobs.

358369 subsequently COMPLETED1h17m08s/exit0 at2026-09-14 12:43:16 KST,
tests8/8. Both48-summary mixing criteria pass: fixed8 max Rhat1.0095/min bulk
ESS820 versus random16–48 1.0129/419; no warmup/retained divergences. Keep
adequately sampled fixed8 for the small target; fine-model mixing unverified.
Force differences decrease, but64→128 still changes probe masses up to10.4%,
so accuracy is not closed. R1 consumed1h23m of4GPUh. User approves the next
priority: same interpolated initial displacement/velocity with32³→64³→128³
particles at fixed force128³, then force256³ and timestep halving. One1GPU/
4CPU/16GiB/1h Slurm job, no added high-k or new sampler. Two pre-crossing
plane-wave analytic controls check spatial evolution against known trajectories.
Exact initial/final finest states and common.1875 readout maps are retained;
no actual LG, full-power fine IC, independent collapsed-solver validation or
R1 closure. Details and limits: CF4_R1_RUN.md, config/cf4_r1_particle_resolution_v3.json.
Source62ccea1 pushed; Slurm359000 submitted2026-09-14 13:17:10 KST,
initial PENDING(Resources), with the above1h bound. Fixed result/log paths in
CF4_R1_RUN.md; no numerical pass or independent-solver calibration yet.

359000 failed at startup after5s: the driver basename shadowed its src library,
causing a circular import before numerical work. Driver renames the executable,
updates the Slurm entry and adds a module-resolution regression; same approved
comparison/resources, no physics-model change or external review. Preserve
failed logs; CF4_R1_RUN.md records recovery and subsequent job state.
Recovery90bae45 pushed; retry359203 ran2026-09-14 14:13:01–14:15:04 KST
on syn103/a100_pcie, COMPLETED2m03s/exit0,7 regressions pass. All15 numerical
cases/two planar controls finish. Probe mass sensitivity particles64→128
still15.81%, force128→2562.902%, timestep halving.1240%. CRITICAL unresolved
control: analytic planar velocity relative RMS worsens13.57%→82.68% with finer
force mesh; possible reference/setup or particle-force error must be separated,
not labelled a unique cause yet. HOLD accuracy promotion/R2 despite small
smoothed-aperture residuals and successful implementation tests. Next priority
is bounded planar-reference/initial-force/growth cause separation, not blindly
finer meshes. R1 used5108/14400 GPU-seconds; no next job launched by this record.
User now requests cause confirmation. CF4_R1_PLANAR_DIAGNOSIS.md freezes ONE
15min Slurm diagnostic: independent growth quadrature, pure-mode LPT/units,
original endpoint reproduction, initial-force/growth traces, half-mesh-cell
lattice shift, transverse/axial alias controls and exact planar-sheet force
with the same time integrator. No installed or production gravity changes,
new posterior or mesh escalation.7 controls/1GPU/4CPU/8GiB within R1 budget;
interpretation must distinguish reference error, time integration and spatial
force/discreteness effects rather than presupposing a PMWD defect.
359206 COMPLETED70s/exit0 at2026-09-14 14:36:09 KST on syn103/a100_pcie.
Independent growth/pure-mode IC checks pass and old endpoints reproduce.
Replacing ONLY the spatial force by exact planar sheets reduces velocity RMS
error to2.88e-14 with the SAME integrator; shift/projection/alias controls also
reduce errors. The benchmark discrepancy is localized to particle-mesh spatial
force, with lattice-phase/CIC sampling-alias effects supported. No universal
83% cosmology-error or unique PMWD-line-bug claim. Accuracy/R2 still held.
Details CF4_R1_PLANAR_DIAGNOSIS.md; important-finding Fable advice returned
and supports localization plus an isolated higher-order deposit/gather trial.
Driver rejects calling this single fixture a worst-case bound; no universal
3D accuracy claim or production fix. Suggested corrective experiment is not
launched in this diagnose-only turn. R1 allocation5178/14400 GPU-seconds.
User approves continuation: CF4_R1_TSC_TRIAL.md specifies an isolated,
consistent TSC deposit/gather with unchanged Fourier force/time integration.
Three focused tests and six frozen plane evolutions, one1GPU/4CPU/10GiB/20min
Slurm allocation within R1. No installed/production kernel changes, mode
filtering, real-data posterior or MW/M31/M33 identification claim. Compare
all alignments and preserve residuals; plane improvement alone cannot close
R1 or promote the actual inference backend.
TSC trial359341 COMPLETED68s/exit0 on syn103/a100_pcie2026-09-14 18:15:01 KST;
3 tests pass, six evolutions complete. Nodal final velocity RMS13.57→2.93%
(force128),82.68→6.31% (force256); half-cell4.40→2.42%,25.64→14.15%.
Leading CIC corner artifact is suppressed, but alignment/finite-amplitude
error remains; PARTIAL correction only, no production/R2 promotion. Next
priority is fixed3D and independent force/evolution evidence, not finer-grid/
sampler escalation. Details CF4_R1_TSC_TRIAL.md. No next job launched;
R1 allocation5246/14400 GPU-seconds, all MW/M31/M33 identification limits remain.
User approves next priority. CF4_R1_3D_COMPARISON.md freezes three archived
3D ICs / four CIC-TSC-time arms plus same-state force, field-band, particle
and aperture comparisons and a short trajectory-gradient test. One1GPU/
4CPU/10GiB/30min Slurm job. This is NOT the independent reference: discovered
old RAMSES binaries/source are not verified current, and exact-state input
handoff remains unchecked; user asked for current executable path. No external
project edits or historical launcher reuse. No R2 promotion on code agreement.

## Scientific destination

Use actual CF4 observations, galaxy-density observations, and explicit local
structure constraints to infer a **present-day density/velocity posterior**;
construct compatible LCDM initial conditions; evolve them forward and verify
the observed local environment and MW–M31–M33 system. Deliver phase-consistent
zoom ICs for studying LG formation and evolution, not a unique historical
reconstruction of every small-scale phase.

- Surroundings: 1–2 cMpc/h numerical reconstruction scale is sufficient.
- LG: target <=0.3 cMpc/h numerical reconstruction scale, with explicit LG
  mass, position, distance and relative-velocity uncertainties.
- Finer IC particle/force resolution is a separate zoom requirement. AMR
  alone does not improve particle mass resolution. Fine modes not fixed by
  observations remain conditional/prior content, not recovered observations.
- Retain Virgo/Coma and Local/Bootes Void environment constraints. Separate
  cluster zooms, full-volume 0.3 phase recovery, and RT/stellar/AGN/dust work
  are not prerequisites for the LG deliverable.
- Push defensible observation/structure information toward high k, especially
  in the LG region. Measure information gain; do not arbitrarily choose a
  tiny low-k domain and call the goal achieved. Conversely, global frontier
  certification must not block a useful, explicitly qualified LG prototype.

## Four outcome-based bundles

| Bundle | Deliverable | Exit decision |
| --- | --- | --- |
| A — usable present-field model | One bounded prior comparison, then an actual CF4 + galaxy-data preliminary z=0 density/velocity product with uncertainty and held-out predictions | Declare model limitations and whether it is usable; no indefinite toy-model repair loop |
| B — LG and dynamics connection | Explicit LG constraints in the z=0 route and a small mock z=0 → IC → forward-z=0 demonstration | Check LG/environment conditions and field/velocity consistency after evolution; test this before expensive high resolution |
| C — multiresolution reconstruction | Surroundings at 1–2 and LG at <=0.3 cMpc/h, with coarse/fine coupling and measured observation/structure information | Show LG-conditioned information gain, resolution/cost feasibility and residual uncertainty; recalibrate scale-dependent priors as needed |
| D — zoom IC and evolution | Phase-consistent nested ICs from buffered Lagrangian particle membership and a forward-evolved LG ensemble | Check LG masses/separations/velocities, environment and contamination, then decide on production |

The scientific order remains observations → z=0 posterior → IC → forward
validation. Bundle B is an early end-to-end mock risk test, not permission to
bypass the present-field stage or restart the historical direct-CF4 route.
Do not double-count the same data by treating a data-derived z=0 posterior as
an independent likelihood. Any approximate inversion/proposal must disclose
its target and correction before calling its output an IC posterior.

## Current state and bounded immediate work

**Latest authority2026-09-12, after347085:** user explicitly grants autonomous
continuation without repeated per-bundle approval. Historical "wait for
approval/no automatic next bundle" statements below preserve their original
scope but no longer block driver-designed in-goal follow-up. Keep bounded
experiments, Fable5 advisory planning, source-pinned Slurm, memory sizing,
commit/push and honest outcome reports. Do not resurrect a failed repair line
without new evidence or quietly switch to direct-CF4 IC generation. The
scientific destination above remains unchanged; no promise of a validated
posterior from position-score success.

**Bundle A is closed as a diagnostic delivery, not scientific certification.
Bundle B is closed as a development/diagnostic delivery. Bundle C entry is
approved and implementation is in progress. D requires user approval.**
Current design: [BUNDLE_C_DESIGN.md](BUNDLE_C_DESIGN.md).

Bundle B: environment334398 and four LG interface tests334401 completed;
bounded dynamics retry334402 passed both fixed development cases in4m58s.
Density RMS residuals0.964/0.972 became0.118/0.118; velocity333/300 became
17.8/18.1 km/s, with conservative readout. These are regularized mock IC
candidates, NOT actual-data IC posterior samples. The actual map's Local
Void and precise cluster localization remain unresolved. The source-backed
LG interface still needs a resolved fine-field operator and covariance/model
discrepancy calibration before actual conditioning. No further N32 sampling
or bridge extensions: next work must bring LG observations into the fine z=0
field while retaining its coarse mass/momentum environment. C now begins with
native observations at1.5 cMpc/h, an LG0.1875 layout and conservative coupling;
these are not yet a high-resolution posterior or an evaluated LG operator.
Native-data334408 and conservative/JAX tests passed. Selected actual inputs
contain no direct CF4 or count rows inside LG R2 cMpc/h (CF4 minimum radius
15.8152). LG observations must supply that information explicitly. Selection
integration334409 completed, but order4 missed the positive angular footprint
of one occupied population/cell. Native row positions have valid source-mask
and LF support. Geometry-only repair334508 and preservation check334512
completed: occupied zero-support keys1→0;70260 unoccupied population/cells
also repaired; all prior positive entries unchanged. The v2 selection is a
development input, NOT precision-certified:120 geometry controls still find
33 tiny-support population/cell entries missed by2048-point cubature, with
maximum tested shell-L1 absolute error9.37e-4. Preserve v1 and this limitation.
No high-resolution posterior inference is running. Next substantive C work
is the physically specified fine-field prior and LG halo/subhalo operator,
then the bounded LG-on/off comparison, not another coarse sampler extension.
See the C run record for current products and limits.

Native resolved-operator jobs334521/334522 now completed. A real TNG FoF
fixture with989815 particles supplies mass/mean-velocity/physical-dispersion
fields at0.1875 and1.5, with conservative restriction and particle/SUBFIND
mass/COM agreement. This is a simulated FoF component, NOT a reconstructed
LG or full matter map. The catalogue readout is valid for that unmodified
particle realization; arbitrary changes to fine cells cannot inherit its
halo catalogue. A full-matter conditional fine prior and actual joint LG
conditioning remain outstanding. No new high-resolution posterior is running.

The next C implementation is a finite whole-patch coarse-summary conditional
prior, plus a total-matter source including diffuse/non-FoF matter. It retains
native field/catalogue consistency and separates physical sigma_v from
posterior uncertainty in mean velocity. This finite support baseline does
NOT yet define a continuous LG posterior or exact full-parent conditioning.
Two-file timing334524 and5 regressions passed; full source job334528 submitted
(2 CPUs,9600 MiB,4h cap, estimated2–3h). It builds native400^3/50^3 moments
and18 train/9 nonoverlap check patches from existing TNG. No new simulation,
Hong retraining, actual CF4/LG weighting or automatic downstream inference.
See BUNDLE_C_RUN.md. Do not broaden prior kernels or narrow the science target
just to disguise insufficient support from a small finite patch collection.

Update2026-09-08:334528 completed03:27:20 KST in1h58m04s. All448 files,
11935938442 massive particles/cells processed; conservation passes, absolute
native cosmological mass error4.8093e-5. The18-component prior fails support
on all9 heldout targets (ESS1.00–2.06). This is not a CF4/LG posterior.
User authorized the next C step: ONE bounded support comparison using dense
native translations and24 exact proper rotations, unchanged32-dimensional
conditioning and prior bandwidth, strict heldout spatial exclusion. Report
raw, distinct-anchor and source-spatial-group concentration separately; do
not count correlated/rotated copies as independent universes. If grouped
support still fails, close finite-bank expansion and move to a continuous
joint matter/halo-model design, not repeated kernel/seed tuning.

That bounded comparison335875 has now completed:23275 native anchors and24
proper rotations (558600 correlated hypotheses). Raw component ESS rises to
231.6–2439.3, but native spatial-group concentration ESS is only1.70–5.35;
4/9 heldout targets still fail ESS>=4/max-group-weight<=0.5. Conditions and
bandwidth were unchanged; original results reproduced and the transformed
actual fine field passed its summary check. Status
NO_GO_FINITE_BANK_SUPPORT_CLOSE_THIS_REPAIR. CLOSE this finite-bank expansion.
The next C design must vary matter and halo state jointly and continuously,
with a defined diffuse component and calibrated physical/model discrepancy;
no further bank densification, kernel widening, N32 extension or fake LG map.
This result does not prove that every finite prior fails, nor that a continuous
model will automatically succeed. No actual CF4/LG fine inference is running.

Continuous-state implementation335878 now completed in24s: a native total-
matter patch with three disjoint member components supports21 continuous
position/member-mass/COM-velocity marks, coupled positive remainder mass and
momentum compensation, and correct second-moment transport. Conservation and
one noiseless same-generator inverse control pass. This is a KINEMATIC
OPERATOR, not a calibrated continuous LCDM prior, resolved transformed halo,
independent validation or actual LG posterior. Close the toy-control step.
The next substantive requirement remains the physical joint distribution of
environment, halo marks/profiles and remainder response with COM/model
discrepancy. The21 local marks cannot by themselves fix support on all32
coarse environmental features. Do not promote the precise numerical inverse
error to astronomical accuracy or restart bank expansion. Actual LG-on/off
information comparison remains undelivered; see BUNDLE_C_RUN.md.

Actual LG MARK conditioning has now run (335879,42s):51971 native two-primary/
third-object triples from5601 distinct observer candidates; separate M33
satellite and independent-primary alternatives. A continuous transformed-mark/
shell-summary distribution and measured stellar-minus-halo COM proxy connect
actual distances/LOS/proper motions, including shared MW nuisances and required
probability Jacobians. This is NOT a CF4-conditioned .1875 matter-field prior
or spatial map. Review335880 exposes separate-primary extrapolation despite
high proposal ESS; that alternative is NO-GO for scientific adoption, not
proof of physical impossibility. Gaussian leakage below the1000-DM particle
mass floor was fixed and existing draws restricted/re-normalized in335881,
preserving original outputs. Corrected satellite-conditional marks are usable
as a development input, with broad prior-dominated masses/environment, not a
calibrated LG reconstruction. Current products: `resolved_support_samples.h5`
and `physical_summary_v2.json` in `bundle_c_v1/lg_population_v1`.

Next substantive requirement is the SPATIAL profile/remainder-field model and
its joint connection to actual CF4/galaxy environment, not more mark-only
fits. Native membership/force resolution, missing mass/LMC/stellar-disk priors,
one-box covariance calibration and the32 environmental-feature support problem
are not solved by the marked Gaussian. The desired spatial LG-on/off map and
information gain remain undelivered. No job from this step remains active.

User authorized the next execution bundle, C-spatial, within the unfinished
master C. Scope/first calculation: BUNDLE_C_SPATIAL_DESIGN.md. Start with
native disjoint member profiles plus remaining total matter for16 training
and16 retained spatial cases; then define the continuous spatial distribution
and joint CF4/LG target. Do not reinterpret this as entry to D or completion
of the .1875 spatial deliverable. Source calibration alone does not close the
new execution bundle.

C-spatial first source calculation335916 completed4m08s:93 native member
profiles from16638729 massive rows,32 full member/remainder decompositions
at0.1875 with1.5 restriction and native mass/COM checks passing. These are
spatial CALIBRATION INPUTS, not a fitted conditional field or bundle closure.
Current record BUNDLE_C_SPATIAL_RUN.md. Whole source patches can overlap even
when halo IDs are disjoint; spatial fitting must prevent heldout-voxel leakage.
Next remains the continuous spatial distribution and actual observation-space
connection, not IC generation or another mark-only fit. No current job.

C-spatial conditional remainder candidate335968 completed2m46s, two focused
tests and all12 realizable/conservative draws passed numerical checks. A
geometry-only split provided13 training cubes and three disjoint retained
cubes without training/test voxel leakage. However all12 draws fail the
predeclared small-scale power criterion, both with and without the fixed
native member halos. Status NO_GO_CONDITIONAL_REMAINDER_MORPHOLOGY. This
stationary five-channel Gaussian-copula candidate is CLOSED, not adopted
or patched by amplitude/seed tuning. Native coarse7 moments and halos were
oracle conditions; no actual CF4/LG spatial inference was run. The fitted
profile regression is separate and not a joint halo/field law. Conserving
mass/momentum/second moments does not certify cosmological spatial structure.
See BUNDLE_C_SPATIAL_RUN.md for quantitative results. No active or downstream
job remains. C-spatial is not complete: the next design must supply missing
nonlinear environment/halo–matter spatial dependence before observation
conditioning. Do not close C or enter D on these diagnostic generated fields.

Current execution record: [BUNDLE_C_RUN.md](BUNDLE_C_RUN.md).

User-approved cause separation336263 completed42s; no new fit/draws. Native
density encode/decode is exact to2.6e-16 L1, and quantile+conservation roundtrip
power changes<=0.0944%, unlike the large stochastic generation failure. This
localizes the problem to the generative path/conditioning interaction without
uniquely identifying random phases as cause. Separate confirmed defect: scalar
fine sigma loses directional variance, changing mean velocity by3.4–16.8 km/s
per-axis RMS even for native input. Retain three fine variances in the next
representation. Details BUNDLE_C_SPATIAL_DIAGNOSIS.md and its comparison PNG.
Proposed next implementation BUNDLE_C_CONDITIONAL_FLOW_DESIGN.md: one spatially
conditioned multiscale flow with lossless conservative moment coordinates,
explicitly identifying the missing1.5 environment prior/inference and joint
member-field readout. This is DESIGN, not validated ML or an actual map. Current
approval ends at diagnosis/design; replacement pilot awaits approval. No job.
Subsequent user approval authorizes the implementation/one-GPU4h pilot.
Implementation/source1dc835b submitted as336268: shared multiscale spatial
conditional flow, explicit atom/continuous branches and conservative binary
moment coordinates retaining all directional variances. Frozen configuration,
within-fit spatial split/history limits and gates: BUNDLE_C_FLOW_PILOT_RUN.md.
One job includes native roundtrip, fixed learning and environment/fine mock
generation/evaluation. q_S, actual1.5/CF4/LG posterior and global384 inference
remain outside this pilot; no automatic follow-up science job.
336268 has now completed6000 steps in4m23s: native seven-moment/directional-
variance roundtrip passes, all16 generated cases fail development gates. No
actual-data adoption. User requested continued work2026-09-09. One identified
implementation mismatch is cropped24-parent training versus full64-parent
generation context. BUNDLE_C_FLOW_PILOT_RUN.md now freezes checkpoint-based
paired diagnosis and, ONLY if confirmed, one full-context6000-step correction
with unchanged model/seeds/gates. This is not an automatic longer-training
or prior-family series. All v1 products remain preserved. No actual CF4/LG or
global384 posterior launch; source/representation success is not C completion.
The gated repair337195 stopped after39s before training: crop-context mismatch
is confirmed, but trained-flow inverse/logdet errors exceed numerical limits.
User now authorizes cause separation and necessary correction/reverification:
one frozen same-input default-FP32/strict-FP32/FP64 comparison, not additional
training or relaxed morphology criteria. See BUNDLE_C_FLOW_PILOT_RUN.md.
337268 completed41s: all strict-FP32/FP64 cases pass, all default-FP32 cases
fail. TF32 convolution is supported as the inversion error source, not the
cause of all morphology failures. User now approves strict-FP32 paired gate
and, only on pass, the previously blocked single full-context6000-step fit.
Keep existing numerical/science thresholds and preserve all earlier outputs.
The current repair changes context and precision; no actual CF4/LG inference.
337279 completed9m30s,6000 steps, all numerical gates pass but all16 generated
fields still fail development morphology gates (fine high-band ratio0.235–0.593).
User approved ONE frozen analysis covering before/after quality, train/heldout
fixed likelihood, teacher-parent versus rollout losses, and likelihood/quality
alignment; driver then judges model viability AND actual CF4/LG connection.
Scope BUNDLE_C_FLOW_DECISION.md. No new training or replacement model authorized.
337496 analysis completed2m43s: v2 heldout fine high-band mean0.650 with true
parent versus0.390 in rollout; training0.676 versus0.297. Both one-step and
accumulated deficiencies remain. User accepted withholding current-model
adoption and authorized continued redesign. Candidate proposal:
BUNDLE_C_STRUCTURE_LG_REDESIGN.md, matched-update native-NLL control versus
structural-score repair plus a separately labelled LG member-state readout.
This is DESIGN, not a new fit authorization or a completed q_S/global law.
No actual LG posterior, direct-CF4 IC restart or amplitude repair is allowed.
Subsequent user approval authorizes entry to this next bundle, with a renewed
plan audit FIRST: Fable5 primary, Astra backup only if Fable audit cannot be
completed. Ask Q-GOAL (final-goal contribution) and Q-LEAN (excessive versus
necessary instrumentation/gates), feasibility, and essential/deferred scope.
An adverse scientific verdict is evidence to address, not an invocation failure
to bypass by seeking approval elsewhere. Preserve existing bundle boundaries.
Fable5 plan audit completed normally with CONDITIONAL GO (Q-GOAL conditional,
Q-LEAN broadly proportionate). Driver review found the suggested Gaussian
proxy circular unless its g(F) can infer members from total field alone; native
component inputs are not that operator. See BUNDLE_C_STRUCTURE_LG_PLAN_AUDIT.md.
No Astra fallback/no new fit. Ask before prioritizing the field-only LG proxy
connection ahead of the approved paired learning experiment; do not silently
claim that a Gaussian covariance supplies the missing joint physical q_S.
User now approves prioritizing field-only LG identification/readout before
paired learning. Every future bundle plan AND audit request must explicitly
cover MW/M31/M33 identification, ambiguous/unresolved cases and field-observation
connection. Current bounded implementation plan: BUNDLE_C_LG_IDENTIFICATION.md.
Native labels are for evaluation/calibration after blind candidates are frozen,
not inference inputs. Fable5 plan audit first, Astra fallback on invocation failure.
Fable5 completed the identification plan audit normally: CONDITIONAL GO, with
Q-GOAL direct/Q-LEAN proportionate. All four disclosure conditions (support
source, boundary truth counts, velocity/residual convention, role completeness/
shared peaks/aperture overlap) are implemented in the single CPU job. No extra
auditor/gate, new training or proxy likelihood is authorized by that result.
Identification job337986 completed1m13s, tests2/2, all32 cached native fixtures.
All32 show a shared M31/M33 nearest peak; distinct three-way matching0/32.
Two heldout M33 matches replace M31 on the shared peak, not a third detection.
No boundary truth loss; M33 training calibration unavailable. This closes the
bounded diagnostic, NOT q_S or an observed LG posterior. See the identification
report for counts and limitations. Recommend an explicit unresolved-member/
assignment observation-link design (or justified finer LOCAL information),
with Fable5 plan audit and user bundle approval before implementation. Do not
resume paired learning or new simulations automatically.
User subsequently approved the unresolved-member/observation-link DESIGN bundle.
Current proposal: BUNDLE_C_UNRESOLVED_MEMBER_DESIGN.md, with Fable5 plan audit
requested before any numerical implementation. It must address MW/M31 role
ambiguity as well as M33, normalized same-field component budgets and actual
field information, not repeat native-known-member transport or mark-only fits.
No new training, simulation, actual LG weighting or q_F promotion is implied.
Fable5 returned DESIGN CONDITIONAL GO (normal completion105402ms). Driver
adopts its Q-LEAN recommendation: a14-parameter composite-aperture kinematic
proxy first, not a high-dimensional cell-member learner. Exact finite pair
mixture, joint MW/M31/M33 uncertainty and fixed heldout field cross-scoring are
specified in BUNDLE_C_COMPOSITE_PROXY_PLAN.md. This is the NEXT proposed CPU
implementation, awaiting approval; no job submitted. Cell-member anchor measure
and p(E|F,O)/E-conditioned field prior remain unresolved and explicitly deferred,
not declared solved by conditional covariance or deterministic anchor encoding.
Fable5 follow-up on the concrete14-coefficient composite design returned GO.
Three required disclosures/tests (K spectrum/condition and sample counts,
consistent physical scale determinants, explicit unsupported-context labels)
are incorporated. Design bundle COMPLETE; proposed single2-CPU/1200MiB/10min
implementation now awaits user approval. No active job or new fit from this
design turn. No additional audit micro-stage for these accepted report items.
User now approved the concrete composite-proxy implementation and single Slurm
CPU diagnostic. Sources: src/cf4_lg_composite_proxy.py and
scripts/cf4_bundle_c_composite_proxy.py. Three focused tests run in the same
job before the one14-coefficient fit and frozen13/3 native evaluation. No new
neural training, simulation, actual-data weighting or extra audit is requested.
Composite diagnostic337991 now COMPLETED11s, tests3/3,13 training pairs and all
9 cross-field scores valid. Full/host-only scores prefer the source field in
all3 retained cases; incremental conditional-M33 contrasts versus other-field
means are[+.479,-1.358,-.023] nats and never rank the original field first.
Thus host field dependence is demonstrated diagnostically, but consistent
incremental M33 information is NOT established. No observed posterior/physical
member partition is promoted. See BUNDLE_C_COMPOSITE_PROXY_PLAN.md for full
results/limitations. Close this14-coefficient trial; no automatic refit or
neural model. Next design/approval must address the missing M33 mass/subcell
or finer-local-field connection and retain unresolved q_F/selection caveats.
User approved the next M33 mass/subcell DESIGN. BUNDLE_C_MEMBER_BUDGET_PLAN.md
now specifies one no-fit necessary mass/momentum/diagonal-second-moment budget
test over seven support unions, preventing host/satellite double counting.
Fable5 DESIGN CONDITIONAL GO received normally; four essential conditions are
incorporated (identical cell masks, congruent PSD scaling, unavailable contexts,
common BOX velocity convention). Native supplied mock positions/COMs/masses
are explicit stronger conditioning, not recovered identities or actual masses.
If hosts exclude every alternative, report inconclusive M33 conditional power,
not no M33 information. No posterior/support likelihood or sufficient physical
decomposition is claimed. Design complete; single2-CPU/1200MiB/10min diagnostic
implementation awaits approval. No new numerical job/ML/simulation launched.
User now approves that exact member-budget implementation/one CPU execution.
Sources src/cf4_member_budget.py and scripts/cf4_bundle_c_member_budget.py:
two focused tests then cached fixed16/17/20 mocks and both radii in one job.
No new plan audit, fit, actual-data weighting or automatic downstream task.
Member-budget338040 completed11s, tests2/2,18 comparisons,783 cached cell reads.
Both fixed radii and mass/moment modes: own-field controls3/3 pass, all6
alternatives already fail host-only mass requirements, unavailable0. Decision
INCONCLUSIVE_NO_HOST_COMPATIBLE_ALTERNATIVES, NOT no M33 information. The
necessary union-budget operator is implemented but not a posterior or physical
allocation. Close this one-shot test without further contexts/radii/refits.
Recommend next planning return to the missing present-field prior/inference
with MW/M31 conditions and explicit unresolved M33, rather than more standalone
M33 diagnostic variants. No next bundle is authorized by this result; no job
remains active. See BUNDLE_C_MEMBER_BUDGET_PLAN.md for results and limitations.
User approved the next central-work DESIGN on2026-09-10. Proposed plan:
BUNDLE_C_FIELD_RECOVERY_PLAN.md. It closes standalone member diagnostics and
asks Fable5 whether ONE bounded matched generative-objective repair is the
right immediate bottleneck, explicitly retaining the missing member/selection
and global environment laws. No actual-data fine inference or numerical run
is authorized by this planning approval. Fable5 review completed normally in
137035ms: CONDITIONAL GO, Q-GOAL/Q-LEAN accept one terminal matched experiment.
Three essential corrections are incorporated: training-gradient-based frozen
weight, worst-case full-rollout gradient/memory screen before training, and
one-step/rollout reporting even on failure. Driver does not adopt the audit's
overstrong causal classification or its coarse-posterior/prior-fine-IC fallback
as an automatic route change. See BUNDLE_C_FIELD_RECOVERY_AUDIT.md. DESIGN
complete; one1000-update-per-branch/4h Slurm implementation awaits approval.
No numerical job submitted, no high-resolution observed LG posterior exists.
User now approves the exact field-recovery implementation and one Slurm GPU
experiment. BUNDLE_C_FIELD_RECOVERY_RUN.md records source/configuration:
all-trace energy-score gradient, bounded8-pair screen, equal1000-step branches,
unchanged morphology gates and terminal comparison. No new Fable request or
standalone M33 diagnostic; no actual-data fine posterior or automatic follow-up.
Implementation source43692a1 pushed and submitted as Slurm338194. Initial
state PENDING(Resources), no node/tests/training yet. One job runs the7 focused
regressions, fixed feasibility segment, matched learning and final evaluation.
Execution status belongs in BUNDLE_C_FIELD_RECOVERY_RUN.md, not inferred from
submission success. No new scientific output or next-bundle authorization.
338194 subsequently finished both1000-step branches and saved checkpoints,
but failed at evaluation entry on2026-09-10 02:04:45 KST: FP64 native split
coordinates were passed to FP32 convolution. Seven tests and the operational
screen passed; morphology remains unevaluated. User requests this code fix.
The correction adds the missing network dtype conversion and one regression,
plus evaluation-only resumption using unchanged checkpoints/criteria into a
new output directory. No retraining, new model trial or additional plan audit.
Details BUNDLE_C_FIELD_RECOVERY_RUN.md; evaluation-only execution via Slurm.
Evaluation-only338389 now COMPLETED4m54s,8 tests pass, both saved-model inverse
checks pass, additional training updates0. Intermediate338388 test cleanup
defect is corrected/preserved in the run record. Fixed evaluation source18b84f8.
Both branches pass0/16 original morphology cases. Retained .1875 true-parent
P/native control/repair0.654972/0.664406; rollout0.436332/0.462160. ES modestly
improves both summaries but remains below acceptance: CLOSE_THIS_REPAIR_LINE_
BOTH_FAIL. Close this objective/current-architecture repair without more
steps/seeds/weights. No actual high-resolution LG posterior, no next-bundle
launch or automatic prior-fine-IC fallback. Detailed results and preserved
checkpoints: BUNDLE_C_FIELD_RECOVERY_RUN.md. Evaluation code error is resolved.
User now approves designing the recommended conditional3D U-Net diffusion
replacement, with augmentation and an explicit same-field LG connection plan.
BUNDLE_C_DIFFUSION_PLAN.md is the next proposed implementation, not a launch.
Fable5 plan audit completed normally121178ms: CONDITIONAL GO, Q-GOAL/Q-LEAN
accept ONE bounded field-prior attempt. Five conditions are incorporated:
pre-submission sizing, legal-branch identity before optimization, invalid draws
count as failures, budget-short runs are inconclusive, and no further field
training after failure/inconclusive until native-data member identifiability/
learnability is addressed. Disposition BUNDLE_C_DIFFUSION_AUDIT.md. The member
q_S, selection, global environment and efficient actual-data inference remain
unimplemented; neither morphology nor a normalized diffusion sampler solves
them. No repeated closed member diagnostics or direct-IC fallback. DESIGN
complete; proposed1-GPU/4-CPU/48GiB/24h,30,000-update implementation awaits
user approval at the next bundle boundary. No numerical job submitted.
User now approves this implementation and single Slurm diffusion experiment.
Execution record BUNDLE_C_DIFFUSION_RUN.md; source/model config and static
sizing are implemented.34,095,557 parameters; concrete bounded-cache sizing
reduces host request from provisional48GiB to15GiB (12GiB estimated peak+20%,
rounded), within the approved envelope. One allocation runs8 focused tests,
native identity,32 real largest-scale operational updates, then the same fit
and fixed evaluation. Only the8 original fine draws are comparable here; the
old8 environment draws are explicitly out of scope, not declared passing.
No new audit, q_S learner, observed posterior or automatic follow-up launched.
Implementation341b419 pushed; Slurm338402 submitted2026-09-10 09:03:02 KST.
Initial state PENDING(Priority), no allocation/tests/training yet. One job
will execute the approved test/fit/evaluation sequence when resources arrive.
Actual run state/results belong in BUNDLE_C_DIFFUSION_RUN.md; submission is
not a numerical pass or scientific result. No separate monitoring daemon.
Previous B delivery: [BUNDLE_B_RUN.md](BUNDLE_B_RUN.md).
Update2026-09-10 18:26 KST:338402 COMPLETED/exit0 after9h02m57s;
8 tests passed and30k updates finished, but all8 retained draws are
GENERATION/SUPPORT FAILURES, not valid fields with measured bad morphology.
All12 rollout cases fail first refinement; all native-parent scales fail too.
Continuous loss1.000238 is consistent with a zero predictor, not proof of
its cause. No fine-field or observed-LG promotion. User 'next proceed'
authorizes ONE20min Slurm frozen-checkpoint diagnosis, zero optimizer steps.
Fable5 CONDITIONAL GO; plan and incorporated conditions:
`BUNDLE_C_DIFFUSION_DIAGNOSIS{,_AUDIT}.md`. No verifiable original step0
exists: do not trust seed reconstruction as proof of weight updates.
Check actual normalization, simple denoising baselines, raw/EMA gradients
and unchanged failed reverse chain. No further field training before
member learnability is addressed and a new concrete plan is approved.
Frozen diagnostic implementation3240750 pushed; Slurm338746 submitted
2026-09-10 18:41:37 KST, initial PENDING(Priority), no numerical result yet.
1 GPU/2 CPUs/6GiB host/20min; no automatic additional training or bundle.

338746 completed2026-09-10 18:42:44 KST in36s, zero optimizer steps. On both
native tested scales raw/EMA epsilon RMS~.005, MSE~1 and near-zero correlation;
at t100 the algebraic reference MSE~2.4e-7. Reverse latent RMS rises1->2032 and
fails fraction support. Normalization agrees; continuous gradients are nonzero
and checkpoint/plain backward agrees. Failed denoising is established, but its
unique architectural/optimization cause is not. No new morphology or actual LG
product. Frozen diagnosis CLOSED, not another monitoring/diagnostic series.
User requested next repair PROPOSAL2026-09-11; `BUNDLE_C_REPAIR_PROPOSAL.md`
is design only, with Fable5 plan review. No new implementation/learning job
authorized or submitted. Next bundle requires user approval.
Fable5 repair-plan review returned CONDITIONAL GO111090ms. Current proposed
implementation specification is `BUNDLE_C_REPAIR_DISPOSITION.md`; submitted
plan preserved separately. Next proposed bundle is ONE native field-only
MW/M31/M33/remainder mass-allocation learner,13/3 development fixtures,
<=2000 updates/90min Slurm. No further total-field-prior learning in that job.
Driver corrects the proposed class-balanced CE to per-field/per-role normalized
map-L1 to avoid biasing small-member mass fractions. Pilot is not a calibrated
member posterior or a member-velocity model. Stable v-prediction and separated
category gradients remain a deferred repair hypothesis, not code or a job.
Await user approval; do not revive closed peak/proxy/budget tests or launch a
long density fit automatically. A pass leads to joint-member/denoiser design.
User approved the single native member-mass implementation/pilot. Source and
fixed choices are recorded in `BUNDLE_C_MEMBER_MASS_RUN.md`. One530804-parameter
U-Net,13/3 fixed fixtures, equal-role normalized map-L1, <=2000 updates/70min
within one90min Slurm job,1GPU/2CPU/6GiB. Three focused/reused tests run in the
same allocation; no new audit ladder. Mass-readout only, no member velocities,
observed fine field, diffusion repair/training or automatic next bundle.
Implementationee1c8e7 pushed; Slurm341713 submitted2026-09-11 14:57:49 KST,
initial PENDING(Priority), no allocation or numerical test/fit result yet.
Execution record `BUNDLE_C_MEMBER_MASS_RUN.md`; no separate polling daemon.
341713 completed2026-09-11 15:15:31 KST,17m34s,3 tests passed,2000 updates.
NO_GO_MEMBER_MASS_READOUT: training and all development criteria fail, with
large member mass excess and near-zero native overlap. No accepted mass
readout, member velocities or observed LG; no automatic follow-up. User now
requests independent driver/Fable plans and comparison, DESIGN ONLY. Driver
draft frozen before fresh Fable call in `MEMBER_REPAIR_DRIVER_INDEPENDENT.md`;
Fable receives current failed source/results, not the driver's new proposal.
Independent Fable proposal completed normally123782ms. Original driver/Fable
drafts preserved in `MEMBER_REPAIR_{DRIVER,FABLE}_INDEPENDENT.md`; comparison
and driver recommendation `MEMBER_REPAIR_COMPARISON.md`. Both favor correcting
training conditioning. Driver rejects Fable's unsupported permanent M33 waiver,
exact-baseline-with-epsilon claim and universal learnability failure inference;
also defers the driver's own unnecessary hierarchical architecture change.
Proposed smaller combination: unchanged backbone/four-way output, TRAINING
mass-fraction initialization, separated mass/shape objective, one-field learning
segment then conditional continuation. This is a comparison recommendation,
not Fable approval of a combined execution plan or user launch authorization.
User now approves that final smaller combination. Frozen concrete plan:
`BUNDLE_C_MEMBER_REPAIR_PLAN.md`; implementation/run disposition:
`BUNDLE_C_MEMBER_REPAIR_RUN.md`. Fable5 returned CONDITIONAL GO normally126851ms
(Q-GOAL/Q-LEAN pass), requiring all16 fixtures' four integrated target masses
positive before optimization; incorporated using existing target validation.
Keep flat head/backbone; TRAINING integrated-fraction initialization plus
log-mass-squared/spatial-KL objective. One300-step single-field screen then,
ONLY on pass, the SAME optimizer continues1700 steps on13 fields. Failed
screen is short-budget inconclusive, not fundamental M33 unidentifiability.
1GPU/2CPU/6GiB/90min Slurm, new member_mass_repair_v2 output, preserve v1.
No automatic density fit/IC/next bundle; actual-data member-to-field link
and joint uncertainty remain unimplemented. Launch record belongs in run file.
Implementation8dbaeae pushed; Slurm342013 started2026-09-11 18:26:28 KST on
syn05,1GPU/2CPU/6GiB/90min, initial RUNNING. Numerical/scientific outcomes
not established at submission; single-field continuation is gated inside
that same allocation. No automatic downstream science job.
342013 now closed: COMPLETED3m09s/exit0,5 tests pass,300 single-field updates;
screen failed and no multi-field learning/evaluation ran. Final mass ratios
MW/M31/M33=2.064/1.146/.957; overlap=.915/.838/.789; L1=1.235/.469/.379.
INCONCLUSIVE_SINGLE_FIELD_LEARNING, not fundamental impossibility. User now
explicitly requests5000 cumulative updates on the single field. Resume300
model AND Adam state, add4700, unchanged computation/criteria, fixed endpoint;
no automatic13-field learning afterward. Frozen plan/run:
`BUNDLE_C_MEMBER_5000_{PLAN,RUN}.md`. Fable5 CONDITIONAL GO58623ms, Q-GOAL/
Q-LEAN accepted; hard-abort on incomplete/mismatched resume state incorporated.
One1GPU/2CPU/6GiB/90min Slurm job; source outputs preserved, new directory
member_mass_single5000_v3. Launch/results recorded in its run file.
Implementation787497f pushed; single5000 continuation Slurm342086 started
2026-09-11 23:09:47 KST on syn05 via Slurm, initial RUNNING with no numerical
pass yet. One allocation runs six tests, validates/restores300 then trains
4700 additional updates; report endpoint and wait before another bundle.
342086 completed38m25s/exit0,6 tests pass,5000 cumulative updates. All single-
field criteria pass: MW/M31/M33 L1 .03147/.01031/.01602, mass errors<1.7%,
overlap>=.9886. PASS_SINGLE_FIELD_LEARNING_ONLY, not generalization. User
approves next13-field learning/3-development evaluation. Frozen plan/run:
`BUNDLE_C_MEMBER_MULTI_{PLAN,RUN}.md`. Fable5 CONDITIONAL GO98118ms; primary
training gate excludes pretrained fixture0, existing13-field memory accounting
and48-symmetry tests resolve remaining conditions. One13000-new-update fit
from full5000 model/Adam state,1000 new exposures per training field, unchanged
model/loss/LR, signed augmentations. Same retained development criteria, no
best-epoch choice or actual CF4/LG posterior claim.1GPU/2CPU/6GiB/3h Slurm;
new member_mass_multifield_v4, no automatic downstream science bundle.
Implementation20253e1 pushed; Slurm342129 submitted2026-09-12 00:30:11 KST,
initial PENDING(Resources), no numerical tests/learning yet. Allocation will
execute the full one-fit/evaluation sequence; run record contains actual state.

342129 COMPLETED1h44m53s/exit0,7 tests pass,13000 new updates (1000/field).
NO_GO_MEMBER_MASS_READOUT: training/development criteria fail; formerly fitted
fixture0 also degrades. Same field/symmetry log comparisons nevertheless show
learning improvement, not total learning failure. Source random companion
selection is not encoded in field-only inputs, but neither unique cause nor
duplicate-input contradictions are established. User approves ONE frozen
evaluation-only diagnosis and observation-aware redesign, no retraining:
`BUNDLE_C_MEMBER_DIAGNOSIS_PLAN.md`. Fable5 plan audit first; two saved models,
six fixed fields/all48 orientations and existing small selection tables in
one15min Slurm GPU allocation. Do not revive closed prior/peak/proxy repairs
or claim a high-resolution observed field; next training awaits approval.
Fable5 normal78825ms CONDITIONAL GO; incorporated tolerance/material-spread
criteria and positive finite mass-denominator guard plus absolute errors.
Frozen diagnostic sourcedd006b5 committed/pushed, Slurm343469 submitted,
initial PENDING(Priority). Runtime results belong in BUNDLE_C_MEMBER_DIAGNOSIS_RUN.md.

343469 completed3m24s on syn07/A40,8 tests pass,576 frozen forwards and no
saved-model updates. Both identity results reproduce. Orientation sensitivity
pre-exists the multi-field fit; final multi model remains sensitive in6/6
tested fields.12/16 source cases have companion alternatives, but unique-choice
development16 also fails; no identical-origin/different-label pair found.
Diagnosis closed, no unique failure-cause or accepted member/field claim.
User requests next correction. Proposed `BUNDLE_C_ROLE_LOCATION_PLAN.md`
replaces deterministic mass allocation with an autoregressive distribution
over MW/M31/M33 center cells and structurally cubic-equivariant scalar kernels.
It is LOCATION ONLY, not q_S masses/COM or an observed .1875 field posterior.
Fable returned a read-intention preamble without a verdict; not an audit pass.
Driver backup review `BUNDLE_C_ROLE_LOCATION_REVIEW.md` is conditional feasibility
GO, with explicit probability/selection/teacher-forcing/center-label caveats.
User clarifies audits are advisory: driver verifies and decides with reasons;
independent evidence and separate user bundle approval remain required.
User now explicitly approves implementation and ONE1GPU/2CPU/6GiB/4h pilot,
6500 joint updates (500 per13 fields). Code implements the fresh72,417-parameter
location law, sequential factor gradients, native center-cell references,
nonoracle autoregressive draws and before/after48-view probability tests.
Static syntax checks pass; numerical tests and fit run only in the SAME Slurm
allocation. Job347007 submitted from pushed source50841a2 and RUNNING on
syn05 from2026-09-12 15:52:02 KST. See `BUNDLE_C_ROLE_LOCATION_RUN.md`.
347007 has now COMPLETED2h36m33s/exit0 at18:28:35 KST,6500 updates and final
evaluation complete. Tests4/4 and symmetry before/after pass, but
NO_GO_ROLE_LOCATION_AT_FIXED_BUDGET. Mean train joint NLL .0000387 versus
development63.0029 (density reference10.5120). Autonomous positions fail too.
Close the13-field U-Net pilot; no longer-training continuation or promotion.

User approved the recommended next bundle: population-weighted multiple
native center labels using existing total field/catalogue, ONE small density-
anchored location model and spatially separated comparison. Concrete plan:
`BUNDLE_C_POPULATION_LOCATION_PLAN.md`. Fable5 returned conditional approval
normally138976ms; driver disposition in the corresponding REVIEW document.
The adviser incorrectly added nested time caps; driver rejects that arithmetic,
keeps110min TOTAL including70min-capped learning inside120min Slurm. Empty-
cell/periodic-overlap/early-count conditions are incorporated without extra
audit stages. Model21 ridge-regularized linear coefficients plus one scalar
calibration weight,12 epochs with all alternatives and uniform-observer/
M31/M33 hierarchical weights. Fixed spatial slabs include feature halos;
old three development volumes excluded from new test. No new raw snapshots,
profile extraction, q_F learning, actual-data posterior or IC job. Execution
record: `BUNDLE_C_POPULATION_LOCATION_RUN.md`. Approval covers this single
comparison; next bundle still requires its concrete result and user direction.
Slurm347085 started19:08:47 KST2026-09-12 on syn05/A40, pushed sourceaa024b4,
1GPU/2CPU/10GiB/2h. Four numerical regressions pass. Fixed split counts are
1388 training /87 calibration /127 test observers, with5006/261/449 native
triples and no cross-split native-ID overlap. Population scope passes.
347085 now COMPLETED/exit0 in3m11s at19:11:58 KST;12 epochs/2088 updates,
all tests/symmetry checks and all five fixed location-feasibility criteria
pass. Calibration alpha=1. Mean test joint NLL9.9064 versus density11.2234;
M33 NLL3.4396 versus density3.8976/shared-cell3.7562. Training joint10.6850
versus density12.0591; no comparable extreme train/test score gap here.
Autonomous512 triples complete, with32 distinct draws per observer/law; broad
weighted-target distances remain (e.g. M331.214–3.603 cMpc/h over8 testcases).
.1875 cell size is NOT .1875 position accuracy. Driver accepts and CLOSES the
bounded position comparison, not a physical halo/mass/COM model or q_F.
These are correlated one-box samples, not independent universes or observed
LG fields. Next proposed outcome: connect LG position observations to a
SAME-field likelihood and the remaining field/mass/COM model, working toward
an LG-on/off field response, not another location-score repair loop. No new
bundle is launched; user direction is required. See the run record for full
proper scores, autoregressive caveats and preserved artifacts.

Autonomous follow-up now implements BUNDLE_C_POSITION_LINK_PLAN.md: frozen
location law -> joint LG distance/sky observation density on the SAME field,
with shared distance covariance, within-cell integration and no native
candidate parents supplied to inference. Fable5 PROCEED; driver disposition
uses exact exponential tilting/cell-face intervals instead of GH aliasing,
and finite differences at an interior conservative mixture to avoid cold/
empty boundary nondifferentiability. One GPU/2CPU/6GiB/30min, eight fixed native
test fields/all archived alternatives, actual-data interface scores and one
field-sensitivity projection. No training, field selection, density painting,
posterior or IC. Missing selection, offsets, physical field/mass/COM joint law
remain explicit. Execution/results: BUNDLE_C_POSITION_LINK_RUN.md. Autonomous
approval replaces another wait; closed model repairs remain closed.
Position-link347086 completed14s but an unweighted worst-case quadrature error
estimate was too loose. Correction01971b8 weights each rectangle error by the
same qA*qT used in the integral; old worst-case estimate remains visible,
threshold unchanged, numerical estimates distinguished from exact tail bounds.
347087 COMPLETED15s/exit0, all3 tests and integrals/conservative derivatives
pass. All208 mock +8 actual-data logL values unchanged EXACTLY. Eight mock
source fields rank first both in joint score and incremental M33 score;
development evidence only. CLOSE position diagnostics: usable observation-
likelihood component, NOT field selection, masses/COM or observed posterior.
Next autonomous design targets a physically conservative joint-field prior;
old field diffusion already used dense random native translations (unlike
13-field member fits), so more data alone is not a justified repair.

Autonomous next generator experiment: BUNDLE_C_STABLE_FIELD_PLAN.md, Fable5
GO with driver corrections in REVIEW. Fresh1,490,406-parameter full-resolution
v-parameterized joint moment model, independent continuous/category branches,
explicit observer conditioning and matched archive-E native fields. This
changes the unstable reverse reference and gradient paths, not just length.
80^3 generated buffered field /inner64 at.1875, native1.5 parent, three scales;
all physical seven moments retained. Source-backed v parameterization is a
plausible remedy, not a proven cosmological reconstruction. One24k-step fit,
1GPU/2CPU/6GiB/4h with tests and fixed evaluation in the same job. Sixteen
draws/eight parents plus frozen field-only LG role/observation machinery;
no true fine buffer or native role parents passed to generated-field readout.
Conditional-E/native-coarse development only; no observed posterior, IC or
automatic continuation of this fit after a miss. Run record:
BUNDLE_C_STABLE_FIELD_RUN.md.
Generator source0bfd832 pushed, Slurm347088 started2026-09-12 20:19:41 KST
on syn05/A40,1GPU/2CPU/6GiB/4h. All3 tests pass;64 normalization observers
and192 native roundtrips pass,1388 training observers retained. Training400/
24000 at application131s, early v-loss improves on its zero-v training
reference but NO validation/generated-field result yet. Host3.030GiB,
GPU reserved1.176GiB. Rough initial total runtime80–95min; hard Slurm limit
00:19:41 KST next day. The same allocation automatically proceeds through
checkpoint, heldout denoising,16 generated fields and role readout. Next
driver action: read fixed347088/final artifacts and judge, continuing under
autonomous authority without another routine approval request.

Previous diagnostic delivery: [BUNDLE_A_RUN.md](BUNDLE_A_RUN.md).

Completed A history, not instructions to rerun: the four-fit comparison
failed the frozen quantile-prior adoption rule. That repair series closed.
The original24-nuisance PM-calibrated model then supplied an actual-data
model-stress diagnostic, followed by the approved observation-model correction
and length check below. None scientifically promotes the baseline.

Actual-data preflight333862 passed; fit333872 and aggregation333990 completed.
The actual-data result is NO_GO_SAMPLER_NOT_VALIDATED (max Rhat1.325,
minimum bulk ESS11.1), not a usable posterior. Saved-chain diagnostic334240
completed without refitting. It confirms field/H0 coupling and identifies a
physical-Mpc versus Mpc/h magnitude convention mismatch in the imported
2M++ population/selection setup. See
[ACTUAL_PREVIEW_DIAGNOSIS.md](ACTUAL_PREVIEW_DIAGNOSIS.md).
The user subsequently approved the focused observation-model correction,
field-aware sampler adjustment and ONE corrected actual-data refit.
Active correction plan: `config/cf4_actual_data_corrected_v2.json`, documented
in [ACTUAL_CORRECTION_RUN.md](ACTUAL_CORRECTION_RUN.md). Preserve the old fit.
The disjoint sample is retained; calibrate its survival from a separate20%
parent mark sample, retain the old20% heldout, and use60% for field fitting.
This supersedes the earlier one-fit cap only for this approved refit, not for
further prior-family experiments. No Bundle B execution is active.

Corrected fit334345 completed in22m22s. Max Rhat1.034 passes, but H0 and
its field-dependent conditional mean have bulk ESS97.1/85.1 (<100); retain
NO_GO_SAMPLER_NOT_VALIDATED. The user now approved ONE sampling-length check:
`config/cf4_actual_data_longer_v3.json`, four fresh chains with2048 samples
each,512 warmup, unchanged model/data/gates. Reuse frozen actual inputs,
preserve old chains, and do not claim exact continuation or pool the runs.
No automatic further extension or Bundle B launch.

Final length-check job334358 completed2026-09-07 22:24:15 KST in36m33s;
aggregation334359 completed22:24:23. All predefined sampler gates pass:
max Rhat1.03183, min bulk ESS113.59, min tail ESS316.88, divergences0.
Heldout count/velocity moment-residual SDs .99628/1.11173 are diagnostic,
not scientific calibration. Product remains12 cMpc/h and contains no
resolved LG. Stop adding samples. The user accepted preparing the next
bundle: actual environment comparison, explicit LG observational likelihood
contract, and a bounded coarse z=0→IC→forward-z=0 mock bridge. No B jobs yet.

Z4–Z11 were N32, 12 cMpc/h development experiments, not actual-data maps.
Z8's amplitude interpretation was erroneous and has been withdrawn. Z10
fixed-field and Z11 prior-compatible controls recover the injected tracer
curvature; Z9 free-field PM cases do not. This supports a field/observation
model mismatch but neither isolates its sole cause nor proves a quantile
transform will fix it. Z11 completed all four fits; its results are at
`/gpfs/kjhan/CF4/z0_density/z11_prior_control_v1/comparison.json`.

The following is the historical Bundle A scope, now completed at diagnostic
level; it does not authorize another comparison or actual-data fit:

1. Finish ONE training-only non-Gaussian-prior comparison against the saved
   native-PM controls, with at most four new mock fits. Reuse existing
   simulations, likelihood, sampler and reports. Require spatial field
   recovery and held-out predictions, not just a nicer density histogram or
   recovered nuisance coefficient. Exact cases/resources/decision rule:
   `config/cf4_bundle_a_prior_to_data_v1.json`.
2. Driver judges the comparison. No automatic additional prior families,
   nuisance terms or new simulations if it fails. State the model limitation
   and decide its suitability for a preliminary observation-space diagnosis.
3. Within A, implement and run at most one actual-data preliminary four-chain
   fit after its input semantics and chosen model are recorded. Use actual
   CF4 velocity data and prepared 2M++ counts, preserving selection, errors,
   overlap treatment and heldout separation. No pseudo-truth or truth-based
   accuracy/coverage numbers for the real universe. If model suitability
   fails, the product is a model-stress diagnostic, not a validated posterior.
4. Deliver actual-data mean/sample density maps, mean velocity and posterior
   velocity uncertainty, predictive residuals and limitations. Distinguish
   uncertainty in the mean field from physical velocity dispersion/FoG;
   current population FoG parameters are not a resolved sigma_v(x) field.
   Close A with a concrete Bundle B design and request approval.

The real-data preview remains 12 cMpc/h and cannot certify the target LG
scale. Independent galaxy/selection mocks and physically relevant validation
are required before scientific promotion, but not an unbounded prerequisite
to viewing a clearly labelled diagnostic map. Reused development fields and
shared generator/inference code are never independent validation.

Update2026-09-12 21:57 KST:347088 completed24000 updates/final checkpoint,
then FAILED at evaluation entry (duplicate status keyword in driver code).
No generated quality verdict exists. Autonomous technical recovery fixes the
assignment and adds evaluation-only final-EMA loading with identical saved
normalization/config/seeds, zero optimizer steps, original outputs preserved.
New stable_field_eval_v2, one GPU/2CPU/6GiB/30min Slurm allocation runs four
tests and the original fixed endpoint. No new model/advisory audit required
for this same-bundle technical correction. See BUNDLE_C_STABLE_FIELD_RUN.md.

Update2026-09-13:347108 COMPLETED2m23s, tests4/4, final EMA evaluation with
zero optimizer steps. All16 draws valid/conservative, terminal latent RMS
.926–1.077, but individual morphology0/16 and two-draw summary means2/8 pass.
Boundary-gradient ratio fails16/16; power5/16; connected fraction2/16.
NO_GO_E_CONDITIONAL_FIELD_GENERATOR: stable generation is progress, not
scientific adoption. User approves saved-field boundary/metric diagnosis;
no retraining, new samples or threshold change. BUNDLE_C_BOUNDARY_DIAGNOSIS.md.

Boundary diagnosis347159 completed8s with two controls and all eight stored
first draws; original24 axis-ratios reproduce. Raw squared-gradient boundary
energy is dominated by10 edges (median85–87%); native ratios themselves vary
widely. Log-density still shows modest grid-phase roughness: median boundary/
internal .994 native versus1.141 generated. Mixed metric sensitivity and
generated hierarchy artifact; not a pure code bug or significance claim.
Close this diagnosis, retain original NO-GO/power failures, no blind longer
training or current-model actual-LG promotion. BUNDLE_C_BOUNDARY_DIAGNOSIS.md
records evidence and next model-design target; no active follow-up job.

Latest audit policy2026-09-13: external review only for an important discovery,
goal revision or large calculation. Driver handles all remaining evaluations
and routine plans. This supersedes historical every-bundle external reviews;
retain scientific goal/lean reasoning without another gate framework.

User reconfirms continuous autonomous work without routine approval2026-09-13.
Next bounded correction-localization uses saved rollout restrictions at three
scales and sixteen frozen-model native-parent one-step draws; no fit, new
full rollout, threshold change or actual posterior. It separates inherited
parent error from one-step field/velocity-variance partition error before
selecting one correction. BUNDLE_C_SCALE_LINK.md, driver review only,
1GPU/2CPU/4GiB/10min Slurm. No further standalone role-identification checks.

347160 now COMPLETED57s on2026-09-13, tests4/4 and sixteen native-parent
draws/physical variance identities pass. Both one-step error and propagation:
at.1875 bulk RMS ratios teacher1.185/rollout1.567; sigma .949/.799. Already
at.75 bulk budget fraction .235 versus native.094. Grid roughness persists
with true parents. Close diagnosis. User requests correction; driver implements
ONE matched6000-update-per-arm comparison from final EMA, original continuous
model versus fine-lattice residual path + parent-only mass/variance-weighted
v loss. Freeze category in BOTH arms; same noise/data/optimizer/normalization,
unchanged physical decoder/sampler/criteria and generated-field LG readout.
BUNDLE_C_FIELD_LINK_REPAIR.md;1GPU/2CPU/6GiB/2h, routine driver review,
no new posterior or automatic repair series. Do not claim correction efficacy
before final physical fields are compared.

## Working rules

User2026-09-13 preapproves continuous multiple in-goal bundles without routine
approval stops. Next BUNDLE_C_FIRST_SCALE_EXPERT.md: one first-scale-only
6000-update expert, original finer networks frozen, existing sixteen-draw
physical/LG readout comparison in one1GPU/2CPU/6GiB/30min job. Scale competition
is a hypothesis, not diagnosed fact. No output amplitude adjustment, no
actual posterior promotion or blind subsequent specialist/epoch sweep.

347264 completed42m56s, both6000-update arms and7 tests complete. Individual
quality0/16 versus1/16; ensemble0/8 both. Physical partition/grid improvement
small, no accepted field prior or observed posterior. Close this extension.
User approves next bundle: BUNDLE_C_SPLIT_ATTRIBUTION.md, one saved-field
first-split decoder intervention (native/generated codes versus values),
2CPU/3GiB/5min Slurm, no training or new stochastic draws. Oracle hybrids
are diagnostics only, not inference inputs or proof of unique neural cause.
Next model choice requires this concrete evidence, not more blind steps.
352595 now COMPLETED26s:64 rows/32 roundtrips pass, zero learning/draws.
All first-split legal category arrays match native EXACTLY in both arms.
Swapping categories does nothing; native continuous values recover native
physics. Error resides in continuous output values at this split; no claim
of unique training cause or exclusion of sampling-category history/finer
scale effects. Close diagnostic; no category-only repair justified. Next
design targets continuous physical-budget distribution, not amplitude repair.

- Prefer substantive scientific outputs over more generic validation code.
  Reuse tests; add only checks necessary for the changed computation.
- A clean sampler is necessary, not scientific success. Separate posterior
  mean smoothing, individual-draw structure, uncertainty and phase recovery.
- Syntax is a Slurm login server. Numerical jobs use Slurm; no manual syn101
  or login-node calculations. Current GPU partitions: a100_pcie,a40,a100,h100,h200;
  exclude syn06 for these fits. Request estimated peak memory plus ~20%.
- GPFS is ordinary shared storage. Read/write scoped project artifacts;
  do not implement storage/inode/renameat2 probes or process-scan monitoring.
- Use fixed job IDs and final artifacts for bounded checks; no pgrep loops.
- The driver plans, implements, runs, evaluates and commits. External review
  is reserved for important discoveries, goal revisions or large calculations;
  routine work is driver-reviewed. When required: Fable5 primary, Astra backup
  if the invocation fails or produces no usable audit. Reviews answer
  Q-GOAL and Q-LEAN plus feasibility
  and essential/deferred scope. Do not turn these two questions into a new
  gate framework or per-step audit series. Prior Astra closure-audit waiver
  remains unless changed by the user; this instruction concerns plan audits.
  Explicitly audit the MW/M31/M33 identification plan in every bundle; require
  no oracle component access on new fields and honest unresolved-M33 handling.
- Commit/push coherent changes. Preserve unrelated user work and all failed
  scientific results. Keep run summaries current; do not confuse submission,
  sampler pass, scientific acceptance and final-goal completion.
- At every bundle boundary report the result, goal contribution, unresolved
  risk and next deliverable. When approval is needed say **승인해주세요**.
