# Evidence for end-goal redesign — 2026-09-13

This is OurUniv / CF4 in /home/kjhan/BACKUP/CF4, branch agent/freeze-zoom-pipeline.
DESIGN ONLY. Do not edit code, submit jobs, run shell diagnostics, inspect
credentials, or change the goal. Give substantive advice, not a read-intention
preamble. The driver supplies this packet to Fable5 for separate consultations.

User destination: ACTUAL CF4 + galaxy observations + explicit LG observations
-> present-day density/velocity posterior -> compatible LCDM ICs -> evolution
-> phase-consistent LG zooms for studying MW/M31/M33 formation. LG map cell
<=0.3 cMpc/h; surroundings1–2 sufficient; unconstrained fine modes remain prior
content. No unique historical fine-phase claim. Virgo, Coma, Local/Bootes Void
environment retained within verified data/box support. User strongly objects
to quietly returning to historical direct CF4->IC or spending months perfecting
ML/verification instead of making observed z=0 maps. A mathematical use of IC
latent variables as a dynamical prior must be explicit, not hidden relabelling.

Assets/results, independently read from project records:
1. Actual CF4+2M++ corrected N32/L384cMpc/h posterior at12cMpc/h: four chains,
   Rhat1.03183,minbulkESS113.59,tail316.88,zero divergences. Sampler passes,
   scientific model remains diagnostic; no1–2cMpc/h actual posterior exists.
   Fixed actual rows/masks/selection at1.5 exist, including known small angular
   quadrature errors. Selected CF4 minimumradius15.8152cMpc/h; no CF4/count
   row inside LG R2. Need explicit LG constraints, not claim CF4 fixes it.
2. Mock z0->regularizedIC->PM->z0 bridge: two development cases, density RMS
   .964/.972->.118/.118, velocity333/300->17.8/18.1km/s. Not an IC posterior,
   independent validation, nonlinear halo guarantee or fine-scale closure.
3. TNG100-1 z0 total massive matter including diffuse:400^3@.1875,
   50^3@1.5 in75cMpc/h box; mass/Pxyz/Qdiag and nativeSUBFIND membership.
   Conservative moment encoding preserves mass/momentum/diagonal2ndmoments;
   physical velocity dispersion distinguished from mean-field uncertainty.
4. Finite whole-field bank18 then558600 translated/rotated hypotheses:
   spatial-group support stillfails4/9; closed. Gaussian remainder fails all
   fine-powertests; closed. Continuous transport21mark prototype is kinematic
   only, not a calibrated joint halo+remainder or dynamical prior.
5. Conditional spatial flow with precision/context fixes and proper energy
   score repair stillfails16/16; closed.30k epsilon diffusion blewup; stable
   v-prediction1.49M parameters now generates valid seven-moment fields but
   is not accepted.24k baseline:16/16valid,individualquality0/16,ensemble2/8.
   Fine-lattice+parent-budget-weighted6k matched repair:control0/16,repair1/16,
   ensemble0/8both. Physical bulk-RMS/native1.55->1.52; internal sigma.806->.822.
   First-scale expert3526236000 updates finished8m21s:individual1/16,
   ensemble2/8; bulk1.5566,internal.8060. First .75 split bulkbudgetfraction
   .22645 vsnative.09429. Final legal code arrays identical to native at that
   split. It is not a final-code mismatch, but unique learning cause unknown.
   Raw squared boundary metric is peak-sensitive; modest grid artefacts also
   remain with logdensity. Preserve originalfailures; universalall-sample
   agreement to one native realization is not a calibrated posterior test.
6. Blind density peaks cannot separate M31/M33 in32nativefixtures. Membermass
   U-Net learns one field5000steps but13-fieldgeneralizationfails. Small
   population21coefficient probabilistic MW/M31/M33 POSITION law works in
   spatiallyseparatedoneboxdevelopment(1388train/87cal/127test):jointtestNLL
   9.906vsdensity11.223;M333.440vs3.898. Positionaccuracy1–3cMpc/h, NOT.1875.
   Same-field LG distance/sky likelihood implemented, shared-distance
   covariance/cellintegrals, no oracleparents; gradientscontrols pass.
   It is NOT physical membermasses/COM/subhalo detection or full LG likelihood.
7. Existing actualLG distances/LOS/propermotions with frame/covariance/offset
   caveats. Native mark-conditioning component exists but no spatialposterior.
8. Historical PMWD, HOP, RAMSES, zoom IC/mask/trace code exists. Its validity
   for new state/resolution/phase convention must be checked on reuse, not
   trusted merely because scripts exist. HOP not guaranteed subhalo finder.

Operational: syntax is Slurmlogin; numericalworkonlySlurm(a40/a100/h100/h200,
exclude syn06),or explicitly scoped lageunha. Shared /gpfs ordinary storage;
noGPFS/inode/rename diagnostics/pgrep loops. Memory predictedpeak+20%.
Driver can autonomously do bounded in-goal work, but not silently change route
or launch largeproduction/BundleD. External advice only major decisions;
user now explicitly requests multi-aspect Fable advice for this wholeplan.

Questions all advice must address: Q-GOAL, Q-LEAN; distinguish numerical map
scale/observationalinformation/particlemass/force resolution; proper statistical
target and probability transport; realistic compute with unknown queue time;
NEW field-based MW/M31/M33 identity/ambiguity/missing members and observations
changing THAT state. No nativehalo labels as inference inputs. No handpainted
NFW components passed off as dynamically realizable, arbitraryA(k), gate
loosening, anotheruncappedMLsweep or guarantee of success.

Driver starting judgement BEFORE consultation: failure of these particular
learners is not failure of constrainedLG cosmology. Making universalTNG
superresolution the sole critical path was a planning error. Arbitrary z0
rho/u/sigma is not invertible to uniqueLCDMIC after shellcrossing. Need either
(A) a dynamically supported z0 joint state with explicitlatentphase history,
or(B) a genuine z0 intermediate posterior plus correctlydefined conditional
history inference/consistency correction, not posterior-as-extra-likelihood.
The literalindependentMLmap->reverse route is not currently justified.
This is a hypothesis to scrutinize, not an instruction to rubberstamp.

Relevant public precedents for driver verification (do not fabricate details):
Wempe etal2024, arXiv2406.02228, multiresolutionLG Bayesianfieldinference;
SIBELIUS-DARK2022, arXiv2202.04099, embeddingLG inobservedcosmicenvironment.
They demonstrate narrower relevant capabilities, not our completeCF4+M33goal
or success of a standalonesuperresolutionthenuniqueinversion algorithm.
