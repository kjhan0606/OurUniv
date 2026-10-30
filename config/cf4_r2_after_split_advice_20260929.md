# R2 remaining-work / prospective larger-compute advice

Read-only science and compute-cost advice, <=1000 words. Do not edit files,
build, submit jobs, inspect processes, walk storage trees, or invoke agents.
This is a prospective larger-calculation decision, NOT routine code review.
Read ONLY CF4_R2_SPLIT_SAMPLING_20260929.md,
CF4_R2_FIELD_LEVERAGE_20260929.md, and these two exact live result files:
/gpfs/kjhan/CF4/z0_density/r2_prior_split_pilot_v1/result.json
/gpfs/kjhan/CF4/z0_density/r2_v6_single_mark_fp_readout_v1/result.json
The pilot may still be running: explicitly distinguish observed prefix from
terminal evidence. Do not invent its endpoint or any effective sample size.

Goal: actual CF4+galaxy-conditioned z0 posterior, environment1–2cMpc/h,
then SAME NEW state with MW/M31/M33 observations and LG<=.3, then phase-linked
zoomIC. CurrentN128/384=3cMpc/h is development; targetN256/1.5 is NOT started.
MW/M31 ambiguous,M33 unresolved, no native truth IDs or known components
assumed in g(F). Their observables must later constrain the very same field.
User asks autonomous continuation THROUGH R2, not another endless diagnostic
ladder. No TNG dependency, heldout tuning, random high-k painting, P(k)
rescaling, likelihood reweighting, or promotion of a MAP into an ensemble.

Joint32-step expanded1414 single-FP-per-physical-group working fit completed
42m57s, F146854->144836, FP-49.306->-31.074, not convergence. Saved terminal
count/FP scores independently reproduce; count4x32/8x32 difference.000344,
relative exposureL1 2e-7, scalar gradient comparison4.3e-8. Training FP
correlation-.0038, shared zero6.19priorSD (.02477dex), residual mean.01918dex.
Those source PDF moments are NOT an established frequentist residual law.
Counts predicted46433vs47121, with distance-dependent residuals. Conditional
FP alone previously favored a homogeneous benchmark; not full-model evidence.

We now run ONE exact-target, fixed-SPD preconditioned prior-split HMC pilot:
16 discarded adaptive warmup+16 fixed-step proposals,2 integrations/proposal.
Full potential/gradient includes IC Gaussian and10 white nuisances. Exact
Gaussian-prior/kinetic flow, nonlinear kicks, full MH correction. Every force
rebuilds live support. Every endpoint compares independent value-only path
against derivative primal, tolerance1e-7. Finite-U bad derivatives stop.
IC Fourier inverse mass [1+5999/|integer_mode|^2]^-1, DC1; nuisance proposal
metric from transformed saved SPD optimizer secants. These are efficiency
guesses, NOT priors or posterior covariance. No mass adaptation during trace.
At6 proposals (prefix, possibly superseded):5 accepted, step~.15,
IC mean square .004747 at start ->.124605, accepted IC white RMS jumps~.12–.19;
fundamental cosine jumps~.001–.005; full gradient~44s. Clear cold-start drift,
not stationarity; white IC power must NOT be forced to1. No ESS claim from16.
Budget90minSlurm/75minapplication,H1002CPU24GiB. Saved scalar diagnostics and
rolling accepted state only. Prior small tests cover reversibility, volume,
Gaussian target mean/covariance, units and rejected states.

Open science: selected association, global rather than calibrated
group/environment LOS, source-FP common fit covariance beyond one .004dex
zero, bias/selection adequacy, independent posterior prediction and target
resolution. Source group redshift defines FP numerator d(z_group), NOT a
second group-velocity likelihood; exactlyONE scored mark/point per group, no
multiple-row precision gain. Do not repeat the blanket claim these rows are
mathematically inadmissible merely because physically grouped. Likewise
absolute log-score magnitudes do not measure their relative information.

Q-GOAL: Which concrete next bundle most efficiently closes the gap to the
actual R2 delivery, rather than accumulating mechanics? Q-LEAN: what MUST be
done before longer sampling/N256, and what can be reported as explicitly
conditional model sensitivity without demanding perfect unknown astrophysics?
Please rank a maximum of3 substantive actions. In particular:
1. If terminal motion persists but power/zero still drift, what bounded
   warmup/mixing experiment is justified before any long production request?
   Do NOT invent a fixed sample count as proof of stationarity. Is there a
   materially cheaper same-target move worth testing, given44s/gradient?
2. Which of the open observational laws is actually identifiable/repairable
   from the available likelihood/data, and what independent evidence is
   indispensable? Distinguish a calibrated physical posterior from the
   conditional working posterior; don't quietly redefine R2 as the latter.
3. Give an honest stop/scale decision and compute envelope. No automatic
   5000-step/large-chain launch; estimates must separate burn-in, mixing,
   field summaries and N256 memory scaling. Current derivative temporary
   memory17.94GiB is not the whole futureN256 peak.

Verdict ADVISE PROCEED / MODIFY / STOP, with concrete evidence and caveats.
Driver independently adopts/amends/rejects advice; it is not automatic authority.
