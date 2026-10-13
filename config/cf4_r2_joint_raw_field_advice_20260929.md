# Focused advice before replacing the live field's observation target

Read-only audit. Do not edit files, build, submit jobs, run simulations or send
mail. Work only in /home/kjhan/BACKUP/CF4. No repository survey. <=800 words.
Verdict ADVISE PROCEED / MODIFY / STOP / NO VERDICT. Advice is not authority.

Read CF4_R2_RAW_SELECTED_MODEL_20260929.md and these two source modules:
src/cf4_r2_raw_selected_fp.py and src/cf4_r2_raw_selected_fp_jax.py.
Optional: CF4_R2_ZERO_ANCHOR_SOURCE_20260929.md. Do not repeat every old gate.

Important NEW results after the prior consultation: the driver corrected
correlated cut probabilities and preserved the count Schechter LF instead of
substituting a conflicting Gaussian K marginal. Actual same-field raw model:
1414 training marks,219401 exact source/bin components; mu=b0+bM(M+23)+
bN*centered_log1p(richness), one intrinsic SPD covariance,15 common parameters.
It fits on a frozen accepted N128 field in59s/29 iterations. No published eta
factor is multiplied, no data-derived prior. CPU score/gradient checks pass.
Fixed-field raw predictive draws reproduce the four marginal training rank
distributions reasonably: mean ranks(r,s,i,K)=(.497,.496,.501,.507), fractions
below .1=(.095,.091,.093,.088), above .9=(.092,.095,.103,.089). These are SAME
training-data checks, NOT heldout, p-values or physical selection calibration.
Known optical cut acceptance per row is .9725–1. No free per-row corrections.

The next substantive action should connect this raw law to the same LIVE
field, replacing old source-eta factors and their separate .004 zero. All15
population parameters must remain shared and inferred, not frozen at this
fit or turned into an independent prior. At fixed cosmology h=.746, IC prior
is proper LCDM Gaussian;9 existing count nuisances retain their stated priors.
Do not add a redundant separate FP zero alongside the free r intercept.

Proposed proper WEAK regularization, explicitly a modelling choice NOT an
external calibration measurement: intercept means(.3,2.2,2.7) with SD(1,.5,1)
in source log units; luminosity slopes N(0,.5^2), richness slopes N(0,.5^2);
Cholesky diagonal log-scales N(log(.3),1), off-diagonals N(0,.3^2). Fifteen
Gaussian-white coordinates. Initial fit used only for warm start/proposal
metric. Prior sensitivity and absolute-scale/monopole dependence must be
reported. No published shared-FP covariance exists in current inputs.
Dam2020 explicitly treats FP calibration and cosmic velocities jointly,
including free centroids and their monopole covariance:
https://arxiv.org/html/2002.05898 sections3.4/4. This supports a route, not
proof that our nonlinear count/mark/selection model is calibrated.

Known limitations to retain: conditional mark design, constant type/graph
incidence, source groupz as reference only, TSC-count/point-radius coarsening,
8-sigma source support/periodic aliases, physical LOS discrepancy. Current
sigma_LOS~38.5km/s vs3cMpc/h grid may make source-cell quadrature important.
Do not assume these vanish; equally do not demand unknowable perfection as
a prerequisite for ANY conditional inference. No new N256 or long chain yet.

Please assess:
1. Is joint calibration with proper broad priors a defensible next conditional
   field target, without pretending independent absolute calibration? Correct
   any prior/gauge/selection issue that is actually fatal, with its equation.
2. Is a new external absolute-scale prior logically indispensable to obtain
   ANY conditional present-field posterior at fixed cosmology, versus needed
   for data-driven absolute-scale claims? Avoid conflating those questions.
3. Choose ONE necessary next action before live short sampling: live raw-field
   value/gradient wiring, or a same-field source-cell quadrature comparison.
   Do not replace either with an unrelated new cosmological simulation.
4. Give finite operational R2 completion criteria (actual1–2cMpc/h maps and
   uncertainty, mixing, untouched heldout prediction, quantified model/numeric
   limitations). Do not bless a3cMpc/h mechanics pilot as production, but avoid
   infinite unidentifiable-calibration gates that prevent delivery.

Q-GOAL: real CF4-conditioned present state, later SAME-field MW/M31/M33 and
LG<=.3/phase-consistent zoom IC. MW/M31 ambiguous,M33 unresolved; native truth
cannot seed/select candidates. Q-LEAN: prioritize actual field inference and
discriminating observation checks, not accumulating verification frameworks.
