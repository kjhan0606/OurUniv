# One prospective sampling-feasibility decision, not production approval

Read-only science/cost advice. Do not edit, build, submit jobs or invoke agents.
Read CF4_R2_FIELD_LEVERAGE_20260929.md and scripts/cf4_r2_prior_split_hmc.py.
The current1414-row joint fit408222 is RUNNING; do not invent its final result.
The driver will decide after that fit/readouts. This consultation is triggered
by a prospective first, differently preconditioned sampling calculation, not
routine code review. Q-GOAL and Q-LEAN required. <=900 words please.

Goal remains CF4+galaxy-conditioned actual z0 posterior (environment1–2cMpc/h)
before same-NEW-field LG<=.3 and zoomIC. CurrentN128/3 is development. MW/M31
roles ambiguous,M33 unresolved; later observed constraints must act on that
same state, never truth-identified peaks. No heldout has been consumed.

Completed saved-field evidence: broadened one-FP-per-physical-group working
cohort429+985, no multiple-FP groups or anchors. Groupz is the fixed source
distance reference, not independent velocity data. Same globalLOS parameter
in counts and marks. Conditional-zero-adjusted current score trails homogeneous
benchmark63.78log units; this is descriptive, not Bayes evidence/significance.
Current best fixed-field zero8.23priorSD versus benchmark3.53SD. No prior was
widened. Missing selection/association/shared-FP and group-LOS physics remain.

408222 at12 accepted updates: objective146854.44->145946.36,FP-49.31->-45.61,
max canonical zero-coordinate gradient~10.45,IC L2~599. Not converged. Full
gradient~44s,temporaryGPU17.94GiB,successful actual-size primal and initial
adjoint check3.4e-6. Stop at32updates or60min app,not automatically extended.
Prior-Hessian feasibility had fundamental-x directional curvature~6000 but
14% HVP epsilon variation; randomIC curvature~1.17 but102% vector variation.
These do NOT certify a Hessian/SPD/Laplace. Old identity-mass HMC ESS2–5 closed.

Proposed next bounded feasibility (not posterior promotion): fixed-metric,
Metropolis-corrected prior-split HMC in canonical white coordinates, applying
exact Gaussian-prior/kinetic rotations and full nonlinear likelihood kicks.
The small implemented mechanics pass exact-prior energy/reverse, full
trajectory reverse/volume, known correlated Gaussian mean/covariance and
invalid-support checks. A second run also tests nonconstant Fourier symbol
and conversion from the optimizer's100*white_tracer to canonical white.
These are algorithm controls, not CF4 mixing evidence.

Candidate FIXED proposal inverse mass:
- IC Fourier modes: C(k)=1/[1+(6000-1)*(k_fund/k)^2] for k!=0;C(0)=1.
  This is only a positive, even, inverse-Laplacian-shaped preconditioner
  loosely anchored by the old stiff direction; NOT a measured covariance,
  LCDM power modification, or isotropic Hessian claim. No new Hessian sweep.
-10 nuisance coordinates: saved positive conditional secant metric transformed
  from x=(IC,100*tracers,zero) to canonical units with D^-1 H_x D^-1.
  It may be stale as an efficiency guess but is fixedSPD; acceptance uses the
  exact target, so it does not define the posterior or reuse a data fit as prior.
- Rebuild live candidate support at every force evaluation; exact same target,
  not frozen-neighbor approximation. No IC high-k replacement or tempering.

Pilot candidate32 proposals (first16 bounded warmup, next16 fixedstep),2 leap
steps each; startingstep.1, bounded warmup adjustment based on MH acceptance,
freeze before retained steps. H1002CPU24GiB,application75min/Slurm90min; at~44s
gradient64 calls cost~47min plus compilation/checks. Reject only true
zero-density target trials; finite-target nonfinite derivatives or support
implementation capacity failures stop, not a new hidden posterior restriction.
Keep all rejected states. Save scalar traces/limited checkpoints (not huge
simulation dumps), conditionalzero/ICpower/field summaries for motion and drift.
No ESS certification from16 retained states or claim that MAP stationarity is
required for correctness. Failure/poor motion ->replan,not repeatidentityHMC.

Please decide: is this minimum useful step toward R2 uncertainty, or is a
specific observation-model repair demonstrably prior to ANY sampling of this
working target? If repair takes precedence, specify the concrete missing law
and source or feasible approximation; don't demand perfect unknown galaxy
physics or claim equation20 correct means selection/covariance calibrated.
If sampling is warranted, amend the metric/budget/adaptation only as necessary
and state what16 retained states can/cannot decide. No arbitraryFP reweighting,
prior widening, heldout tuning, extra simulations, TNG dependency or R2 closure.
Do not conflate tiny FP objective movement with zero posterior information.
