# End-to-end redesign: Fable5 advice and driver disposition

2026-09-13, OurUniv / CF4, DESIGN ONLY. The user explicitly requested an
end-goal plan and multiple Fable consultations if helpful. No numerical
Slurm job or new science route was started by these consultations.

## Actual consultations

All three were completed through the installed Claude CLI with
`--model claude-fable-5 --effort high`, Read-only tools, no MCP servers,
`dontAsk`, no session persistence and JSON output. No credential files were
requested. Each returned `subtype=success`, `is_error=false`, no permission
denials. Reported model usage includes Fable5 and auxiliary Haiku; this is
three consultations of the same primary model, not three independent auditors.

| Consultation | Duration | Prompt | Raw response |
| --- | ---: | --- | --- |
| Science route / probabilistic target | 162.909s | [prompt](config/cf4_goal_reset_20260913_science_prompt.md) | [response](config/cf4_goal_reset_20260913_science.response.json) |
| LG identification / zoom feasibility | 142.517s | [prompt](config/cf4_goal_reset_20260913_lg_zoom_prompt.md) | [response](config/cf4_goal_reset_20260913_lg_zoom.response.json) |
| Corrected joint design / bundle ordering | 117.350s | [prompt](config/cf4_goal_reset_20260913_reconciliation_prompt.md) | [response](config/cf4_goal_reset_20260913_reconciliation.response.json) |

The supplied [evidence packet](config/cf4_goal_reset_20260913_evidence.md)
includes failures, existing capabilities, user priorities, Q-GOAL, Q-LEAN
and the requirement for NEW-state MW/M31/M33 identification without truth IDs.

## Driver checks and decisions

Adopted: remove universal TNG superresolution as the prerequisite for every
scientific deliverable; recommend a dynamically supported joint current-state
and history distribution, with explicit disclosure that its latent-IC
implementation IS Bayesian forward IC inference. This requires user route
ratification, not a renamed restart. The first visible scientific product
remains actual-data z=0 maps with uncertainty. No promise of unique history.

Adopted: use actual LG observations on the same evolving particle state;
model correspondence/ambiguity and physical halo/subhalo observables. Keep
MW, M31 AND M33 in the final target. M33-unresolved partial experiments may
be useful but do not complete the goal. Distinguish map, information, mass
and force resolution, and preserve constrained modes during zoom refinement.

The driver corrected the first two responses before asking the third:

| Claim or implication not adopted | Evidence / correction |
| --- | --- |
| Every generated moment map is provably measure-zero off a dynamical manifold; microscopic gravity loses all past information | Not established. Coarse moment fields omit phase-space/history information; the actual problem is an unverified compatible prior/conditional history, not a proved impossibility theorem. |
| Existing actual 12 cMpc/h sampler already samples through PM | `src/cf4_pm_calibrated_z0.py` uses a calibrated Gaussian z=0 covariance model. `src/cf4_z0_pm_bridge.py` is a separate hardcoded N64/L384 forward; two bridge results are optimization, not posterior sampling. |
| No CF4 or galaxy-count information inside15.8 cMpc/h | CF4 selected minimum is15.8152; counts start at5. Both omit the LG R2 region, but the5–15.8 shell is not count-free. |
| The21-coefficient position law provides calibrated physical marks or conditional IC proposals | It predicts positions on development fields, not bound mass, COM velocity, subhalo identity or a normalized IC proposal. |
| Top-ranked10/100 seeds are a valid posterior importance sample | No known proposal/selection correction was supplied. Preserve proper weights/selection and account for simulation changes; good-looking seed selection is not posterior inference. |
| Existing PMWD/zoom scripts already supply a verified phase-consistent multiresolution posterior backend | Installed PMWD gravity is single-mesh periodic. `cf4_zoom_ic2.py` has measured transfer/log-tail extrapolation, parent interpolation and sub-box modes requiring revalidation, including global covariance and 2LPT. |
| HOP or inherited particle IDs alone establish a surviving M33 | HOP grouping can merge substructure; boundness and generated-state tracking are needed. Native truth IDs cannot select inferred objects. |
| Every member's mass can use M200c; assumed baryonic/LMC offsets can be fixed | Host, bound, enclosed and infall mass differ. Calibrate nuisance scales from source data/matched simulations, not invented numerical allowances. TNG-Dark local availability was not verified. |
| IC level14 needs only2–3 extra levels to reach roughly1 ckpc/h atL384 | Approximately5 additional levels; mass and actual force convergence remain separate requirements. |
| All components exist, so a few days are enough | No measured production PM sampling ESS cost or verified LG multiresolution solver exists here. Published demonstrations do not supply our code or timing. |

Third consultation accepted the factual corrections and the five-bundle
ordering with amendments. Adopted amendments:

- A small same-phase approximate/high-fidelity comparison belongs BEFORE
  actual-data production, to ground the model-discrepancy estimate.
- TNG-calibrated observables need PM-output transfer calibration. Coarse
  field proxies are not resolved MW/M31/M33; the LG likelihood remains a
  substantive open research task.
- Keep nuisance/cosmology dimension modest at entry. Measure sampling cost,
  not merely optimization or gradient throughput.
- Choose the local dynamics/refinement strategy explicitly before claiming
  R3 feasible; coarse-only inference plus properly corrected fine conditional
  sampling is a possible alternative, not permission for arbitrary fine seeds.
- Report actual information gain separately from nominal1.5/.1875 grids.
- Report importance-weight ESS and sample correlation; reweighting collapse
  requires renewed inference, not retaining only a few favored seeds.
- Test global coarse/fine IC covariance and any2LPT coupling; final zoom is
  a pilot/validated handoff, not automatic large production authorization.

Further driver qualifications to the final response:

- A standalone z=0 prior does NOT logically have to be non-dynamical or ML.
  A dynamics-derived marginal with a consistent conditional history is
  possible. The reason to prefer the joint implementation is that the needed
  factorization is not currently available here, not a proof that every other
  route is impossible or that failed learners exhaust all candidate priors.
- Its suggested3–5 cMpc/h effective support and days-to-weeks engineering
  timeline are not project measurements; neither is adopted as a guarantee.
- One mock or short chain tests mechanics; it cannot certify coverage or
  reliably determine production ESS. Proposed alternative samplers and
  MAP/Laplace are options needing their own correctness/approximation account.
- An optional extension of the old12-map is not added to the critical path;
  it would repeat a diagnostic product without closing the LG bottleneck.
- CPU-side numerical tests still use Slurm, never the Syntax login node.

## Evidence and resulting recommendation

Driver inspected the relevant project source and352623 raw result/Slurm
record, and checked primary sources: [Wempe et al.2024](https://arxiv.org/html/2406.02228v1),
[SIBELIUS-DARK](https://arxiv.org/abs/2202.04099),
[MUSIC](https://arxiv.org/abs/1103.6031),
[HOP author documentation](https://lweb.cfa.harvard.edu/~deisenst/hop/),
[TNG specifications](https://www.tng-project.org/data/docs/specifications/),
[PMWD](https://arxiv.org/abs/2211.09958),
and [zoom contamination study](https://arxiv.org/abs/1305.6923).
These support relevant components, not the claim that our combined CF4+M33
goal is already achieved or guaranteed at a known computational cost.

Q-GOAL: the revised design retains actual CF4 environment, all three LG
members, <=0.3 LG maps and usable zoom ICs; no silent all-volume high-resolution
or unrelated feedback/RT target. Route ratification remains necessary.
Q-LEAN: stop the failed ML repair series; five outcome bundles reuse inputs
and existing forward utilities, with one bounded feasibility entry. The
multiresolution dynamics and LG operator are essential science work, not
new generic monitoring/validation infrastructure. No GPFS diagnostics.

Driver verdict: recommend the [revised plan](CF4_END_TO_END_REPLAN_20260913.md),
conditional on explicit user acceptance of its internal-IC mechanism and on
R1 evidence. Feasible in scientific principle; computational mixing and
LG observable fidelity remain unresolved research risks. No blanket GO
for an expensive production inference.
