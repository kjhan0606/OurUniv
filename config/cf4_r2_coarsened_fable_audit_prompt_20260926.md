Read-only science/plan advice for an important R2 observation-model decision. Do not edit files, build, submit jobs, or run simulations. Read CF4_END_TO_END_REPLAN_20260913.md (R1-R5 and R2), CF4_R2_POINT_MARK_CONTRACT_20260926.md, CF4_R2_COARSENED_MARKED_COUNTS_20260926.md, src/cf4_r2_coarsened_observation.py, and the two scoped results:
/gpfs/kjhan/CF4/z0_density/r2_ares_catalogue_bridge_v1/result.json
/gpfs/kjhan/CF4/z0_density/r2_coarsened_observation_v2/result.json

The final goal is an actual-CF4-conditioned present-day density/velocity posterior from latent LCDM ICs, then same-state MW/M31/M33 LG constraints and phase-consistent zoom ICs; surroundings 1-2 cMpc/h, LG <=0.3 cMpc/h numerical map. R2 first science delivery is the z=0 posterior. The original ARES/BORG 2M++ paper (arXiv:1509.05040, equations 1 and 5) uses voxel-integrated selection and counts. Our earlier individual-point model fails at one observed zero-map pixel; the new source bridge shows that same galaxy is in ARES's own example data. At N128 a coarsened count factor has positive integrated cell support for every occupied cell and includes all eligible galaxies. We propose retaining individual redshifts only as covariates for a future CF4 conditional group-mark law. We do NOT claim a complete joint likelihood, calibrated within-cell law, CF4 association/selection, or posterior.

Please answer concisely:
1. Q-GOAL: Does this coarsened-count direction correctly advance the actual CF4/LG goal, or is a full individual-point process scientifically indispensable for R2? State the exact assumptions.
2. Q-LEAN: Is the next essential task association/group selection plus conditional CF4 distance/velocity marks, without more count-only tests, or is there a simpler necessary route?
3. Is p(C|F,S) alone normalized and finite under the stated source selection while the complete p(C,U,A,G|F,S) remains missing? Flag any mathematically incorrect claims in the new document, including conditioning on A or U.
4. How should a later bundle identify MW/M31/M33 from each new evolved field, retain role ambiguity and unresolved M33, and make those observables constrain the same field without native truth IDs?

Verdict: PASS, CONDITIONAL PASS, REJECT, or NO VERDICT for the *development direction*, not production posterior. Cite concrete evidence/files and distinguish essential fixes from deferrable refinements. No automatic authority; the driver will check your reasoning against sources.
