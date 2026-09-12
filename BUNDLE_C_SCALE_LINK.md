# Scale-localized correction work — autonomous, driver-reviewed

2026-09-13. No routine approval or external audit. The saved boundary
diagnosis is closed, not repeated with another threshold. Keep current
generator NO-GO and the CF4 -> present field -> IC route.

Before changing the generator, ONE frozen-checkpoint comparison distinguishes
the correction target: inherent one-step conditional error versus propagated
coarser-parent error. Reuse all eight saved first draws, whose restrictions
recover their previously generated .375/.75 fields by moment conservation.
Compare those at .75/.375/.1875 against native fields. At .375 and .1875,
also generate ONE step from each true native immediate parent (16 controlled
draws, final EMA frozen). No new full rollout or training. The .75 stage
already has its native1.5 parent, so do not duplicate that calculation.

Report unchanged power and other physical metrics against the same native
1.5 parent. Decompose its fixed velocity-variance budget into resolved bulk
fluctuations and remaining internal variance at each scale; verify their sum.
Report log-density face gradients for the immediate refinement and1.5 grid.
This is a conditional model intervention using truth PARENTS, explicitly not
observed field inference, actual LG conditioning or native role candidates.
Teacher draws are compared descriptively; stochastic differences are not a
proof of unique architectural causes. No post-hoc best-seed/step selection.

One GPU/2CPU/4GiB/10min Slurm, estimated host2.5GiB+20% rounded4; previously
evaluation used1.75GiB, GPU.74GiB. No source slabs cached, outputs JSON only.
Existing four stable-model regressions and physical-budget identity run in
the same allocation. No new generic gate framework. Source literal commit.

Q-GOAL: identify where density and mean/sigma velocity depart before choosing
a substantive correction, avoiding blind longer learning. Q-LEAN: reuse
saved fields/model/tests,16 one-step draws rather than a new model sweep.
MW/M31/M33 identification stays the frozen field-only probabilistic location
law and joint position likelihood; unresolved mass/COM and M33 ambiguity are
not solved here. No truth labels supplied to new-field role inference. Next
correction must preserve this same-field observation connection and seven
moments, not paint a chosen LG or revive old standalone role diagnostics.

Driver will select ONE justified correction or state remaining uncertainty
after this comparison, not automatically launch more diagnostic variants.

Implementation00c5035 committed/pushed; Slurm347160 submitted2026-09-13
02:03:26 KST,1GPU/2CPU/4GiB/10min. Initial PENDING(Priority). At02:04 the
scheduler estimated05:51:37 start; this is provisional, not reserved progress.
Tests and all comparisons execute inside the allocation. No manual node
execution, extra fit or approval wait. Artifacts will be in
/gpfs/kjhan/CF4/z0_density/bundle_c_v1/stable_field_scale_link_v1/;
logs /gpfs/kjhan/CF4/logs/cf4_C_scale_link_347160.{out,err}.
Read result.json AND scores.json before selecting the correction. A finished
scheduler job without complete result is not a scientific pass. There is no
separate monitor daemon or automatically selected next learning job.
