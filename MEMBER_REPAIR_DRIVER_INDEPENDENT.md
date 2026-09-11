# Driver independent proposal, frozen before viewing Fable's new proposal

2026-09-11. User requests two independent plans and comparison, not a launch.
This draft is written before invoking/reading the new Fable5 planning response.
It incorporates previous project knowledge; independence means no access to the
other planner's CURRENT draft, not independent data or a new science result.

## Diagnosis and main hypothesis

341713 completed2000 updates, tests3/3,17m34s allocation. Training-role L1
MW/M31/M33=13.44/6.29/17.59 versus baseline about.94/.97/.95. Retained M33
L1=10.50/5.63/9.44 and overlap=8.16e-6/1.23e-6/1.66e-6. Retained host overlap
is only about1e-4. Predictions therefore put very little member mass where
native members actually are. This is NOT just an amplitude calibration error.

For positive maps, e_L1=A+1-2O, where A=sum predicted/sum truth and
O=sum min(predicted,truth)/sum truth. Existing scalar mass, L1 and overlap
already disclose mass excess plus spatial nonoverlap; another expensive
frozen-checkpoint framework is unnecessary. They do not isolate the cause.

Working hypothesis: sparse members within a huge total mass reservoir make
flat randomly initialized four-way allocation plus linear normalized-L1
optimization a poor learning problem. Dense leakage/near-uniform allocations
can dominate mass loss while spatial learning gets suppressed. Background
competition, saturation, loss/gradient geometry and representation limits
remain hypotheses; no claim that adjusting one bias proves the cause.

## ONE repair candidate with a short learning segment inside the same fit

Keep input features, full128^3 support/.1875 resolution,53e4-parameter spatial
backbone, existing13/3 split, no new native data or source-box pass. Change
the allocation parameterization/objective, not network size or grid resolution.

1. Hierarchical positive allocation: a foreground fraction g=sigmoid(a),
   three conditional role shares h=softmax(b), member masses M_r=M*g*h_r,
   remainder M*(1-g). All sums conserved, same-cell M31/M33 allowed. This
   parameterization alone is equivalent in representable support to four-way
   softmax; its hoped-for advantage is optimization, not extra information.
   Initialize constant foreground/role biases from TRAINING integrated mass
   fractions, not heldout masses or spatial member templates. If the head
   weights are zero-initialized, acknowledge initial backbone gradients are
   zero for the first update and require subsequent nonzero learning.

2. Separate each role's total mass and normalized spatial shape in the loss.
   Given T_r=sum truth and P_r=sum prediction, use the per-field/per-role mean

       log(P_r/T_r)^2 + KL(truth_r/T_r || prediction_r/P_r).

   Both terms are dimensionless and zero for the native decomposition. This
   is spatial-shape KL, NOT class-reweighted cell CE that biases member mass.
   Compute log predicted mass using stable log-sigmoid/log-softmax plus log
   total mass, then logsumexp over positive-total cells. Empty-total cells
   contribute zero, never floor physical masses. Preserve fractional native
   targets, not binary member masks. Equal component/field weights, no tuned
   coefficient grid; coefficient1 is a declared design choice, not uniquely
   optimal. KL trains the location of small members even where their predicted
   mass is tiny; log-mass loss avoids the initial linear huge mass-ratio scale.
   Exact sparse support is approached as a limit by sigmoid/softmax; no fake
   zeros/cuts imposed on the generated roles and no foreground truth at test.

3. First300 updates on fixed training fixture0 WITHOUT augmentation, then
   immediately evaluate its mass maps. Proposed engineering screen: every
   member mass ratio.8–1.2, overlap>=.8 and normalized map-L1<=.3; positive
   remainder and conservation still required. This is deliberately a simple
   fixed-input learning question, not generalization or a second model fit.
   Freeze criteria before launch, no threshold shopping. If it fails, save
   maps/why and stop. Insufficient updates remains possible; failure is NOT
   evidence that member separation is scientifically impossible.
   If it passes, CONTINUE that exact optimizer/model for1700 further updates
   with the13 training fields and48 symmetries; no reinitialization, new seed
   or extra audit between segments. Training fixture0 is not heldout; report
   this pretraining exposure explicitly.

Evaluation: preserve original baseline and metrics, add no general framework.
Report masses, normalized L1, overlap and centroid for every role; zero-member
prediction has L1=1 and must not qualify as useful member separation. Proposed
retained M33 requirement: e<.8*min(old_baseline_e,1) in EACH of16/17/20; report
original decision too. Host median errors must not worsen versus baseline,
and host median overlaps>=.5. These are stronger development feasibility
criteria, not physical calibration or a posterior acceptance test. Compare
fixed training-mean-field ablation only as secondary OOD sensitivity evidence.

Resources:1 Slurm GPU,2CPUs,6GiB host,90min cap, <=2000 TOTAL updates across
both segments. Previous run took~16min for learning; proposed20–45min is a
rough estimate allowing log-probability work, not a measured finish promise.
Keep measured/sized memory limits and no larger model. No automatic retries.
One report + actual member maps; no filesystem/process-monitoring diagnostics.

## What this does NOT solve and what comes next

No oracle member locations/IDs at new-field inference. MW is observer-associated;
M31/M33 labels are uncertain hypotheses; shared cells and stochastic role
selection can make a deterministic target insufficient. A single-patch pass
does not establish unique identification on new fields. Failure after a pass
on one patch would point toward generalization/label ambiguity or distribution
issues without choosing uniquely between them.

Member COM velocities, stellar/subcell offsets, boundness and M200c mapping
remain missing. A mass map is not a physical component phase-space state.
For the ultimate CF4/galaxy/LG->z0 posterior->LCDM IC->LG zoom chain, LG
observables must constrain the SAME total field via a valid joint F,S law,
selection E and uncertainty, not just sharpen labels on unchanged mass.
Target remains LG<=.3 cMpc/h and environment1–2; no direct-IC fallback.
If the repaired member learner succeeds, next design joins probabilistic member
state (distinct velocities/offsets) to the deferred stable-denoiser repair;
neither is authorized now. If this candidate fails, do not launch another
blind architecture/seed series; report whether the one-input fit worked and
request a decision about information representation or learning strategy.

Q-GOAL: recover spatial member information needed to constrain the same z0
field, not another cosmologically plausible but unidentified field. Q-LEAN:
one fixed candidate/fit with an internal early-stop segment; reuse source,
backbone, metrics, plots. Main risk: changes confound causal attribution, a
300-step threshold may be too short, and deterministic native roles may not
be sufficiently determined by these moment inputs. No guarantees of success.
