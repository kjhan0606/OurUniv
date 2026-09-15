# Repair proposal: current specification and audit disposition

2026-09-11. This specification supersedes conflicting details in the exact
submitted `BUNDLE_C_REPAIR_PROPOSAL.md`, which is preserved unchanged for audit
provenance. User requested a proposal, NOT implementation or Slurm submission.

Fable5 completed normally in111090ms: CONDITIONAL GO for the one member-map
pilot; stable denoiser repair is design-only. Full request/response:
`config/cf4_repair_proposal_fable5_v1.{txt,response.json}`. Q-GOAL and Q-LEAN
accept one reusable mass-map readout rather than another peak/proxy diagnostic.
No backup auditor, no code implementation or numerical work launched.

## Fixed choices answering the audit

1. For field i and role r, normalized map error is

       e_ir = sum_cells |Mhat_ir - Mtrue_ir| / sum_cells Mtrue_ir.

   Training decision uses the UNWEIGHTED arithmetic mean across the13 original
   unaugmented fields, separately for each of4 roles. Require learner mean
   <=0.8 times baseline mean for EVERY role. No pooling native mass over fields,
   no selecting good rotations. Development M33 requires strictly lower e_ir
   on EACH of fixed fields16,17,20; MW and M31 each require nonworsening median
   e_ir across those three. Remainder errors also reported; no role-averaged
   statement disguises M33 failure. Mass relative error is |Mhat/Mtrue-1|,
   overlap is sum min(Mhat,Mtrue)/sum Mtrue, and centroid error is Euclidean
   distance between component mass centroids in the full24-cMpc/h patch,
   in cMpc/h. A centroid is NOT a stellar position. No zero baseline division:
   zero baseline error requires equally zero learner error for training and
   cannot yield a claimed strict development improvement.

   Missing/zero-mass required member in any fixed fixture means
   INCONCLUSIVE_TARGET_UNAVAILABLE, reported per case. Do not drop/replace it,
   change denominator, or quietly score only the easier positive subset.
   The existing selected fixtures are expected positive; inference on arbitrary
   unselected fields is outside this conditional-on-E pilot's support claim.

2. Baseline uses the canonical UNROTATED observer-cell frame. At every cell,
   average TRAINING native fractions Mtrue_ir/Mtotal_i across all13 fields with
   equal field weight. Empty-total cells have member fractions0/remainder1
   by convention; their actual mass contribution is zero. This gives one
   fixed4-channel map, evaluated on canonical unaugmented training/development
   total masses. No augmentation-averaged or heldout-tuned template. Optional
   displayed transformed versions are the same frozen baseline, not new fits.

3. Driver correction beyond the auditor's per-field-normalization request:
   the submitted class-balanced cross entropy is NOT appropriate for unbiased
   physical mass allocation. At a cell it minimizes

       -sum_r (Mtrue_ir / Mtrue_ir_total) log w_ir,

   whose optimum is proportional to Mtrue_ir/Mtrue_ir_total, rather than the
   desired native mass fractions Mtrue_ir/Mtotal_i. Thus it can deliberately
   inflate the small satellite fraction even with perfect training. Replace
   it by the mean of the4 PER-FIELD normalized map-L1 errors e_ir above.
   The exact native decomposition has zero loss; role balancing then does not
   move that optimum. Use the standard subgradient of abs; no epsilon mass
   floor, class-reweighted softmax interpretation or loss-weight search.
   This is still a deterministic supervised mass readout, not q(S|F) training.
   The original audit evaluated cross entropy; this algebraic correction is
   explicitly the DRIVER'S revision, not a claim the auditor tested the code.

4. Keep the proposed widths16/32/64, <=1M parameters, one seed and one fit,
   <=2000 updates,70min training within ONE90min Slurm job. Sizing actual
   parameter/state/activation buffers is required during implementation BEFORE
   submission. Current host6GiB and GPU24GiB envelopes are estimates. If the
   concrete implementation cannot fit those envelopes, report the mismatch
   before launch; no silent model/cap expansion. No model constructed/benchmarked
   on syntax in this proposal turn. No repeated external preflight jobs.

5. A pass leads to ONE joint-member-state plus stable-denoiser-repair design,
   with distinct member velocities/subcell offsets and observation coupling,
   not another readout variant/radius/seed. Further field-prior training is
   NOT authorized by passing this mass-only pilot. Failure/budget truncation
   closes this attempt and requires a user decision; it does not prove that
   the universe or overall LG objective is unrecoverable.

## Limits on the audit's causal language

Accept that observed denoising failed, not an identified unique optimization
bug. Also do NOT adopt the stronger claim that this mass-map learner's failure
would make LG conditioning impossible: explicit latent components, additional
local information and probabilistic ambiguity could still support inference.
A deterministic mass readout cannot certify full member-state learnability.

v-prediction supplies a numerically better conditioned reconstruction formula;
it does not guarantee noncollapsed representations, valid fractions or LCDM
morphology. Detaching category gradients is an UNCONFIRMED competition remedy.
If subsequently authorized, the bounded clean-reconstruction learning check
must compare against zero-v at intermediate noise and report stem/head gradient
RMS using existing small diagnostics, not build a new monitoring framework.
No zero-v/shrinkage baseline qualifies as cosmological learning or actual LG.

## Proposed deliverable and approval boundary

Next authorization would cover ONLY implementing and running the one member
mass-map pilot, its essential allocation/augmentation tests in the SAME job,
one compact report and per-role native/predicted/baseline maps. No completed
z=0 posterior, member velocity solution, production training, actual-data fit,
IC generation, new simulations or entry to Bundle D is claimed or authorized.
Final objective and original scientific acceptance standards stay unchanged.

The current proposal and diagnosis records will be committed/pushed. Wait for
user approval at the bundle boundary; say '승인해주세요' when asking to execute.
