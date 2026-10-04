# R2 bundle disposition — Fable5 applicability audit, 2026-10-04

## Verdict and driver decision

Fable5 returned **CONDITIONAL PASS** for the proposed next R2 bundle. The
driver adopts its applicability correction and bounded shared-factor checks,
but not a claim that the joint count/mark likelihood is already available.
The ray exposure calculation read the predecessor disjoint catalogue
(`r2_common_catalogue_128_v1`: 24,993 training and 8,375 held-out counts),
while the active target is the inclusive v6 split (47,121 training counts)
whose count path uses the shell-CDF/TSC source-to-key kernel. The ray operator
is not wired into that active path. Its NSIDE pairwise statistics therefore
remain frozen historical diagnostics, not a v6 resolution, likelihood, or
posterior gate. No v6 ray rerun or NSIDE4096 escalation is adopted.

## Independently checked evidence

- `CF4_R2_COMMON_COSMOLOGY_CONTRACT_20260925.md` documents the predecessor
  disjoint counts (24,993 train / 8,375 held out).
- `scripts/cf4_r2_raw_joint_pilot.py` reads
  `r2_sky_closed_split_v6/split.npz` and its 47,121 training counts.
- `src/cf4_r2_raw_volume_target.py` routes the active count term through
  `count_field_loglike` and `predict_source_volume_intensity`, i.e. the
  shell-CDF/TSC source-to-key path; it does not consume the ray exposure
  artifact.
- `conditional_mark_log_likelihood_from_shared_latent` in
  `src/cf4_2mpp_joint_likelihood_local.py` implements the normalized
  log-sum-exp identity, but is a generic reference primitive. The current
  `joint_log_likelihood` is explicitly additive/development-only and does not
  pass a common latent into the count kernel.
- `scripts/cf4_r2_count_fp_ownership_control.py` is a one-point saved-field
  control; it checks count-key training ownership and a conditional raw-FP
  score, but does not implement a common latent for the two T10106 members.
  The score-blind identity control records T10106 as 2M++ recnos52802/52824,
  FP rows PGC54049/54054, CF4 1PGC53982, and 2M++ GID2887. These catalog
  associations do not establish physical group membership.

## Adopted bounded continuation (A2–A4)

1. Build one source-only v6 training ownership ledger: each count key/count is
   owned once by the 2M++ count factor; each secure or ambiguous crossmatch is
   classified; each CF4 group mark is identified; and groups are separated
   into count-only, linked-singleton, multi-member/shared-latent, and
   unanchored-conditional classes. A collision/ambiguous association must be
   represented as a mixture or flagged unresolved, never silently duplicated.
   Do not read held-out values or score a field.
2. Inspect the T10106 saved-state mechanics against actual kernel interfaces.
   A shared-latent calculation may proceed only if a well-defined latent node
   controls both the source-to-count-key factor and both group marks with
   compatible measures and a normalized denominator. Do not substitute a
   radius-conditioning argument for a derived common latent or invent a
   covariance, selection, or velocity model.
3. Test zero-information and one-member reductions, compare a real
   two-member shared-latent conditional to the present additive comparator,
   and run one AD/FD check only if the actual common-latent path exists. If it
   does not, report a precise technical NO-GO and missing contract rather
   than a synthetic “joint score.”

The driver rejects repeating any source census already present in the v6
identity/overlap artifacts, re-running the ray exposure on v6 keys, an NSIDE
sweep, another PM evolution, posterior fitting, or held-out scoring. After
this bounded factor-ownership decision, continue to the next material R2
blocker: external or otherwise identifiable calibration of selection,
group inclusion, member covariance and bias/RSD-FoG. The selected mock lacks
the recovered-group/preselection parent needed to estimate those terms;
state them as unidentified rather than tuning them in-sample.

## Goal checks

- **Q-GOAL:** This bundle is necessary observation-law plumbing for the
  CF4-conditioned z=0 field, but is not itself a field result and cannot
  satisfy R2. MW/M31 remain role-ambiguous and M33 unresolved; their
  observables must constrain those same roles in the same NEW evolved LG field
  at `<=0.3 cMpc/h`. Truth identities are calibration/evaluation-only.
- **Q-LEAN:** The source ledger plus one saved-state T10106 mechanics decision
  is proportionate. Stop at a technical NO-GO if the real shared latent is not
  defined by the current interfaces. Do not add another census, a synthetic
  covariance, ray-resolution escalation, PM run, fit, or posterior gate.

Fable's review is advisory. Its useful applicability finding was adopted
because the split counts and active source code independently confirm it; any
future shared-latent factor must likewise be supported by explicit kernel and
measure checks before adoption.
