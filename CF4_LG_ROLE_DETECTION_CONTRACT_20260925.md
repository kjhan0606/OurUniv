# LG role/detection contract before IC conditioning

Status: **DESIGN ONLY; no actual LG likelihood or IC update.** This records
the 2026-09-25 Fable/Astra advice and driver decision under the active R1–R5
route. The z=0 posterior remains the first science delivery.

For one evolved state F from latent IC, the missing LG factor is

    p(D_LG | F,S) = integral p(theta) sum_(A,B) p(A,B | F,S,theta)
                    p(D_LG | F,A,B,S,theta) dtheta.

S is observer selection and finder resolution. A assigns generated components
to MW/M31/M33 without truth IDs; B describes association and detection. MW
must represent the fixed observer, not an arbitrary distant component. An
unresolved catalogue M33 is still an observed galaxy: its likelihood needs a
latent M33 state tied to F and a calibrated missed-detection probability.
It is not an absent-M33 hypothesis or a constant likelihood floor.

The present role adapter enumerates support and predicts resolved triples,
but has no p(A,B|F,S,theta). The observation contract has only partial
distance covariance. Unknown PM/COM correlations, solar/LMC nuisances,
halo–galaxy centre offsets and distance-dependent PM reduction remain open.
Its 0.01-degree sky tolerance handles source rounding, not model scatter;
its mass assumptions are disabled, not bound-mass data.

| Existing product | What it supports | What it does not calibrate |
| --- | --- | --- |
| One seed40349 L19 DMO/NewGalFinder catalogue | Parser and generated-state component support | Role frequencies, missed M33 rate, or any observational posterior; seed is NO-GO |
| TNG100-1 hydro native Subfind catalogue | Partial galaxy/subhalo demographics and COM development in its own finder | Matched DMO NewGalFinder detection/fragmentation and observer-selected LG prevalence |
| Selected32 TNG coarse-field fixtures | Their peak method misses distinct M33 in all32 | NewGalFinder completeness or unbiased population prior |
| R1 PM/AMR control | Broad unnamed-aperture mass/mean-flow comparisons | Resolved MW/M31/M33 observation discrepancy |

The staged TNG hydro catalogue exists (6,291,349 FoF groups and 4,371,211
subgroups), but the exact checked TNG100-1-Dark groups path is absent. No
tracked product pairs NewGalFinder detection with independent galaxy truth
on an ensemble of relevant forward states. This does not prove that no such
data exist elsewhere; establish an exact source before proposing a transfer
calibration or large run. Do not recopy the staged hydro catalogue.

The fixed5-cMpc/h support in seed40349 contains521 hosts,579 bound components
and193,434,636 ordered assignments. These are counts, not probabilities.
Uniform component weights would privilege finder fragmentation and numerous
low-mass objects; they are allowed only in a labelled toy arithmetic test.
Best-match selection and post-result mass cuts are not posterior inference.
An exact O(n^2) sum requires conditional factorization given MW and explicit
shared nuisances. Pair-dependent M31–M33 association, selection, and general
covariance can restore three-way coupling; importance proposals need known
q and p/q correction with support coverage.

Next bounded science action: establish one matched calibration source or a
defensible documented approximation for observer-to-MW alignment,
galaxy-to-DMO assignment and M33 detection at the actual finder resolution.
Only then compare exact and proposed factorized/importance sums on a tiny
synthetic set, followed by one generated-state feasibility calculation.
**Stop before any actual-data LG weight** if the role law, omitted-support
bound, shared covariance, or state-linked unresolved-M33 term is absent.
Do not substitute another random seed, widened sky error, or more zoom runs.
R2 environment work is separate and still has development-GO/production-
NO-GO selection status.

Q-GOAL: this specifies the missing MW/M31/M33 observation-to-same-state link
that would constrain IC modes. Q-LEAN: a probability contract and calibration-
source decision before further code or large simulation, not another gate
framework or role-counting exercise.

Driver disposition: accept Astra's calibration and conditional-factorization
warnings. Reject Fable's immediate R2+LG-dipole calculation: the local
`cf4_lg_bulkvel.py` value lacks covariance and aperture-to-LG calibration,
and R1's unnamed probes do not validate that likelihood. Reject generic
O(n^2), uniform-role-prior and exact-zero-likelihood claims as unsupported.
Fable correctly notes that R2 is the first science delivery and long precise
RAMSES evolution cannot sit inside every approximate sampling step. Advice
does not override the source-backed driver decision.
