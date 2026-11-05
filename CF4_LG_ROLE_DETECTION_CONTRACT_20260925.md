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
The checked GalaxyFinder `NewDD/README4GADGET.md` is an *adaptation guide*:
NewDD currently reads RAMSES, not the TNG/GADGET HDF5 snapshot. Applying the
same NewGalFinder pipeline to TNG requires a separately verified conversion
and finder/domain-shift design, not a free invocation of the existing binary.

### TNG-to-finder feasibility decision (2026-09-25)

The local TNG100-1 snapshot-99 header reports 448 files, a 75,000-ckpc/h
periodic box and **6,028,568,000 PartType1 DM particles** (combining its low
and high count words). The existing 37-MB staged particle fixture is instead
one native FoF group (ID468): 604,989 DM particles, 233,820 gas cells,
151,001 stars and five black holes. Its three native Subfind members were
selected for an engineering operator check, not as LG analogues or a random
sample. The separate TNG100-1-Dark counterpart is not at the checked local
path. These are source/header facts, not a search of all storage.

The [official TNG100-1 supplementary download page](https://www.tng-project.org/data/downloads/TNG100-1/)
does provide an all-snapshot hydro-to-Dark **Subfind** matching table (3.6 GB),
and the [official TNG100-1-Dark page](https://www.tng-project.org/data/downloads/TNG100-1-Dark/)
lists the z=0 Dark group catalogue (1.7 GB). The
[specification](https://www.tng-project.org/data/docs/specifications/)
defines `-1` for no match and describes two matching algorithms. These are
credible, smaller **external** sources for a hydro-galaxy to DMO-*Subfind*
baseline; neither file is at the three checked project paths. They do not
contain NewGalFinder outputs or by themselves calibrate its resolution-
dependent M33 detection. No download was started in this feasibility check.

Source inspection confirms that NewDD writes raw `DmType` x-slabs plus
`SN.<step>.<slab>.info`, and opFoF reads those slabs and an ABI-dependent
`RamsesType` header. A direct TNG-to-slab converter is possible in principle,
but **does not yet exist or have a verified ABI/unit test**. More importantly,
running opFoF on the one-group fixture or a spatial cutout changes the
periodic full-volume host-selection boundary. It can test parser/finder
integration but cannot estimate observer-selected MW/M31/M33 role frequencies,
NewGalFinder fragmentation, or M33 missed-detection probability. A whole-box
conversion/finder run would process billions of particles and still need a
matched hydro-to-DMO transfer argument; do not launch it as a quick fix.

**Superseded decision:** the matching table and z=0 Dark catalogue were briefly
proposed as the next optional calibration input. The user then directed that
TNG data be omitted if not essential. They are **not essential to the R2
CF4+galaxy present-field inference**, so no new TNG download, matching job,
converter or finder run is on the active critical path. The prepared but
unsubmitted download/matching code was removed; commit `3453bb8` preserves it
in Git history. Existing TNG fixtures remain historical engineering evidence,
not a source of generated-field truth identities.

This does **not** waive the R3 requirement for a normalized same-state role
law, an M33 state-linked unresolved branch, observational covariance and a
quantified DMO-to-galaxy discrepancy. The active route is to use generated
particle states and actual LG observations directly, with explicit uncertain
role/galaxy mapping; test finder incompleteness and solver-resolution effects
on the project's own forward states. Independent external mocks or TNG can be
used later to validate that mapping, but are not a prerequisite for starting
R2. Until those uncertainties are bounded, a resolved MW/M31 diagnostic is
not a production LG posterior. A single selected halo or uniform role prior
is not an approximation with a quantified error.

The fixed5-cMpc/h support in seed40349 contains521 hosts,579 bound components
and193,434,636 ordered assignments. These are counts, not probabilities.
Uniform component weights would privilege finder fragmentation and numerous
low-mass objects; they are allowed only in a labelled toy arithmetic test.
Best-match selection and post-result mass cuts are not posterior inference.
An exact O(n^2) sum requires conditional factorization given MW and explicit
shared nuisances. Pair-dependent M31–M33 association, selection, and general
covariance can restore three-way coupling; importance proposals need known
q and p/q correction with support coverage.

Next bounded science action: advance the CF4+galaxy R2 current-field route
without a TNG dependency, while separately specifying a defensible
state-linked approximation for observer-to-MW alignment, galaxy-to-DMO
assignment and M33 detectability at the actual finder resolution. Only after
that compare exact and proposed factorized/importance sums on a tiny synthetic
set, followed by one generated-state feasibility calculation.
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
