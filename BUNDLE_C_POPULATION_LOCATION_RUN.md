# Population-weighted location comparison

2026-09-12: user approved the next recommended bundle. Plan/review:
`BUNDLE_C_POPULATION_LOCATION_{PLAN,REVIEW}.md`. Implementation uses the staged
native SubhaloPos catalogue, archived branch0 alternatives, total400^3 moments,
fixed12cMpc/h cores at .1875, guarded train/calibration/test slabs. It does not
use observed LG data or revive the closed13-field U-Net.

New code: `src/cf4_population_locations.py`,
`scripts/cf4_bundle_c_population_locations.py`,
`tests/test_cf4_population_locations.py`,
`config/cf4_population_locations_v1.json` and the matching Slurm script.
Four focused regressions cover weighted targets/gradient, normalization and
nonoracle sampling, physical scalar symmetries/boost, periodic footprints/
splits, and scalar calibration. Numerical tests run in the SAME allocation.
Static Python/shell syntax checks pass. All4 numerical regressions subsequently
passed in0.500s inside Slurm347085 (not on the login node).

One21-coefficient fit,12 complete observer epochs/batches8, Adam.02/ridge.1;
every eligible A/T alternative contributes each observer visit. Calibration
fits one mixture coefficient shared by roles; test is never used to fit it.
Final proper scores cover every selected observer/all alternatives. Eight
fixed-before-fit test observers get32 autoregressive draws under EACH of
the calibrated and density laws; no oracle parents or nearest-target selection.

Slurm1GPU/2CPU/10GiB/2h on a40,a100,h100,h200 excluding syn06. Host estimate
8GiB+20% rounded10GiB; GPU8GiB envelope. Application110min TOTAL includes
preparation,70min-capped learning and evaluation, not the sum of two phases.
No process scans, manual node execution or filesystem diagnostics.

Output `/gpfs/kjhan/CF4/z0_density/bundle_c_v1/population_locations_v1/`, new
directory only; preserve old results. Compact metadata/checkpoint/scores/
samples/plots only, no raw snapshot or large probability-volume duplication.

Slurm347085 started2026-09-12 19:08:47 KST on syn05/A40 from pushed source
`aa024b40f7ae95cfe0bdf2d0d03b6d8947395c8d`; allocated1GPU/2CPU/10GiB/2h.
Scheduler end limit21:08:47 KST; the application has its own110min TOTAL cap.
Actual retained counts (all fixed minimums pass):

| Split | MW observers | MW/M31 pairs | MW/M31/M33 triples |
| --- | ---: | ---: | ---: |
| Training | 1388 | 2440 | 5006 |
| Calibration | 87 | 136 | 261 |
| Test | 127 | 206 | 449 |

Source branch0 contains14208 distinct triples /3940 MW observers. Fixed slab
support excludes1989 observers; old-development-footprint exclusion removes
349 more. Cross-split native IDs are disjoint. These remain correlated samples
from ONE simulation box, not1388 independent universes. Calibration observer
x spans42.0912–44.9119 cMpc/h. Training-weighted same-cell reference pi=.2480083;
the eight test observers for paired sampling were frozen before fitting.

## Completed result and driver decision

Slurm347085 COMPLETED/exit0 at19:11:58 KST in3m11s; application181.35s,
learning123.91s. All12 full observer epochs /2088 updates completed. Fixed
features took about30.5s. Host application peak5.689GiB, GPU reserved1.264GiB;
Slurm sampled batch MaxRSS4475772K. No resource/budget failure or retry.
The calibration-only optimum is alpha=1 (raw learned law fully retained).
All five predefined feasibility criteria pass; status
`PASS_POPULATION_LOCATION_FEASIBILITY_ONLY`. Driver accepts this bounded
location comparison and closes it WITHOUT further tuning or learning.

Test127 observers, hierarchical expected conditional NLL (nats; smaller better):

| Role/score | Geometry | Density | Shared-cell reference | Learned/calibrated |
| --- | ---: | ---: | ---: | ---: |
| MW | 7.4613 | 3.0203 | 3.0203 | 2.6903 |
| M31 | 9.4786 | 4.3055 | 4.3055 | 3.7765 |
| M33 | 7.8100 | 3.8976 | 3.7562 | 3.4396 |
| Joint | 24.7499 | 11.2234 | 11.0821 | 9.9064 |

Mean joint training NLL10.6850 versus density12.0591; calibration9.7099 versus
11.0703. The former extreme train/development gap is absent in THESE splits;
do not identify a unique causal cure or compare absolute old/new NLL as if
their fields, context sizes and targets were unchanged. Four tests pass;
all48 probability symmetries pass before, raw-after and calibrated-after on
the fixed training fixture. Maximum before L1=9.514e-7; after=6.475e-8.

Eight frozen test observers each produced32 complete triples under EACH of
learned and density laws (512 triples), zero sampling failures,32 distinct
triples per observer/law. All8 cases improve the descriptive weighted-target
distance for M31/M33; MW improves6/8. This is neither nearest-truth matching
nor independent coverage. Distances to the weighted native alternatives remain
broad: learned MW0.605–1.562, M310.916–3.586, M331.214–3.603 cMpc/h across
the8 cases. A .1875 cell size emphatically is NOT .1875 position accuracy.
The first preselected observer's plot was visually checked: broad candidate
clouds remain; native density in the background is not a reconstructed field.

Evidence in the output directory: `result.json`, `population.json`,
`calibration.json`, `scores_{train,calibration,test}.json`, `rollouts.json`,
the16 frozen sample files, eight `locations_<observer>.png`, checkpoint,
history/exposure and numerical-test/symmetry records. CLI stdout/stderr:
`/gpfs/kjhan/CF4/logs/cf4_C_population_locations_347085.{out,err}`.

Goal contribution: a small SAME-field MW/M31/M33 position probability law now
improves on the frozen density/shared-cell references in spatially separated
one-box data. It does NOT certify a physical halo catalogue, MW selection or
existence law, mass/COM readout, q_F, an observed high-resolution density/
velocity posterior, or IC. M31/M33 scores condition on native preceding roles;
the autonomous draws separately show remaining ambiguity.

Next recommended deliverable, pending user direction: connect LG positional
observations to this law as a SAME-field likelihood, with explicit observer/
existence and subcell-offset treatment, and specify the still-missing mass/
COM and field-prior connection. Prioritize an honest LG-on/off field response
over another location-score repair loop. Do not use q(S|F,d) as an independent
likelihood or call fixed-field reweighting a density posterior. No automatic
new bundle, posterior inference, simulation or IC submission.
