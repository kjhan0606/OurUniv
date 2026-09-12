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
Static Python/shell syntax checks pass; no numerical test pass claimed yet.

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

Status: implementation/static review complete, preparing source-pinned Slurm
submission. Actual counts, test outcomes and job ID will be recorded here.
Even pass is one-box POSITION feasibility, not a halo catalogue, mass/COM law,
q_F or an observed high-resolution density/velocity posterior. No automatic
downstream science run after this fixed comparison.
