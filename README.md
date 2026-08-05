# OurUniv

OurUniv reconstructs statistically controlled ΛCDM initial conditions for the
local Universe from Cosmicflows-4 (CF4), then searches the posterior ensemble
for realizations reproducing:

- the Milky Way, M31, and M33;
- Virgo and Coma;
- the Local Void and Boötes void;
- the observed large-scale peculiar-velocity field;
- the target ΛCDM power spectrum and phase statistics.

The intended deliverable is one validated parent realization and
phase-consistent zoom families for the Local Group, Virgo, and Coma.

## Current status

Priority P0 is complete for the accepted BGc linear-Gaussian model. The
observer-aware P1 search is complete through seed 3192. Its two valid parent
survivors produced zero P2 pairs in 64 frozen trials. The final blind parent
extension passed P0 and supplied one new P1 survivor, seed 3429, whose P2 is
running.

- Grid: `N=192`, `L=384 Mpc/h`
- Cosmology: `Om=0.31`, `Ob=0.05`, `h=0.746`,
  `A_s=1.63e-9`, `n_s=0.96`
- CF4 constraints: 19,265 grouped distances
- Posterior: matrix-free Wiener filter plus exact Matheron conditional draws
- Accepted BGc product: 16-member all-data ensemble, seeds 3001–3016
- Held-out test: standardized residual `+0.023 ± 1.012`;
  `Delta log predictive density = +374.9` over noise-only

See [the CR gate report](recon/linear_cr/gate_v1/CR_GATE_REPORT.md) and
[the full scientific plan](CODEX_PLAN.md).

The separate Hong-style present-density reconstruction has reached its sealed
cross-code test.  V13 passes TNG/SIMBA development and historical SIMBA, but
fails the one-time EAGLE confirmation-32 gate through high-k underdispersion
(`0.868/0.629` power ratios) and calibration/environment failures.  Grid-HOP
and CF4 application are therefore disabled; see
[the restart audit](HONG2021_RESTART_PLAN.md) and
[the EAGLE gate record](EAGLE_INDEPENDENT_GATE.md).

The next model cycle now has a committed cross-code data firewall.  Public
CAMELS Swift-EAGLE is development-only, while all 27 CAMELS Astrid CV
realizations remain unopened for a one-time independent V14 gate.  The sealed
EAGLE failure is never used for tuning.  The v2 firewall also removes a V7
target-operator confound by rebuilding TNG, SIMBA, Swift-EAGLE, and eventually
Astrid with one raw-particle CIC definition; CAMELS adaptive-smoothed grids are
not used for V14.  See
[the V14 independent-gate protocol](NEW_INDEPENDENT_GATE.md).

The original P1 gates selected seed 3003 as the only member simultaneously
passing Virgo, Coma, Local Void, and Boötes-void criteria. P2 then found three
N576 morphology-screen passes (small-scale seeds 2001, 2005, 2008), with 2008
ranked first. These are exploratory rather than promotable: generating N512
with the same integer RNG seed redraws its unconstrained Fourier modes. The
L9 check instead projected the canonical N576 field by physical Fourier mode
truncation before PM forwarding.

That audit also found the decisive problem: seed 3003 repeatedly forms
`2--4e13 Msun/h` groups within roughly `3 Mpc/h` of the observer. The old
isolation cut was measured only from a candidate pair midpoint and could miss
them. The frozen v2 P1 scorecard now includes observer-centred excess-mass
bounds at 5 and 8 Mpc/h; seed 3003 and all other members in the first set fail
at least one of the five gates. A same-model 48-member extension (3017--3064)
passed P0, and seed 3023 was its unique five-gate P1 survivor. Seed 3023 then
produced no LG morphology pass in a frozen 32-member N576 small-scale
ensemble, so adding more high-k phases to that parent was stopped. A
conditionally preregistered 128-parent extension (3065--3192) passed P0, with
seed 3096 as its sole five-gate P1 survivor. Its observer environment is mildly
underdense (`delta_R5=-0.317`, `delta_R8=-0.312`), unlike seed 3003. The same
frozen N576 small-scale seeds 2001--2032 and unchanged morphology gates gave
zero passes for seed 3096, for a combined 0/64 across valid parents 3023 and
3096.

The final 256-parent blind extension (3193--3448), frozen in
`config/v3_bgc_parent_extension_v3.json`, passed P0. Seed 3429 was its sole
five-gate P1 survivor and now receives the unchanged 32-seed N576 P2 screen.
Failure of that batch ends blind seed expansion; subsequent MW/M31/M33
conditioning must be declared as information additional to CF4.

### BGc likelihood update

The Hoffman et al. Bias Gaussianization correction is now independently
mock-validated and connected to the velocity-only CR sampler with held-out
rows excluded from its running-median reference pool. A single BGc error scale
of `0.90` was selected on a development split and passed a reserved fresh N64
test. The subsequent fresh N192 test also passed: `z=+0.023 +/- 1.012`,
68/95% coverage `0.708/0.951`, and `Delta log score=+374.9`. The observer's
Gaussian `R=5 Mpc/h` linear density changed from the old central cavity to
`+0.430` in the N192 BGc posterior mean.

The holdout-free ensemble subsequently passed P0, and seed 3003 passed P1.
Large N192 fields and the compact P0/P1/P2 products are stored under
`/gpfs/kjhan/CF4/recon/linear_cr/`.

## Why the reconstruction was replaced

Earlier products used a Wiener-suppressed MAP followed by heuristic
power-completion, or subtracted two nonlinear MAP optimizations as though they
formed a Hoffman–Ribak operator. Neither operation guarantees a sample from
the declared posterior.

The replacement in `src/cf4_linear_cr.py` uses:

1. the Watkins–Feldman log-distance velocity estimator, preserving Gaussian
   distance-modulus errors;
2. a linear ΛCDM radial-velocity response;
3. explicit nuisance marginalization for external bulk flow and the CF4
   distance-scale zero point;
4. a matrix-free Wiener solve;
5. Matheron's rule for conditional Gaussian samples;
6. an untouched held-out posterior-predictive test.

## Repository layout

- `src/cf4_linear_cr.py` — WF15/BGc linear-Gaussian CR sampler
- `src/cf4_bgc.py` — leakage-free fixed-count BGc transformation
- `src/cf4_cr_gate.py` — statistical, Gaussianity, and ΛCDM power gate
- `src/cf4_p2_screen.py` — paired PM/FoF LG morphology screen
- `src/cf4_p2_trace.py` — stable-ID PM traceback for the pilot mask
- `src/cf4_trace_ramses_ids.py` — z=0-to-initial particle-ID trace
- `src/cf4_lagrangian_mask.py` — sparse periodic Lagrangian zoom mask
- `src/cf4_zoom_ic2.py` — phase-consistent multi-level GRAFIC zoom ICs
- `src/cf4_zoom_z0_gate.py` — RAMSES z=0/HOP zoom validation
- `src/` — reconstruction, forward-model, IC, and analysis utilities
- `IC_RESOLUTION.md` — frozen base/pilot/production resolution hierarchy
- `CODEX_PLAN.md` — ordered scientific gates and current status
- `RESULTS.md` — historical experimental record
- `recon/linear_cr/` — compact manifests and gate results only
- `data/PROVENANCE.md` — CF4 catalog provenance

Raw CF4 catalogs, generated NPZ fields, RAMSES snapshots, HOP tags, figures,
logs, and compiled binaries are intentionally excluded from Git. Production
paths and checksums are recorded in manifests instead.

## Reproduce the P0 gate

The project currently uses the `circle` Conda environment:

```bash
PY=/home/kjhan/miniconda3/envs/circle/bin/python

$PY src/cf4_linear_cr.py \
  --N 192 \
  --tag final_test_n192 \
  --sigma-nl 0 \
  --error-scale 0.9 \
  --split-seed 20260731 \
  --sample-seeds 101,102,103,104,105,106,107,108

$PY src/cf4_cr_gate.py \
  --test-manifest recon/linear_cr/manifest_final_test_n192.json \
  --parent-manifest recon/linear_cr/manifest_parent_v1_all.json \
  --outdir recon/linear_cr/gate_v1
```

The commands require the locally prepared `data/cf4_clean.npz`, which is not
distributed in this repository. Its source and construction are documented in
`data/PROVENANCE.md` and `src/cf4_ingest.py`.

## Current zoom hierarchy

The `384 Mpc/h` parent uses global L9 (`512^3`). An LG candidate first receives
an L12-particle/L19-AMR pilot. A passing candidate advances to the aggressive
L13-particle/L21-AMR production run (`m_DM=8.86e6 Msun/h`, formal minimum cell
`0.245 physical kpc` at z=0). A science IC requires a stable-particle-ID
traceback mask; the former box-centre sphere is diagnostic only.

The current provisional candidate is parent 3429 / explicit-likelihood seed
5108. Its L9--L12 `v2_pad6` IC has passed streaming GRAFIC header/record
validation and includes a six-L9-cell file-edge guard around the sparse
refinement mask. The next gate is the two-step RAMSES startup preflight in
`config/ramses_lg_p3429_s5108_pilot_preflight_v1.nml`, followed by a DMO z=0
pilot and HOP validation. A P2 screen is not treated as a final LG detection.

The pad6 RAMSES startup preflight passed with 137,957,344 particles and no
fine-IC boundary warning. Its compact audit record is
`recon/linear_cr/ramses_lg_p3429_s5108_preflight_v3.json`. The z=0 DMO pilot
uses `config/ramses_lg_p3429_s5108_pilot_z0_v1.nml` and is resource-gated so
it does not overlap the existing SIDM3 allocation on Lageunha.

See `IC_RESOLUTION.md` for the exact units, particle counts, and gates.

## Present-density reconstruction reference

The literal Hong et al. (2021) present-density reproduction is tracked
separately in [`HONG2021_REPRODUCTION.md`](HONG2021_REPRODUCTION.md).  The
published `0.3125 Mpc/h` voxel is not treated as a claim of uniform physical
resolution.  The full TNG100-1 snapshot-99 and group catalog passed the local
download validator, the 432/93 paper-sized data split has been prepared, and
the 200-epoch full-width training run is complete.  Held-out density-residual
and 2pCF-KS evaluation is implemented in `src/hong2021_evaluate.py`; the first
run selects the minimum-training-loss checkpoint as in the paper.  Its raw
cosmic-mean-normalized 2pCF does not match the published table, but the result
is not yet a model-failure gate because the paper leaves the finite-cube
normalization convention underspecified.  See the reproduction note for paths,
quantitative results, and the normalization audit.  The from-scratch data,
BatchNorm, split, and retraining plan is in
[`HONG2021_RESTART_PLAN.md`](HONG2021_RESTART_PLAN.md).

The v2 split audit covers all 988 observer candidates.  Three frozen
432/93 splits satisfy cross-split 20-Mpc/h cube non-overlap and have maximum
absolute SMD below 0.10 across six observer/input/target features (v1 was
1.02).  The literal `M_B<-15` split-0 cubes passed deep validation and the
20-epoch Ada pilot is complete.  Epoch 7 passes BN fidelity, but Fourier tests
show loss of correctly phased log-density power above a few `h/Mpc` plus
spurious compact-peak power in linear density.  The pilot therefore does not
pass the high-resolution density gate; see
`config/hong2021_tng100_v2_pilot.json`.  No target stellar-mass cut was added.
