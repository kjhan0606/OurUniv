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

Priority P0, the statistically valid constrained-realization gate, is complete
for the frozen linear-Gaussian model.

- Grid: `N=192`, `L=384 Mpc/h`
- Cosmology: `Om=0.31`, `Ob=0.05`, `h=0.746`,
  `A_s=1.63e-9`, `n_s=0.96`
- CF4 constraints: 19,265 grouped distances
- Posterior: matrix-free Wiener filter plus exact Matheron conditional draws
- Accepted product: 16-member all-data ensemble, seeds 1001–1016
- Held-out test: standardized residual `+0.0077 ± 1.0492`;
  `Delta log predictive density = +815.0` over noise-only

See [the CR gate report](recon/linear_cr/gate_v1/CR_GATE_REPORT.md) and
[the full scientific plan](CODEX_PLAN.md).

Seed 1001 is only the deterministic reference member. No member has yet been
selected using Local-Group morphology or named structures. P1/P2 acceptance
criteria must be frozen before forwarding and ranking the 16 members.

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

- `src/cf4_linear_cr.py` — accepted WF/CR sampler
- `src/cf4_cr_gate.py` — statistical, Gaussianity, and ΛCDM power gate
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

See `IC_RESOLUTION.md` for the exact units, particle counts, and gates.

## Present-density reconstruction reference

The literal Hong et al. (2021) present-density reproduction is tracked
separately in [`HONG2021_REPRODUCTION.md`](HONG2021_REPRODUCTION.md).  The
published `0.3125 Mpc/h` voxel is not treated as a claim of uniform physical
resolution.  The full-width architecture runs locally, while faithful
TNG100-1 training is currently waiting for snapshot-99 particles and group
catalog access.
