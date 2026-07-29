# CF4 — Local-Universe Density Reconstruction from Cosmicflows-4

> **2026-07-30 status correction.** The high-resolution PMWD sub-cube run below
> is not a Hong et al. reproduction.  A 0.3125-Mpc/h array spacing alone is not
> evidence of that information resolution, and it cannot identify MW/M31/M33
> analogs from CF4.  See `HONG2021_REPRODUCTION.md` for the literal architecture,
> data specification, distance-dependent effective resolution, and audit.

**Goal.** From CF4's biased galaxy distribution + sparse peculiar-velocity field,
recover the present matter density field of the local universe (and ultimately its
**initial** density field, which must be **cubic + periodic**). Method = CIRCLE's
differentiable-PM + learned-inference machinery (`/home/kjhan/BACKUP/CIRCLE`)
applied to real data.

## Methodology (user, 2026-07)

Two coupled branches, sharing one pmwd forward:

**Amortized / learning branch — training-data pipeline (this rebuild):**
```
s ~ N(0,I) --pmwd--> z=0 particles --FoF--> halos --HOD--> galaxies + v_pec
        |                                                        |
        +--> delta_m (present matter field) <==== LEARN ====> galaxy v_pec field
```
A network learns the map (background matter density  <->  galaxy peculiar-velocity
field); at inference it recovers the local matter density from CF4's sparse v_pec.

**Explicit branch:** differentiable forward F(s) + MAP/HMC posterior over the
initial field s (single forward, joint inference — no 2-step). Present ⇒ initial
density done end-to-end.

## Data  (data/)
- `cf4_groups.csv`   — 38053 groups, VizieR J/ApJ/944/94/groups (Tully+2023).
- `cf4_galaxies.csv` — 55877 galaxies, J/ApJ/944/94/table2.
- Downloaded via TAP (tapvizier.cds.unistra.fr); cdsarc FTP host blocked. See
  `data/PROVENANCE.md`.
- `cf4_clean.npz` — 22136 groups after cuts (1<Dist≤250 Mpc, |b|>5°). H0=74.6,
  sig_lnd med 0.221, v_pec/sig_v S/N≈0.6 (⇒ joint forward model required).

## Code  (src/)
| file | role | status |
|------|------|--------|
| `cf4_ingest.py`   | CF4 CSV → clean npz + diagnostic | ✅ rebuilt+run |
| `fof.py`          | Friends-of-Friends halo finder (scipy, periodic) | ✅ self-test PASS |
| `hod.py`          | Zheng07 HOD populator (particle/NFW sat modes) | ✅ self-test PASS |
| `mock_pipeline.py`| pmwd→FoF→HOD→v_pec training-pair generator | ⏳ GPU smoke |
| `cf4_job.slurm`   | Slurm launcher (a10..h200, x64, no-prealloc) | ✅ |

## Target resolution & method (from Hong, Jeong, Hwang & Kim 2021, ApJ 913, 76)
The reference amortized reconstruction (J. Kim is a coauthor) — same 2-channel
observable we use — achieved:
- **Target voxel resolution = 0.3125 h⁻¹Mpc (~312 h⁻¹kpc).**
- Method: tile the volume into **sub-cubes** (TNG100: 20 h⁻¹Mpc→64³; TNG300:
  40 h⁻¹Mpc→128³) and reconstruct each → 0.3125 Mpc/voxel.
- Input 2 channels = **galaxy count + radial peculiar velocity** (== our n_gal+vlos).
- Output = DM density log₁₀(ρ/ρ₀). No Gaussian smoothing. Reflective padding
  (sub-cubes are NOT periodic — unlike our full-box PoC).
- Training on high-res sims (IllustrisTNG). For us: pmwd fine spacing (~0.3 Mpc/h)
  or high-res sub-box + FoF+HOD galaxies.
- 312 kpc roughly separates Local-Group majors (MW–M31 0.77 Mpc≈2.5 vox; M31–M33
  0.2 Mpc≈adjacent) → this is the resolution band for the "resolve M31/M33" goal.
- **Caveat:** 0.31 Mpc is the OUTPUT grid; true small-scale fidelity is set by the
  galaxy sampling density. TNG sub-cubes are dense; real CF4 is sparse → fidelity
  drops. This is exactly why we sparse/noise-degrade the observable to re-validate.

**Plan to reach it:** (a) sparse/noisy re-validation at current L now; (b) then
high-res training sims (pmwd spacing≈0.3 Mpc/h or zoom sub-box) reconstructed in
20–40 Mpc sub-cube tiles at 64³–128³ → 0.31 Mpc voxel; U-Net backbone → diffusion.

## Resolution note
Halo mass ≈ n_members × m_p, m_p = Ω_m ρ_crit (spacing)³.
- spacing 4 Mpc/h → m_p 5.5e12 (halos unresolved: 1e13 = 1.8 ptcl)  ← CIRCLE default
- **spacing 1 Mpc/h → m_p 8.6e10 (1e13 = 116 ptcl, resolved)**  ← use for training
Box must be cubic+periodic and ≥ 2×data_radius+buffer (CF4 to ~250 Mpc ⇒ L≳360 Mpc/h).

## Status / next
1. ✅ Download CF4 data (was missing); ingest → clean catalog.
2. ✅ Build + validate FoF and HOD tools.
3. ⏳ End-to-end mock pipeline smoke on GPU (Slurm).
4. ⏭ Scale mock generation (many seeds, spacing 1 Mpc/h) → training set.
5. ⏭ Train amortized net (density↔v_pec) [reuse CIRCLE wp2_train_cnn / score].
6. ⏭ Connect to real CF4; MAP + posterior (initial field).

**GPU jobs go through Slurm** (`sbatch src/cf4_job.slurm <script.py> [args]`) when
cards are contended — cf. `memory/slurm-for-gpu-jobs.md`.
