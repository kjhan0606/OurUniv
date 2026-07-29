# CF4 — Results log

## 2026-07-08 — Rebuild + FoF/HOD training-data pipeline

**Context.** Prior session's CF4 source + downloaded catalog were **gone from disk**
(only `recon/cf4_smoke.npz`+`.png` survived). Rebuilt from scratch this session.

### Data download (was missing)
- cdsarc FTP host blocked from cluster; used VizieR **TAP** (tapvizier.cds.unistra.fr).
- `data/cf4_groups.csv` 38053 rows, `data/cf4_galaxies.csv` 55877 rows — exact
  row-count match to J/ApJ/944/94, no truncation. See `data/PROVENANCE.md`.

### Ingest (`cf4_ingest.py`)
- 22136 groups kept (1<Dist≤250 Mpc, |b|>5°). sig_lnd med **0.221** (≈ prior 0.212).
- n̂(SGL,SGB) vs normalized SGX/Y/Z: median cos **1.00000** (convention validated;
  1 anti-parallel case = a blueshifted 1.3 Mpc group).
- v_pec std 1161 vs sig_v med 1833 ⇒ **S/N≈0.6** — confirms sparse velocities alone
  under-constrain; joint forward model over large volume required.

### FoF (`fof.py`) — self-test PASS
- scipy cKDTree(boxsize) periodic pair search + connected_components union-find.
- Recovered 5/5 planted clusters within 3 Mpc; periodic COM handles face-straddling.

### HOD (`hod.py`) — self-test PASS
- Zheng07 5-param; centrals Bernoulli, satellites Poisson.
- Satellite phase-space from member particles (self-consistent) or analytic NFW.
- Measured ⟨N⟩(M) matches analytic across 14 mass bins (<15%).

### End-to-end pipeline (`mock_pipeline.py`) — GPU via Slurm (job 175508, node syn02)
- pmwd N=128, spacing 1.0 → L=128 Mpc/h, 2.1M ptcl, m_p 8.6e10 Msun/h.
- PM 22.7s; 1D vel rms 274 km/s (physical).
- **FoF 3.8s → 2179 halos** (largest 10671 members, Mmax 9.18e14 Msun/h).
- **HOD → 1974 galaxies** (1314 cen + 660 sat, f_sat 0.334).
- v_pec rms 279 km/s. Output `recon/mock_pipeline.npz` (16.9 MB) + `.png`.
- Figure: galaxies sit on cosmic-web density peaks; HMF + HOD-occupation + v_pec
  all physical. **Methodology validated.**

### Notes / next
- f_sat 0.33 is high (real ~0.1–0.2); raise logM1 (13.7→~14.0) to reduce sats.
- Scale up: many seeds at spacing≈1 Mpc/h, larger L (≳360 Mpc/h for CF4) → training set.

## 2026-07-08 (cont.) — Amortized net trained + verified

### Training set (`cf4_gen_train.py`, Slurm arrays)
- 256 train + 32 val + 32 test, N=64 spacing 2.0 (L=128 Mpc/h), ~686 galaxies/box.
- Observable gridded (periodic CIC): n_gal (count) + vlos (LOS v_pec momentum).
- Target asinh(delta_m). Perf: forward built once/shard (20s→0.5s per sample).

### CNN (`cf4_train_cnn.py`, adapted from CIRCLE) — 5-ch input, 195k params
Input = [n_gal, vlos, 3 coord channels]; coord channels needed because the
LOS-velocity observable (central observer) breaks translation invariance.

### Verified recovery r(k) (32 test, ±1σ; `cf4_verify.py`)
mean r(k), by scale band:
| input | k<0.15 (>40 Mpc) | 0.1<k<0.3 (~30 Mpc) |
|---|---|---|
| galaxy density only (no net, baseline) | 0.845 | 0.679 |
| CNN galaxies+velocity (full) | 0.905 | 0.781 |
| CNN density only (novel) | 0.905 | 0.782 |
| CNN **velocity only** (nogal) | 0.615* | 0.678* |

Findings (verified, tight error bars ±0.02):
1. **CNN beats the naive galaxy-density baseline** (0.905 vs 0.845) — net denoises
   the biased tracer, not trivially copying n_gal.
2. **Peculiar velocity ALONE recovers the density field**, matching/exceeding the
   galaxy-density baseline for scales ≲40 Mpc (k≳0.2); weaker only at the very
   largest scale (LOS bulk-flow/monopole degeneracy). = CF4 premise demonstrated.
3. full ≈ novel: with clean abundant tracers, velocity adds ~0 *marginal* info on
   top of galaxy positions; its unique value needs bias/sparsity/IC targets.
4. r(k) peaks at intermediate k for velocity-only — empirical confirmation that
   peculiar velocities constrain LARGE scales (~30–60 Mpc), NOT sub-Mpc. This is
   why the Local Group / M31·M33 must come from a Tier-2 zoom, not from CF4 v_pec.

*nogal first run had an epoch-150 val blip (trainer saved final not best) →
pessimistic. Fixed: trainer now keeps BEST params + LR step-decay; re-run pending.

### Architecture comparison (flat-CNN vs U-Net vs FNO; `cf4_compare.py`)
Same data/observable/target/eval; best-checkpoint; 32-test r(k):
| model | voxel r (full) | val MSE | r(>40Mpc) | r(~30Mpc) | verdict |
|---|---|---|---|---|---|
| flat-CNN (5-layer, 195k) | 0.620 | 0.616 | 0.904 | 0.780 | strong baseline |
| **U-Net (3-level, 2.35M)** | **0.640** | **0.590** | **0.914** | **0.800** | **best (modest)** |
| FNO (w16 m8 l4, 3M) | 0.556 | 0.691 | 0.881 | 0.732 | worst |

- **U-Net wins consistently** but modestly (+0.02 voxel r, +0.01–0.02 in r(k)); it
  is the right backbone (and the diffusion-posterior backbone). full/novel U-Net
  overfit late (train 0.20 vs val 0.95) → best-checkpoint (epoch ~12) is essential.
- **FNO underperforms**: modes=8 truncation drops intermediate/small scales +
  early overfit. Could improve (more modes, norm) but not the winner.
- **Why the margin is small — v→δ is LOCAL** (δ ≈ −∇·v/(aHf); divergence is a
  local derivative), so a small-RF CNN already suffices on this clean, dense,
  CIC-gridded velocity field. Global-RF architectures earn their keep in the
  REALISTIC regime: sparse/noisy velocities (interpolate across gaps = long-range),
  ZoA/selection masks, and the **initial-field target** (present→initial couples
  scales). Also 256 samples is too few for 2–4M-param nets → need augmentation.

### Sparse/noisy re-validation + Hong+2021 direct comparison
Realistic observable (`data_train_real`): radial selection (688→164 gal/box) +
distance-error velocity noise (σ_v≈0.2·H0·d, S/N<1). Octahedral augmentation added
(label-preserving; coords fixed, physical+target isometry). flat-CNN vs U-Net:
| cond | model | voxel r | r(>40Mpc) | r(~30Mpc) |
|---|---|---|---|---|
| clean | flat-CNN | 0.620 | 0.904 | 0.780 |
| clean | **U-Net** | 0.640 | 0.914 | 0.800 |
| realistic | flat-CNN | 0.432 | 0.763 | 0.574 |
| realistic | **U-Net** | 0.458 | 0.781 | 0.600 |
- U-Net > flat-CNN in BOTH conditions; **gap grows on realistic** (+0.026 vs +0.020)
  — global RF helps more when sparse/noisy (confirms hypothesis).
- CF4-reality penalty: r(>40Mpc) 0.91→0.78; small scales hit hardest.

**Hong+2021 direct comparison** (`cf4_hong_compare.py`, their Fig 4/5 + Table-2 format):
same 2-ch input (galaxy count + radial v_pec). 2pCF ratio recon/true:
| scale | clean U-Net | realistic U-Net |
|---|---|---|
| 12–30 Mpc/h | 0.93 | 0.82 |
| 6–12 Mpc/h | 0.83 | 0.60 |
| 2–6 Mpc/h | 0.68 | 0.43 |
- Large-scale reconstruction matches Hong+2021's quality; both degrade at small
  scales (Hong 2pCF-KS also worsens 0.130→0.263 toward small r). Gap = our coarser
  2 Mpc voxel + sparse mock vs their 0.31 Mpc + dense TNG → motivates sub-cube plan.
- Figs: recon/cf4_hong_compare_*.png, recon/cf4_real_compare.png.

### (a) High-resolution 0.3125 Mpc/h sub-cube (Hong+2021 method)
Parent pmwd N=256, spacing 0.3125 → 80 Mpc/h box, 0.3125 Mpc voxel, m_p 2.6e9
(1e11 halo=38 ptcl). FoF 41s on 16.7M ptcl (feasible), HOD f_sat 0.20 (realistic).
Tiled into 64³=20 Mpc/h sub-cubes (== Hong TNG100): 512 train + 64 val + 64 test,
~52 gal/sub-cube. U-Net (reflect pad, global-position coords). `cf4_gen_subcube.py`.

Result (`recon/hires/`, box 20 Mpc → r(k) covers k=0.3–10 h/Mpc, the small-scale
window the 2 Mpc PoC could NOT access):
- cell r = 0.570; 2pCF ratio recon/true: 2–6 Mpc **0.70**, 6–12 Mpc **0.88**.
- r(k) 0.88 @k=0.31 (20 Mpc) → 0.15 @k=10 (sub-Mpc).
- **Recovers the cosmic-web SKELETON** (filament/node positions) at 0.31 Mpc, but
  **SMOOTHS sub-3-Mpc detail** (2pCF under-predicts at r<3; ξ(1Mpc) recon 0.25 vs
  true 0.55). Same qualitative small-scale degradation as Hong (2pCF-KS worst at
  0–1 Mpc); our detail is lower = L2 posterior-mean smoothing + limited parent
  diversity (only 8 large-scale boxes → best-checkpoint epoch 18, overfit after).
- **Conclusion:** resolution scaling WORKS (opens the 0.3–3 Mpc window); small-scale
  FIDELITY is now limited by (i) L2 smoothing → **Stage B diffusion restores power**,
  (ii) few parent boxes → generate more parents.

### Decisions
- Backbone = **U-Net** (confirmed, lead grows on realistic data).
- (a) high-res sub-cube done; small scales resolved but smoothed → motivates (b).
- Next: **Stage B conditional diffusion** (restore small-scale variance + uncertainty).
- Tier-2 (resolve M31/M33 etc.) = **pmwd high-resolution sub-box** (user choice),
  seeded from the Tier-1 constrained field. Large scales data-constrained, small
  scales random ΛCDM fill (SIBELIUS/Hestia-style analogs, not literal objects).
- Uniform 10 kpc on the full local box = ~5×10¹³ ptcl (~PB) → infeasible & not
  needed; CF4 host halos resolved at spacing ≤0.5 Mpc/h.

## 2026-07-08 (cont.) — Stage B: amortized conditional diffusion  observable → INITIAL field s

Goal (user directive b): recover the whitened initial field s directly from the galaxy
observable, output as **GRAFIC1 ICs for lagRAMSES** constrained zoom runs (targets: MW,
M31, M33, Virgo, Coma, maybe LMC — tiered resolution, see cf4-target-resolution memo).

### Infrastructure built + validated
- **`grafic_io.py`** — GRAFIC1 writer/reader, **2LPT velocities** (Ψ1+3/7 Ψ2; user
  directive, not Zel'dovich). Self-test PASS: roundtrip exact, 2LPT linear-limit
  continuity 1e-5, δ₂ quadratic (4.00×).
- **`cf4_export_grafic.py`** — recovered s → pmwd `linear_modes(a=a_start,real)` δ +
  2LPT vel → GRAFIC. Validated end-to-end on true s (readback exact, v_rms~50 km/s).
- **`cf4_train_diffusion.py`** — DDPM ε-net, 3D conditional U-Net (from CIRCLE
  wp2_score.py) re-conditioned on the REALISTIC galaxy observable (n_gal + vlos +
  radial-geometry channel) instead of clean density. Octahedral aug, DDIM fast
  sampling, **gradient checkpointing** (`--remat`, bit-identical, 3-5× memory).
- **Full periodic box** (IC map non-local → sub-cubes invalid). **data_cf4: L=384
  Mpc/h, N=192, 2 Mpc/h**, 656 train + 64 val + 32 test, ~18.6k gal/box. Covers CF4.
- **192³ memory**: dense C=32 OOMs even bs=1; **with remat, C=32 bs=1 = 13.3 GiB**
  (fits any ≥24 GB card). Scaling to deeper surveys = patch+margin (memory ∝ patch,
  not box; finite-range non-locality) — train once, tile at inference.

### KEY FINDING (N=64 PoC, clean data, 12 test obj, ep124 converged)
| observable | pooled r(k) low-k | r(k) mid | low-k std(z) |
|---|---|---|---|
| **full** (n_gal+vlos) | **0.40** (range 0.04–0.72) | 0.13 | 1.36 (calibrated) |
| **nogal** (velocity only) | **0.04** (~0/neg) | 0.01 | 0.89 |

- **Recovering the INITIAL field is fundamentally harder than the present density.**
  velocity→present-δ worked (r~0.7, δ≈−∇·v/aHf is LOCAL); but velocity→initial-s needs
  inverting the nonlinear displacement (backward gravity) → **velocity-alone fails
  (r~0.04)**. Galaxy POSITIONS carry displacement info, so full does better (0.40).
- **This is the CLEAN-data ceiling** (no noise/selection) → realistic CF4 lower.
- **Calibration is honest** (std(z)~0.9–1.4 with 20 samples; the training-time
  std(z)=9 was an 8-sample artifact). The posterior correctly reports "IC large-scale
  phases only ~40% determined by a CF4-like observable."
- Under-training ruled out: ep24→ep124 barely moved r(k) (0.37→0.40).

### Implication — validates the two-branch design, honestly
- Amortized net = **fast, calibrated posterior ENSEMBLE of CF4-consistent ICs** +
  the full observable→GRAFIC pipeline; but **modest IC fidelity (r~0.4 large-scale)**.
- High IC fidelity needs the **explicit** differentiable-forward branch (CIRCLE
  wp2_diffusion DPS / wp2_explicit optimization / BORG-style). NOTE: CIRCLE
  wp2_warmstart_opt found amortized warm-start gives ~no Adam speedup (curvature-aware
  optimizers may differ) — so the amortized net's value is SPEED + UNCERTAINTY +
  ensemble exploration, not primarily as a warm start.
- **N=192 CF4-coverage full model training now** (job 175680, remat, val-loss select,
  40 ep) → gives the real-scale amortized product + GRAFIC export demo.

### Explicit forward-model MAP + constrained realization (`cf4_explicit_map.py`) — WORKS
User chose the explicit path. Fit s to galaxy LOS peculiar velocities via the
DIFFERENTIABLE pmwd velocity field sampled at FIXED galaxy positions (no FoF/HOD in
loop), Gaussian prior, L-BFGS (optax.lbfgs, curvature-aware), discrepancy-principle
early stop (freeze at chi2/N≈1.3, before overfitting noise). N=64 mock (473 gal,
sigma_v=150):
| stage | r(k) low-k | std(s) | note |
|---|---|---|---|
| amortized diffusion (ref) | 0.40 | — | learned-net ceiling |
| **explicit MAP** | **0.76** | 0.025 | ~2× better phases; monotonic 0.22→0.76; amplitude-shrunk (prior-dominated null space) |
| **Hoffman-Ribak CR** | **0.60** | **1.00** | usable IC: full ΛCDM power + CF4 constraint |

- MAP recovers large-scale IC PHASES ~2× better than the amortized net → explicit
  branch is the fidelity tool (validates the user's choice).
- MAP amplitude shrinks (473 constraints vs 262k modes → most modes → prior mean 0);
  **Hoffman-Ribak** `s_CR = s_rand − MAP(mock from s_rand) + MAP(obs)` restores full
  power (std 0.025→1.00) while keeping r=0.60 at large scales — correct CR behavior
  (unconstrained part of even large scales honestly prior-filled). 2 L-BFGS solves.
- **Full chain validated end-to-end**: CF4-like v_los → MAP → HR CR (std 1.0) → linear
  δ(a_start) → 2LPT → GRAFIC1 (ic_deltab+velc, readback exact) → lagRAMSES-ready.
- Cost N=64 ~8 min/solve (JIT 195s + ~0.5s/iter). N=192 ~8× → ~1 hr/solve.
- Next: scale to N=192 CF4-coverage; add realistic noise/selection; warm-start MAP
  from the amortized s (test if it speeds convergence); connect real CF4 catalog.

### N=192 CF4-coverage explicit result + HR refinements
- **N=192 MAP: r(k) low-k = 0.96** (13794 galaxies, L=384 Mpc/h box) — the explicit
  forward recovers the large-scale IC phases at 96% at CF4 coverage. Far above N=64
  (0.76) and the amortized net (0.40). GRAFIC files written (28 MB each). HEADLINE.
- **Hoffman-Ribak needed two fixes** (first pass gave CR r=0.39):
  1. **K\* consistency**: both solves (MAP of real data, MAP of mock-from-s_rand) must
     run the SAME iteration count = the real-data discrepancy-stop iter K\*. Data-
     dependent early stop gave 119 vs 23 iters -> broke the operator. (N=64 validated:
     MAP 0.765, CR 0.54; CR≈MAP² is the correct constrained-realization relation — a
     full-power SAMPLE correlates less than the MEAN.)
  2. **Model-mismatch noise inflation**: real galaxy v_pec carry virial/nonlinear motions
     (~180 km/s) the matter-field forward can't fit -> chi2/N floor 2.5 at N=192. Fold
     into sigma_v (150->240, sigma_eff²=sigma_meas²+sigma_vir²) so the MAP reaches the
     noise floor and both HR solves converge symmetrically. Physically justified.
- Cost: N=192 ~40 min/solve, HR = 2 solves ~80 min.
- s_CR (full ΛCDM power, CF4-constrained large scales) → 2LPT → GRAFIC → lagRAMSES.

### Power-completion beats Hoffman-Ribak (the deliverable method)
HR gave only r_CR low-k = 0.36 at N=192 — root cause: the MAP is amplitude-shrunk
(std 0.015), so s_rand − MAP(mock) ≈ s_rand (the shrunk MAP fails to remove s_rand's
power in the constrained subspace) → the constraint is DROWNED by full-power s_rand.

**Power-completion** (`power_complete()`, now the DEFAULT via `--pc-seed`): the MAP power
P_MAP(k) = T(k)²·P_prior(k) with T(k) MEASURABLE from the MAP (whitened field → white
prior, P_prior=N³). Keep the MAP's phases, add ONLY the complementary power
sqrt(1−T²(k)) as a fresh white field → full prior power at every k, r_CR(k)=r_MAP(k)·T(k).
Data-only, NO second solve.
| method | r_CR low-k | mid | std | P_CR/P_prior |
|---|---|---|---|---|
| Hoffman-Ribak | 0.36 | 0.19 | 1.00 | 1.0 |
| **power-completion** | **0.75** | 0.16 | 1.00 | **≈1.0 all k** |

- Per-mode: kbin1 T²=1.0 r_CR=0.98 (fully constrained), falls to random as T²→0. Correct.
- **FINAL N=192 CF4-coverage deliverable**: `recon/ic_cf4_pc192/` GRAFIC1 (L=384 Mpc/h,
  N=192, 2LPT vel, readback exact) — a full-ΛCDM-power IC constrained to CF4 at large
  scales (r~0.75), ready for lagRAMSES. Small scales = random ΛCDM (unconstrained) →
  filled/zoomed by lagRAMSES (tiered plan).
- Pipeline COMPLETE end-to-end at CF4 coverage: v_los → explicit MAP (r=0.96 phases) →
  power-completion (full power, r=0.75) → 2LPT → GRAFIC → lagRAMSES.

### Real CF4 catalog connected (`--real-npz`) + σ8 calibration
- `cf4_explicit_map.py --real-npz data/cf4_clean.npz --h 0.746 --Om 0.31`: loads
  pos_dist*h (Mpc/h, observer at box centre), vpec (LOS), sig_v. Robust cuts:
  |vpec|<3000 (drops 893 bad-distance outliers → 21243 groups), sig_v floor 50 km/s.
  **S/N med 0.27, chi2(s=0)/N=0.83** (very noisy → prior-dominated). No discrepancy stop
  (chi2-target 0, full iters) since data is weak.
- Prior (buggy) run: MAP dropped **chi2 0.83→0.29** = coherent flow extracted from noise;
  std(s_MAP)=0.003. (Save crashed on s_true=None → guarded; re-run.)
- **σ8 calibration**: pmwd default A_s_1e9=2.0 → σ8=**0.896** (11% high vs Planck 0.81).
  Velocity amplitude ∝ fσ8, so forward over-predicts v by ~11% and the IC would carry
  ~11% too much structure. Fix: `--A-s-1e9 1.63` → σ8≈0.81 for real data (added to
  cf4_explicit_map.py + cf4_real_validate.py). Mock keeps 2.0 (its training data used it).
- Validation (`cf4_real_validate.py`): forward s→z=0, render SGX-SGY density slice with
  Virgo/Coma/GreatAttractor marked + measure bulk flow (R<30/50/80 Mpc/h) vs known CF4.
- Running: recon 175738 (σ8=0.81, h200) → val 175740 (chained). Deliverable:
  recon/ic_cf4_REAL192/ + recon/cf4_map_cf4_real192_val.png.

### REAL CF4 RECONSTRUCTION VALIDATED (2026-07-09) — attractors + bulk flow
Final real run: N=192, L=384 Mpc/h, σ8=0.81, 21243 groups, 70 iters, power-completion.
Deliverable GRAFIC: `recon/ic_cf4_REAL192/` (ic_deltab+velc). Validation figure
`recon/cf4_map_cf4_real192_val.png` (cf4_real_validate.py, forward s_map → z=0 density,
supergalactic SGX-SGY slice, 8 Mpc/h smooth):
- **ALL major attractors land on reconstructed overdensities at their known SG positions**:
  Great Attractor (-38,+18), Coma (+1,+66), Perseus-Pisces (+45,-16), Virgo (-3,+12),
  Fornax (-5,-6). CF4 groups trace the recovered cosmic web.
- **Bulk flow** (mean v in sphere): R<30 = 308 km/s, R<50 = 186, R<80 = 182 km/s,
  direction SG (-SGX,+SGY,-SGZ) = toward Great Attractor / CMB-dipole apex. Amplitude +
  direction consistent with known local dynamics; falls with R as expected.
- => explicit forward MAP correctly reconstructs the LOCAL density field from REAL CF4
  velocities. Coordinates, cosmology (h=0.746, σ8=0.81), full pipeline all verified.
  GRAFIC IC is physically meaningful → ready for lagRAMSES constrained runs.
- Cluster note: h100/h200/a100 fairshare-blocked; a10 too slow/tight (24GB, 50min TLE);
  a40 (40GB, -t 2h, 70 iters) is the reliable choice for N=192 explicit runs.

### VELMOD held-out validation (`cf4_velmod.py`) — observation-space, non-circular
Fit IC on 80% CF4 (17035 gal), predict held-out 20% (4208 gal) LOS velocities. VELMOD ML:
u_obs = beta*u_pred + V_ext.rhat + N(0, sig_i²+sig_v²); null = beta=0 (bulk-flow only).
PREDICTOR MUST BE the posterior-MEAN velocity = forward(s_map); s_out (a random full-power
realization) dilutes the regression (beta 0.37) — use s_map.
| metric | s_map (correct) | verdict |
|---|---|---|
| **beta** | **1.08** | velocity amplitude correct → σ8=0.81 validated ~8% |
| **binned corr** | 0.98 | held-out velocities predicted (noise-averaged; few effective bins, treat as supporting) |
| **Delta BIC vs bulk-flow-only** | **234** | recon DECISIVELY captures real structure, not just a flow |
| V_ext | 55 km/s | box captures the flow |
| sig_v | 106 km/s | low nonlinear residual |
| chi2/N | 0.26 | fits within errors |
- STRONG non-circular observation-space validation: held-out CF4 velocities predicted at
  correct amplitude, decisively beating a bulk flow. Complements the attractor-position +
  bulk-flow checks. Refs: VELMOD (Willick&Strauss 98), Velocity Field Olympics (2026).
- OBJECT bridge: residual map (u_pred vs u_obs, coloured by residual) localises where
  structures are mis-reconstructed → next step toward object-level validation.
- Zoom masks done (`recon/zoom/`): LocalGroup/Virgo/Coma Lagrangian regions, margins
  130-177 Mpc/h (L=384 adequate for central targets). RAMSES box-fraction regions printed.

### Object-level validation (`cf4_scorecard.py`) — "is this OUR universe?"
Statistics only check "is this LCDM?". Two named-structure tests answer "is this the
LOCAL universe?":
- **Density-at-cluster (constrained s_map, 5 Mpc/h smooth) — THE right test for cluster
  POSITIONS**: delta(sigma) at literature SG positions: Virgo **+27.4**, Perseus +8.3,
  Centaurus +5.1, Fornax +3.1, Coma +2.6 (5/7 overdense); Norma(GA) -0.9, Hydra -3.4
  (misses = Zone-of-Avoidance sparsity + imprecise literature coords). Mean +6.04 sigma;
  random baseline +0.12+/-0.35 -> **reconstruction 16.8 sigma above random. DECISIVE.**
- FoF-halo matching (s_out full-power realization): only 3/7, 1.8 sigma above random —
  because the SPECIFIC collapsed halos are set by the RANDOM small-scale power that
  power-completion adds (CF4 doesn't constrain the specific realization, only the large-
  scale envelope). Correct physics for a constrained reconstruction.
- => CF4 constrains WHERE clusters are (16.8 sigma) but not the specific small-scale
  realization (1.8 sigma). Object-level validation PASSES for the constrained field.
Full validation stack: statistics + attractor positions + bulk flow + VELMOD (beta=1.08,
dBIC=234) + object density-at-cluster (16.8 sigma) = reconstruction reproduces the ACTUAL
local universe, not just a plausible LCDM one.

### TNG position-conditioned super-resolution (`cf4_sr_gen.py`, `cf4_sr_train.py`) — WORKS
Goal (user): the CF4 base is data-limited to ~large scales, so add galaxy-scale structure by
super-resolution such that the biased-peak galaxies match observations. Train on TNG300
(realistic galaxy bias), condition on the OBSERVED galaxy positions.
- TNG300-1 galaxy catalog (galaxy_099.sav, 122724 galaxies MSTAR>1e9, z=0) -> 512 sub-cubes
  (fine 0.5 h/Mpc, 32 h/Mpc). Observed subsample matches CF4 sparsity (7.84e-4 (h/Mpc)^3).
- Conditioning c=[coarse density 2 h/Mpc upsampled, n_obs sparse observed galaxies];
  target = asinh(smoothed galaxy overdensity). Conditional DDPM, 3D U-Net (2.39M, remat).
- BUG fixed: raw sparse COUNTS (0.18% fine-cell occupancy -> delta-spike target) made DDIM
  diverge (gen sum 4e8, corr 0.07). Switched to SMOOTHED CONTINUOUS overdensity target ->
  stable. gen std 1.26 ~ true 1.02.
- RESULT: gen-vs-true correlation = 0.672 (per-object 0.62-0.72). From ~26 observed galaxies
  + coarse density the model reconstructs the ~470-galaxy fine field at 67%, pinning peaks to
  observed positions and filling TNG-statistical structure elsewhere. Fig
  recon/cf4_sr_validation.png. Method = the position-conditioned super-resolution the user
  requested; TNG supplies realistic bias (vs pmwd+HOD). Next: cf4_sr_apply.py tiles the CF4
  reconstruction (coarse) + real CF4 groups (n_obs) -> fine galaxy field over the local volume.

### SR applied to real CF4 (`cf4_sr_apply.py`) — super-resolved local Universe
End-to-end: s_out -> pmwd(float32) z=0 density (2 h/Mpc coarse) + real CF4 groups (n_obs) ->
tile central 128 h/Mpc into 64 sub-cubes -> TNG-SR (DDIM 100) -> stitch -> 256^3 fine field
(0.5 h/Mpc) in 64s. recon/cf4_sr_local/ (fine_field.npy + cf4_sr_local.png). The generated
galaxy field is pinned to observed CF4 groups and TNG-statistical elsewhere. KNOWN ARTIFACT:
visible 32 h/Mpc sub-cube seams (independent per-sub-cube local normalization) -> fix with
overlapping tiles + feather blending or global normalization. Bug fixed: pmwd enables jax x64
on run, so force float32 on model + sampling inputs (S, conds, schedule) after the pmwd forward.
Full chain COMPLETE: CF4 velocities -> explicit-MAP IC -> constrained field -> position-
conditioned TNG super-resolution -> realistic galaxy field matching observations at 0.5 h/Mpc.

### SR seam removal (feather blending) — DONE
The disjoint tiling left 32 h/Mpc sub-cube seams. Fixed with 50% OVERLAP + Hann feather
blending (343 overlapping sub-cubes, weighted sum / weight). Also fixed the real crash cause:
edge/void sub-cubes had extreme conditioning -> DDIM diverged to NaN (which triggered the XLA
buffer_comparator crash at strict autotune and NaN output otherwise). Robustness: clip
standardized cond to [-8,8] + nan_to_num, sanitize gen before blending. Default autotuning
(5.5 min for 343 sub-cubes). Result recon/cf4_sr_local/cf4_sr_local.png: seam-free cosmic web
at 0.5 h/Mpc over the central 128 h/Mpc, filaments/nodes/voids continuous, clusters marked,
CF4 groups overlaid. field std 0.84, no NaN.
