# OurUniv

OurUniv infers a ΛCDM initial-condition (IC) posterior with the original
Cosmicflows-4 (CF4) observation-space likelihood applied directly to a PM
forward model. A statistically controlled present-day density/velocity
posterior is retained as a proposal, initialization, preconditioner, and
diagnostic layer rather than as an independent likelihood or a deterministic
truth map. The forward-evolved ensemble is evaluated against:

- the Milky Way, M31, and M33;
- Virgo and Coma;
- the Local Void and Boötes void;
- the observed large-scale peculiar-velocity field;
- the target ΛCDM power spectrum and phase statistics.

The intended deliverables are a calibrated CF4-supported low-k posterior, an
IC posterior that closes under direct CF4 likelihood evaluation, and
phase-consistent zoom families for the Local Group, Virgo, and Coma. A
`<=0.3 cMpc/h` grid or force scale is a numerical resolution target, not by
itself evidence for `0.3 cMpc/h` observational density-phase resolution.

## Authoritative science route (2026-08-29)

The active route is fixed in
[`config/cf4_science_route_v2.json`](config/cf4_science_route_v2.json):

```text
CF4 observations
  |-> RES-CAL truth mocks -> calibrated low-k k_eff and coverage
  `-> full z=0 density/velocity posterior -> IC proposal/preconditioner
                                                   |
LambdaCDM IC prior ---------------------------------+
  -> PM forward F_z0(IC)
  -> original CF4 observation-space likelihood
  -> exact q-corrected importance / MH / SMC transition
  -> p(IC|CF4) proportional to p(IC) p(CF4|F_z0(IC))
```

The active route is hybrid. The z=0 posterior must retain its ensemble and
covariance; it must not be collapsed to one point map and literally reversed.
Because it is derived from the same CF4 data, it must not be multiplied into
the target as though it were an independent likelihood. Pure direct
`CF4 -> IC` inference defines the statistically clean target but is not the
sole search mechanism because of mixing and cost; the z=0 layer supplies the
proposal machinery. Existing direct-route constrained realizations, parent
banks, P1/P2 results, and the unconstrained reference ensemble remain low-k
proposals, controls, and historical evidence only.

For a proposal `q(IC)`, importance weights are
`w proportional to p(IC)p(CF4|F_z0(IC))/q(IC)`; MH must include the exact target
and forward/reverse proposal ratios, while SMC must use declared normalized
targets and reweight/resample/move steps. An implicit proposal with neither an
evaluable `q` nor a proven target-invariant kernel cannot correct the final
posterior.

The Hong-style model family has a valid negative result; that closes only the
specific estimator, not the hybrid architecture. Changing this route requires
explicit user approval and a prior update to the authoritative route record.

## Current status

The active project is at **RES-CAL: effective-resolution and coverage
calibration**. CF4 selection/noise truth mocks must determine the actual low-k
`k_eff` and uncertainty coverage before the z=0 proposal layer is built. Final
scientific approval of even the low-k reconstruction is currently **NO-GO**;
Z0-PROP, IC-HYBRID, and HIRES/ZOOM are blocked in that order. No new compute is
authorized by this status statement.

Random high-k modes, interpolation, super-resolution, and zoom dynamics must
be labelled `conditional-prior` or `numerically-resolved`, never
CF4-observationally reconstructed. An observational `0.3 cMpc/h` claim is
allowed only after independent truth mocks pass preregistered cross-response,
`r(k)`, phase-coherence, residual-power, information-gain, 68/95% coverage,
and held-out-prediction gates over a continuous k-band. Under the cell-scale
convention that band must extend through
`k_claim = pi / 0.3 ~= 10.47 h/Mpc`.

The following is archived direct-route evidence. Priority P0 was completed for
the accepted BGc linear-Gaussian model. The three
observer-aware P1 parent survivors (3023, 3096, and 3429) produced zero
CF4-only P2 pairs in 96 frozen N576 trials. Additional MW/M31 information is
therefore labelled explicitly rather than attributed to CF4.

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

The subsequent cross-code development program is also complete, with a
negative result.  Successive spatial-transport, monotone-calibration,
conditional-flow, and spliced-tail candidates failed their prospective
development gates.  The final V84C0R feasibility audit found no cross-domain
support for another tail-model expansion.  Consequently Astrid remained
unopened, historical EAGLE was never reused for tuning, and neither HOP nor
RAMSES promotion was authorized.  Its historical recommendation to return to
the direct CF4 constrained-realization route is superseded by the hybrid route
fixed above.  See the
[machine-verifiable completion audit](config/hong2021_eagle_goal_completion_audit.json)
and [Hong reproduction history](HONG2021_REPRODUCTION.md).

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

## Archived zoom hierarchy

The `384 Mpc/h` parent uses global L9 (`512^3`). An LG candidate first receives
an L12-particle/L19-AMR pilot. A passing candidate advances to the aggressive
L13-particle/L21-AMR production run (`m_DM=8.86e6 Msun/h`, formal minimum cell
`0.245 physical kpc` at z=0). A science IC requires a stable-particle-ID
traceback mask; the former box-centre sphere is diagnostic only.

The former provisional candidate, parent 3429 / explicit-likelihood seed
5108, completed its L12/L19 DMO run and HOP gate. The zoom was clean, retained
the intended phases, and contained an M33-scale HOP subpeak, but the pair
failed the frozen mass and separation gates (`3.23e12 Msun/h` for the larger
member and `1.224 Mpc/h` separation) as well as the recentered environment
gate. The fixed protohalo-midpoint model subsequently yielded zero fully valid
pairs in 96 development proposals and is closed.

The 64-member latent-midpoint v6 bank is complete. Five realizations passed
the unchanged N576 hard-P2 screen, but none passed all five P1 environment
gates at the evolved pair midpoint; the best, seed 5238, passed four of five.
V6 is therefore closed and cannot be extended with more seeds. Its consumed
evidence is pinned in `config/cf4_lg_v6_result_record.json`.

The statistically normalized v8 program was frozen as the next one-shot bank in
`config/p2_lg_z0_forward_importance_v8.json`. It uses the unchanged v7 z=0
MW/M31 halo-pair likelihood and a 50/50 defensive latent-midpoint proposal,
with exact `log p(q)-log g(q)` correction. A frozen bootstrap audit selected
256 fresh realizations (0.5-percent lower expected ESS 8.47). A candidate must
pass the same hard P2 pair cuts and all five P1 gates at that same pair's
midpoint. The automated job stops before RAMSES regardless of the outcome.
It is now a historical diagnostic and is not authorized for execution or
promotion under the hybrid route.

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
