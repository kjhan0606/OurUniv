# IC and AMR resolution specification

This file freezes the first zoom hierarchy for the `384 Mpc/h` CF4 parent.
All lengths below are comoving unless explicitly marked physical.

## Frozen hierarchy

| stage | global base | finest zoom-particle level | DM spacing | DM particle mass | runtime `levelmax` | smallest cell at z=0 |
|---|---:|---:|---:|---:|---:|---:|
| pilot | L9 (`512^3`) | L12 | `93.75 ckpc/h` | `7.088e7 Msun/h` | L19 | `0.982 physical kpc` |
| production | L9 (`512^3`) | L13 | `46.875 ckpc/h` | `8.860e6 Msun/h` | L21 | `0.245 physical kpc` |

For the adopted `h=0.746`,

```text
dx(level) = 384 / 2^level Mpc/h
m_DM(level) = Omega_m rho_crit dx(level)^3
rho_crit = 2.775e11 (Msun/h)/(Mpc/h)^3
```

The production L13 load represents a `1.2e12 Msun/h` MW halo with about
`1.35e5` particles and a `1e11 Msun/h` M33 halo with about `1.13e4`
particles.

## RAMSES terminology

The production namelist must use global `levelmin=9`. L13 describes the
finest **initial particle load inside the LG Lagrangian region**, not a
full-box RAMSES `levelmin`. The intended production relationship is:

```text
full box:       L9
nested ICs:     L10, L11, L12, L13 inside the traced patch
runtime AMR:    through L21 where the refinement criterion permits it
```

The L21 cell size is an upper limit on force/cell spacing, not a claim that
the physical solution is converged to 245 pc. Particle mass, time stepping,
refinement criteria, and two-body relaxation still require convergence tests.

## Mandatory zoom gates

1. Identify an LG halo pair at `z=0` in the parent/pilot run.
2. Select stable particle IDs belonging to MW, M31, M33 and a buffered
   `3--5 Mpc/h` Eulerian environment.
3. Trace those IDs to the initial snapshot and construct a periodic,
   non-spherical Lagrangian mask.
4. Run L12/L19 to `z=0`; require a valid HOP/M200c LG, the M33 subpeak, and
   less than the declared contaminant fraction inside each `R200c`.
5. Only then regenerate the same long-wave realization at L13 and run to
   L21.

A sphere placed at the box centre is not a valid science mask. The rejected
cr6/e19 L14 run used that shortcut and contained no finest particles at the
screen-selected LG.

LG, Virgo, and Coma use the same parent phases but separate Lagrangian zoom
runs. The Local and Boötes voids are validated in the parent volume and do
not require L13 particle loads.

The machine-readable source of truth is
`config/ic_resolution_v1.json`.

## Execution sequence

The parent/pilot namelist must write a particle output immediately after IC
loading (`aexp <= 0.03`) as well as the final `z=0` output. After HOP identifies
the LG midpoint:

```bash
PY=/home/kjhan/miniconda3/envs/circle/bin/python

$PY src/cf4_trace_ramses_ids.py \
  --initial-output /path/to/output_initial \
  --final-output /path/to/output_z0 \
  --center-mpc-h X_LG Y_LG Z_LG \
  --radius-mpc-h 5 \
  --out /path/to/lg_trace.npz

$PY src/cf4_lagrangian_mask.py \
  --input /path/to/lg_trace.npz \
  --out /path/to/lg_mask_L9.npz \
  --box-mpc-h 384 \
  --base-level 9 \
  --buffer-mpc-h 1.5
```

Generate the pilot:

```bash
$PY src/cf4_zoom_ic2.py \
  --tier pilot \
  --parent /path/to/accepted_parent/level_010 \
  --parent-level 10 \
  --transfer /path/to/accepted_parent_transfer.npz \
  --mask-npz /path/to/lg_mask_L9.npz \
  --seed SMALL_SCALE_SEED \
  --out /path/to/lg_L12_pilot
```

After the pilot passes HOP and contamination gates, use the same parent and
mask with a preregistered production seed:

```bash
$PY src/cf4_zoom_ic2.py \
  --tier production \
  --parent /path/to/accepted_parent/level_010 \
  --parent-level 10 \
  --transfer /path/to/accepted_parent_transfer.npz \
  --mask-npz /path/to/lg_mask_L9.npz \
  --seed SMALL_SCALE_SEED \
  --out /path/to/lg_L13_production
```

The generator prints the correctly shifted, one-based `initfile()` list and
the corresponding RAMSES `levelmin`/`levelmax`. Its former rejected
cr6/e19 paths are no longer command defaults.
