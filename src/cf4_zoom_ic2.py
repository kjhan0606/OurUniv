#!/usr/bin/env python
"""Two-scale, Lagrangian-mask zoom IC synthesis (pmwd-free).

For a fine level L (dx = L_box/2^L, sub-box only) the displacement splits:
  * LONG range (|k| <= parent Nyquist): interpolate the parent ic_velc* (already the
    2LPT tidal field) onto the fine sub-box grid.  Carries the LG-pair-scale motion,
    identical to the running parent -> realization preserved.
  * SHORT range (|k| > parent Nyquist): fresh small-scale power x the measured transfer,
    then 2LPT displacement on the (bounded) sub-box.  ALL fine levels draw their white
    noise from ONE finest-grid field (seed) via Fourier truncation, so the shared k-band
    is IDENTICAL across levels (nested-consistent; verified xcorr=1, max rel diff ~1e-16).
    The old per-level independent draws (seed+L) disagreed by a full octave in the shared
    band -> refinement-boundary artifacts.  These modes do not exist in the parent, so
    they are new (dwarf-scale) structure and are UNCONSTRAINED by CF4 -> the seed is a
    free realisation (change it alone for an ensemble; environment stays fixed).

Normalization (validated by parent round-trip): a unit-variance real white-noise field
w gives, via  delta = irfftn( rfftn(w) * B(k) ),  physical power P(k)=B(k)^2 dx^3.  The
parent has delta = irfftn( rfftn(s) * T_eff(k) ) at dx_par, so P_par=T_eff^2 dx_par^3, and
to reproduce it on a grid of cell dx (=> full-box-equiv N=2^L):
      B(k) = T_eff(k) * (dx_par/dx)^1.5 = T_eff(k) * (2^L / N_par)^1.5 .
This factor is the same (Nf/Nc)^1.5 embed_ic uses and is independent of the sub-box size.

Coarser-than-parent levels reuse the plain-downsample path (see cf4_zoom_ic.py, P0).
Science runs require a sparse mask made by ``cf4_lagrangian_mask.py`` from
z=0-selected particle IDs traced to the initial snapshot.  The old geometric
box-centre sphere remains available only behind an explicit diagnostic flag.
"""
import os
import sys
import argparse
import json
import numpy as np
from scipy import fft as sfft

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grafic_io as G
from cf4_zoom_ic import fourier_resample_field, snap_subbox, build_refmap
from cf4_lagrangian_mask import (
    level_bounds,
    load_sparse_mask,
    refmap_for_level,
    sha256_file,
)


# ------------------------- transfer function -------------------------
class Transfer:
    def __init__(self, npz):
        z = np.load(npz)
        self.kphys = z["kphys"]; self.T = z["T"]; self.kNyq = float(z["kNyq"])
        self.coeff = z["tail_coeff"]; self.N_par = int(z["N"])
        self.kf_par = float(z["kf"])
        # last measured k with signal
        good = (self.T > 0) & (self.kphys > 0)
        self.kmax_meas = self.kphys[good].max()

    def eval(self, kmag):
        """T_eff at |k| [h/Mpc]: measured (interp) below kNyq, log-log tail fit above."""
        out = np.interp(kmag, self.kphys, self.T, left=0.0, right=0.0)
        hi = kmag > self.kNyq
        if np.any(hi):
            out[hi] = 10.0 ** np.polyval(self.coeff, np.log10(np.maximum(kmag[hi], 1e-6)))
        out[kmag <= 0] = 0.0
        return out


def _kmag(N, L_hmpc):
    """|k| [h/Mpc] on the rfftn grid (N,N,N/2+1), built by broadcasting in float32 so
    the 2048^3 finest level does not materialise three full meshgrid arrays."""
    kf = 2.0 * np.pi / L_hmpc
    kx = (np.fft.fftfreq(N) * N * kf).astype(np.float32)
    kr = (np.fft.rfftfreq(N) * N * kf).astype(np.float32)
    return np.sqrt(kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + kr[None, None, :] ** 2)


def hier_rfftn(Nf, seed, workers=-1):
    """Draw ONE unit-variance real white-noise field on the finest sub-box grid (Nf^3)
    and return its rfftn (complex64).  This is the single source of every fine level's
    small-scale modes: coarser fine levels take a strict SUBSET of these modes
    (truncate_rfftn), so every physical mode |k|<=kNyq_L has an IDENTICAL amplitude at
    every level -> nested-consistent.  numpy/scipy rfftn is unnormalised: E|Wf|^2 = Nf^3."""
    rng = np.random.default_rng(seed)
    wf = rng.standard_normal((Nf, Nf, Nf), dtype=np.float32)
    return sfft.rfftn(wf, workers=workers)


def truncate_rfftn(Wf, Nf, N):
    """The N-grid rfftn as a strict subset of the finest-grid rfftn Wf (same physical
    box, N a power-of-two divisor of Nf): keep modes |m_i|<=N/2 and renormalise
    (N/Nf)^1.5 so the coarse field is unit-variance real white noise on the N-grid."""
    if N == Nf:
        return Wf
    h = N // 2
    idx = np.r_[0:h, Nf - h:Nf]                      # fftfreq order: +0..h-1, then -h..-1
    Wc = Wf[np.ix_(idx, idx, np.arange(h + 1))].copy()
    Wc *= np.float32((float(N) / Nf) ** 1.5)
    return Wc


def synth_delta_fft(Wc, L_hmpc, N_equiv, N_par, tr, kcut, workers=-1):
    """delta = irfftn( Wc * T_eff(k) * (N_equiv/N_par)^1.5 ), high-passed to k>kcut.
    Wc is the level's nested white-noise rfftn from truncate_rfftn; the physical power
    is identical to the old real-noise synth_delta (same B, verified by round-trip)."""
    N = Wc.shape[0]
    km = _kmag(N, L_hmpc)
    B = tr.eval(km).astype(np.float32) * np.float32((float(N_equiv) / N_par) ** 1.5)
    B[km <= kcut] = 0.0
    return sfft.irfftn(
        Wc * B, s=(N, N, N), axes=(0, 1, 2), workers=workers
    ).astype(np.float32)


def interp_parent(field, L_box_hmpc, off_hmpc, N_sub, dx_sub_hmpc, chunk=64):
    """Trilinear-interpolate a periodic parent field (Np^3) onto a sub-box grid
    (N_sub^3, cell dx_sub, corner off_hmpc). Returns (N_sub,N_sub,N_sub).
    CHUNKED over the first axis: the old full-grid meshgrid was 3x N_sub^3 float64
    (~400 GB at N_sub=2048 -> near-OOM on lageunha).  Now coords is only
    (3, chunk*N_sub^2) at a time (~6 GB at chunk=64), so peak memory is the output
    array itself (N_sub^3 f32), not the coordinate scratch."""
    from scipy.ndimage import map_coordinates
    Np = field.shape[0]
    dx_par = L_box_hmpc / Np
    off = np.broadcast_to(np.asarray(off_hmpc, dtype=np.float64), (3,))
    loc = [((off[axis] + (np.arange(N_sub) + 0.5) * dx_sub_hmpc) / dx_par - 0.5)
           for axis in range(3)]
    out = np.empty((N_sub, N_sub, N_sub), dtype=np.float32)
    # 2D j,k plane (reused every i-chunk); meshgrid(indexing='ij') ravel order:
    #   I = repeat(loc, N^2), J = tile(repeat(loc,N), .), K = tile(loc, .)
    Jf = np.repeat(loc[1], N_sub)                  # J on one (N,N) plane, raveled
    Kf = np.tile(loc[2], N_sub)                    # K on one (N,N) plane, raveled
    for a in range(0, N_sub, chunk):
        b = min(a + chunk, N_sub)
        ni = b - a
        coords = np.vstack([np.repeat(loc[0][a:b], N_sub * N_sub),
                            np.tile(Jf, ni), np.tile(Kf, ni)])
        out[a:b] = map_coordinates(field, coords, order=1,
                                   mode="grid-wrap").reshape(ni, N_sub, N_sub)
    return out


def _level_window(level, levelmin, box_hmpc, center_hmpc, half_hmpc, mask):
    """Return three-axis integer bounds for a level."""
    n = 2 ** level
    if level == levelmin:
        return np.zeros(3, dtype=np.int64), np.full(3, n, dtype=np.int64)
    if mask is not None:
        return level_bounds(mask, level)
    i0, i1, _ = snap_subbox(
        n, 2 ** levelmin, box_hmpc, center_hmpc, half_hmpc)
    return np.full(3, i0, dtype=np.int64), np.full(3, i1, dtype=np.int64)


def _slice3(lo, hi):
    return tuple(slice(int(lo[a]), int(hi[a])) for a in range(3))


def _parent_fingerprint(parent):
    result = {}
    for name in ("ic_velcx", "ic_velcy", "ic_velcz"):
        path = os.path.join(parent, name)
        stat = os.stat(path)
        result[name] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return result


def _write_manifest(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--parent", required=True,
                    help="accepted parent GRAFIC level directory; no rejected-run default")
    ap.add_argument("--parent-level", type=int, default=10)
    ap.add_argument("--parent-grid-size", type=int, default=None,
                    help="actual canonical source mesh; permits e.g. N576 as a "
                         "Fourier source without treating it as a RAMSES level")
    ap.add_argument("--transfer", required=True,
                    help="CAMB-based parent transfer NPZ")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resolution-config",
                    default=os.path.join(root, "config", "ic_resolution_v1.json"))
    ap.add_argument("--tier", choices=("pilot", "production"), default="pilot")
    ap.add_argument("--levelmin", type=int, default=None,
                    help="global RAMSES base level; defaults to frozen config L9")
    ap.add_argument("--levelmax-ic", type=int, default=None,
                    help="finest zoom-particle IC level; defaults to selected tier")
    ap.add_argument("--runtime-levelmax", type=int, default=None,
                    help="RAMSES AMR levelmax; defaults to selected tier")
    ap.add_argument("--box-hmpc", type=float, default=None)
    ap.add_argument("--h", type=float, default=0.746)
    ap.add_argument("--Om", type=float, default=0.31)
    ap.add_argument("--OL", type=float, default=0.69)
    ap.add_argument("--astart", type=float, default=0.02)
    ap.add_argument("--mask-npz",
                    help="sparse z=0 particle-trace mask from cf4_lagrangian_mask.py")
    ap.add_argument("--resume", action="store_true",
                    help="resume only when the stored run specification matches exactly")
    ap.add_argument("--allow-geometric-mask", action="store_true",
                    help="DIAGNOSTIC ONLY: permit the rejected box-centre sphere shortcut")
    ap.add_argument("--center-hmpc", type=float, default=192.0)
    ap.add_argument("--mask-R-hmpc", type=float, default=6.0)
    ap.add_argument("--subbox-half-hmpc", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=1970,
                    help="seed for the ONE nested white-noise field (change alone for an "
                         "ensemble; unconstrained by CF4)")
    ap.add_argument("--fft-workers", type=int, default=-1,
                    help="SciPy FFT worker count; use a positive cap on a shared host")
    args = ap.parse_args()

    with open(args.resolution_config) as handle:
        resolution = json.load(handle)
    tier = resolution["zoom_tiers"][args.tier]
    Lc = (resolution["global_base_level"] if args.levelmin is None
          else args.levelmin)
    Lf = (tier["finest_ic_particle_level"] if args.levelmax_ic is None
          else args.levelmax_ic)
    runtime_levelmax = (tier["runtime_amr_levelmax"]
                        if args.runtime_levelmax is None else args.runtime_levelmax)
    args.box_hmpc = (resolution["box_size_mpc_h"] if args.box_hmpc is None
                     else args.box_hmpc)
    if runtime_levelmax < Lf:
        ap.error("--runtime-levelmax cannot be below --levelmax-ic")
    if args.mask_npz is None and not args.allow_geometric_mask:
        ap.error("science ICs require --mask-npz; use --allow-geometric-mask only for diagnostics")
    mask = (None if args.mask_npz is None else
            load_sparse_mask(args.mask_npz, expected_box=args.box_hmpc,
                             expected_base_level=Lc))
    Npar = (2 ** args.parent_level if args.parent_grid_size is None
            else args.parent_grid_size)
    Lbox_mpc = args.box_hmpc / args.h
    h0 = 100.0 * args.h
    kcut = (Npar // 2) * (2.0 * np.pi / args.box_hmpc)          # parent Nyquist [h/Mpc]
    tr = Transfer(args.transfer)
    if tr.N_par != Npar:
        ap.error(f"transfer N={tr.N_par} differs from parent source N={Npar}")
    a, Om, OL = args.astart, args.Om, args.OL
    H_a = h0 * np.sqrt(Om / a ** 3 + OL)
    Oma = Om / a ** 3 / (Om / a ** 3 + OL); f1 = Oma ** 0.545
    mask_kind = args.mask_npz if mask is not None else "GEOMETRIC-DIAGNOSTIC"
    print(f"[z2] tier={args.tier} IC levels {Lc}..{Lf}, runtime L{runtime_levelmax} "
          f"(canonical source N={Npar})", flush=True)
    print(f"[z2] box={args.box_hmpc} Mpc/h mask={mask_kind} "
          f"kcut={kcut:.2f} h/Mpc f1={f1:.4f} H(a)={H_a:.1f}", flush=True)

    run_spec = {
        "schema": "ouruniv-zoom-ic-run-v1",
        "tier": args.tier,
        "global_levelmin": Lc,
        "finest_ic_level": Lf,
        "runtime_levelmax": runtime_levelmax,
        "box_mpc_h": args.box_hmpc,
        "h": args.h,
        "Omega_m": Om,
        "Omega_l": OL,
        "astart": a,
        "small_scale_seed": args.seed,
        "fft_workers": args.fft_workers,
        "parent": os.path.abspath(args.parent),
        "parent_level": args.parent_level,
        "parent_grid_size": Npar,
        "parent_fingerprint": _parent_fingerprint(args.parent),
        "transfer": os.path.abspath(args.transfer),
        "transfer_sha256": sha256_file(args.transfer),
        "mask": (None if args.mask_npz is None else os.path.abspath(args.mask_npz)),
        "mask_sha256": (None if args.mask_npz is None else sha256_file(args.mask_npz)),
        "geometric_mask_diagnostic": mask is None,
        "resolution_config": os.path.abspath(args.resolution_config),
        "resolution_config_sha256": sha256_file(args.resolution_config),
    }
    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "zoom_ic_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as handle:
            old_manifest = json.load(handle)
        if old_manifest.get("run_spec") != run_spec:
            raise RuntimeError(
                f"{manifest_path} belongs to a different seed/mask/input configuration")
        if not args.resume:
            raise RuntimeError(
                f"{manifest_path} already exists; pass --resume only for this exact run")
    elif any(os.scandir(args.out)):
        raise RuntimeError(
            f"{args.out} is non-empty but has no compatible zoom_ic_manifest.json")
    _write_manifest(manifest_path, {"status": "running", "run_spec": run_spec})

    # parent fields (read once): velocities always; delta only for coarse deltab
    print("[z2] reading parent ic_velc* ...", flush=True)
    vpar = [G.read_grafic_field(os.path.join(args.parent, f"ic_velc{ax}"))[0] for ax in "xyz"]
    if any(field.shape != (Npar, Npar, Npar) for field in vpar):
        raise RuntimeError(
            f"parent velocity shape {[field.shape for field in vpar]} "
            f"does not match source N={Npar}")

    # ---- nested-consistent small-scale noise: ONE finest field feeds every fine level ----
    Wf = Nf_fine = Lf_fine = None
    fine_levels = [L for L in range(Lc, Lf + 1) if 2 ** L > Npar]
    if fine_levels:
        Lf_fine = fine_levels[-1]
        i0f, i1f = _level_window(
            Lf_fine, Lc, args.box_hmpc, args.center_hmpc,
            args.subbox_half_hmpc, mask)
        shape_fine = i1f - i0f
        if not np.all(shape_fine == shape_fine[0]):
            raise ValueError(f"fine Fourier subbox is not cubic: {shape_fine}")
        Nf_fine = int(shape_fine[0])
        print(f"[z2] fine levels {fine_levels}: one nested white-noise field N={Nf_fine}^3 "
              f"seed={args.seed} (~{Nf_fine ** 3 * 4 / 1e9:.1f} GB f32) ...", flush=True)
        Wf = hier_rfftn(Nf_fine, args.seed, workers=args.fft_workers)
        print(f"[z2]   rfftn done: <|Wf|^2>/N^3 = {float((np.abs(Wf) ** 2).mean()) / Nf_fine ** 3:.3f} (want ~1)",
              flush=True)

    for L in range(Lc, Lf + 1):
        N = 2 ** L
        dxg = Lbox_mpc / N                                     # GRAFIC comoving Mpc
        dx_h = args.box_hmpc / N
        i0, i1 = _level_window(
            L, Lc, args.box_hmpc, args.center_hmpc,
            args.subbox_half_hmpc, mask)
        shape = i1 - i0
        if not np.all(shape == shape[0]):
            raise ValueError(f"IC subbox is not cubic at L{L}: {shape}")
        off_h = i0.astype(np.float64) * dx_h
        off_mpc = off_h / args.h
        Nsub = int(shape[0])

        # resume: skip a level whose 5 fields are already on disk at full size
        # (generation is deterministic in the seed, so an existing level is correct)
        outdir = os.path.join(args.out, f"level_{L:03d}")
        want = Nsub ** 3 * 4
        fields = ("ic_deltab", "ic_velcx", "ic_velcy", "ic_velcz", "ic_refmap")
        complete = all(
            os.path.exists(os.path.join(outdir, f))
            and os.path.getsize(os.path.join(outdir, f)) >= want
            for f in fields)
        if complete and args.resume:
            print(f"[z2] level {L:2d} N={N:5d} manifest-matched and complete -> skip",
                  flush=True)
            continue
        if os.path.isdir(outdir) and any(os.scandir(outdir)) and not args.resume:
            raise RuntimeError(f"{outdir} already contains files; use a fresh --out")

        if N <= Npar:
            # coarse/equal: exact Fourier projection of the physical parent
            sl = _slice3(i0, i1)
            vx = fourier_resample_field(vpar[0], N, workers=args.fft_workers)[sl]
            vy = fourier_resample_field(vpar[1], N, workers=args.fft_workers)[sl]
            vz = fourier_resample_field(vpar[2], N, workers=args.fft_workers)[sl]
            mode = "fourier-project"
        else:
            # fine: two-scale (long interp parent vel + short fresh synth)
            L_sub_h = Nsub * dx_h
            vlx = interp_parent(vpar[0], args.box_hmpc, off_h, Nsub, dx_h)
            vly = interp_parent(vpar[1], args.box_hmpc, off_h, Nsub, dx_h)
            vlz = interp_parent(vpar[2], args.box_hmpc, off_h, Nsub, dx_h)
            assert Nsub == Nf_fine >> (Lf_fine - L), (L, Nsub, Nf_fine, Lf_fine)   # clean nesting
            Wc = truncate_rfftn(Wf, Nf_fine, Nsub)
            dshort = synth_delta_fft(
                Wc, L_sub_h, N, Npar, tr, kcut=kcut,
                workers=args.fft_workers,
            )
            sx, sy, sz = G.lpt2_velocity(dshort, L_sub_h / args.h, a, H_a, f1)   # short vel (km/s)
            vx, vy, vz = vlx + sx, vly + sy, vlz + sz
            mode = f"two-scale long_rms={vlx.std():.1f} short_rms={sx.std():.1f} km/s"

        if mask is not None:
            refmap = refmap_for_level(mask, L, i0, i1)
        else:
            refmap = build_refmap(
                int(i0[0]), int(i1[0]), dx_h,
                args.center_hmpc, args.mask_R_hmpc)
        os.makedirs(outdir, exist_ok=True)
        wargs = (dxg, tuple(off_mpc), a, Om, OL, h0)
        G.write_grafic_field(os.path.join(outdir, "ic_deltab"), np.zeros_like(vx), *wargs)
        G.write_grafic_field(os.path.join(outdir, "ic_velcx"), vx.astype(np.float32), *wargs)
        G.write_grafic_field(os.path.join(outdir, "ic_velcy"), vy.astype(np.float32), *wargs)
        G.write_grafic_field(os.path.join(outdir, "ic_velcz"), vz.astype(np.float32), *wargs)
        G.write_grafic_field(os.path.join(outdir, "ic_refmap"), refmap, *wargs)
        bounds = ",".join(f"{int(i0[q])}:{int(i1[q])}" for q in range(3))
        offsets = ",".join(f"{off_mpc[q]:.2f}" for q in range(3))
        print(f"[z2] level {L:2d} N={N:5d} sub=[{bounds}]({Nsub}^3) off=({offsets})Mpc "
              f"dx={dxg:.4f}Mpc v_rms=({vx.std():.1f},{vy.std():.1f},{vz.std():.1f}) [{mode}] -> {outdir}", flush=True)

    print("\n[z2] &INIT_PARAMS filetype='grafic'  (initfile 1-based rel. to levelmin)")
    for i, L in enumerate(range(Lc, Lf + 1), start=1):
        print(f"[z2]   initfile({i})='{os.path.join(args.out, f'level_{L:03d}')}'")
    print(f"[z2] &AMR_PARAMS levelmin={Lc} levelmax={runtime_levelmax} ; "
          f"&REFINE_PARAMS ivar_refine=0")
    output_summary = {}
    for L in range(Lc, Lf + 1):
        outdir = os.path.join(args.out, f"level_{L:03d}")
        output_summary[f"level_{L:03d}"] = {
            name: os.path.getsize(os.path.join(outdir, name))
            for name in ("ic_deltab", "ic_velcx", "ic_velcy", "ic_velcz", "ic_refmap")
        }
    _write_manifest(manifest_path, {
        "status": "complete",
        "run_spec": run_spec,
        "output_file_sizes": output_summary,
    })


if __name__ == "__main__":
    main()
