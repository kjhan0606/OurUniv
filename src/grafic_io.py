#!/usr/bin/env python
"""GRAFIC1 initial-conditions writer/reader for lagRAMSES.

The CF4 diffusion posterior recovers the whitened initial field s on a full
periodic cubic box (periodic BC + cubic by construction -- the user's constraint).
To start a constrained lagRAMSES run we serialize that IC as GRAFIC1 files:

    ic_deltab            linear overdensity delta(a_start)   [dimensionless]
    ic_velcx/y/z         peculiar velocity components         [km/s, proper]

GRAFIC1 layout (Bertschinger 2001; the format RAMSES/lagRAMSES reads). Each file is
a Fortran sequential-unformatted stream. Record 1 is the header:

    int32   np1, np2, np3            grid dimensions
    float32 dx                       comoving cell size [Mpc]  (NOT Mpc/h)
    float32 x1o, x2o, x3o            grid-corner offset [Mpc]
    float32 astart                   start scale factor
    float32 omega_m, omega_l, h0     cosmology; h0 in km/s/Mpc

Then np3 records, one per k-plane, each np1*np2 float32 with i1 (x) fastest
(Fortran column-major within the plane). Every record is bracketed by an int32
byte-count marker on each side (Fortran unformatted convention).

Units note: RAMSES grafic works in Mpc (not Mpc/h), so dx = (L_Mpc_h / h) / N and
h0 = 100 h. Velocities are proper peculiar km/s. The Zel'dovich helper builds
consistent v from delta; for the production export we instead take delta AND v
straight from the pmwd LPT solve at a_start so they share the exact cosmology.
"""
import os
import numpy as np


# ----------------------------- low-level Fortran records -----------------------------
def _write_record(fh, payload):
    marker = np.int32(len(payload)).tobytes()
    fh.write(marker); fh.write(payload); fh.write(marker)


def _read_record(fh):
    raw = fh.read(4)
    if len(raw) < 4:
        return None
    n = int(np.frombuffer(raw, np.int32)[0])
    payload = fh.read(n)
    fh.read(4)                                   # trailing marker
    return payload


def _header_bytes(N1, N2, N3, dx, offset, astart, omega_m, omega_l, h0):
    ints = np.array([N1, N2, N3], np.int32).tobytes()
    flts = np.array([dx, offset[0], offset[1], offset[2],
                     astart, omega_m, omega_l, h0], np.float32).tobytes()
    return ints + flts


# ----------------------------- field writer / reader -----------------------------
def write_grafic_field(path, field, dx, offset, astart, omega_m, omega_l, h0):
    """Write one GRAFIC1 field file. field: (N1,N2,N3) real, x-axis first."""
    N1, N2, N3 = field.shape
    with open(path, "wb") as fh:
        _write_record(fh, _header_bytes(N1, N2, N3, dx, offset, astart,
                                        omega_m, omega_l, h0))
        for k in range(N3):                      # one record per k-plane, i1 fastest
            plane = np.ascontiguousarray(field[:, :, k], np.float32).ravel(order="F")
            _write_record(fh, plane.tobytes())


def read_grafic_field(path):
    """Read a GRAFIC1 field file -> (field (N1,N2,N3) float32, header dict)."""
    with open(path, "rb") as fh:
        hdr = _read_record(fh)
        N1, N2, N3 = np.frombuffer(hdr[:12], np.int32)
        dx, x1, x2, x3, astart, om, ol, h0 = np.frombuffer(hdr[12:44], np.float32)
        field = np.empty((int(N1), int(N2), int(N3)), np.float32)
        for k in range(int(N3)):
            plane = np.frombuffer(_read_record(fh), np.float32).reshape(
                (int(N1), int(N2)), order="F")
            field[:, :, k] = plane
    meta = dict(N=(int(N1), int(N2), int(N3)), dx=float(dx),
                offset=(float(x1), float(x2), float(x3)), astart=float(astart),
                omega_m=float(om), omega_l=float(ol), h0=float(h0))
    return field, meta


# ----------------------------- 2LPT displacement / velocity from delta -----------------------------
_AX = (0, 1, 2)


def _kgrid(N, L_mpc):
    kf = np.fft.fftfreq(N, d=L_mpc / N) * 2.0 * np.pi        # [1/Mpc]
    kr = np.fft.rfftfreq(N, d=L_mpc / N) * 2.0 * np.pi
    KX, KY, KZ = np.meshgrid(kf, kf, kr, indexing="ij")
    K2 = KX ** 2 + KY ** 2 + KZ ** 2
    K2 = np.where(K2 == 0.0, 1.0, K2)
    return (KX, KY, KZ), K2


def _irfft(a, N):
    return np.fft.irfftn(a, s=(N, N, N), axes=_AX)


def lpt2_fields(delta, L_mpc):
    """First- and second-order Lagrangian displacement fields (comoving Mpc) from a
    linear overdensity delta on a periodic cube. Returns (Psi1, Psi2, delta2) with
    Psi1, Psi2 each a 3-tuple of (N,N,N) real arrays.

    Convention (Scoccimarro 1998; Crocce et al. 2006, as in 2LPTic):
      grad^2 phi1 = delta                          -> phi1_k = -delta_k / k^2
      Psi1 = -grad phi1                            (Zel'dovich)
      delta2 = sum_{i<j} (phi1_ii phi1_jj - phi1_ij^2)
      grad^2 phi2 = delta2                         -> phi2_k = -delta2_k / k^2
      Psi2 = -grad phi2
    Total 2LPT displacement is Psi1 + (3/7) Psi2 (the 3/7 and growth factors are
    applied by lpt2_velocity so this stays purely geometric)."""
    N = delta.shape[0]
    (KX, KY, KZ), K2 = _kgrid(N, L_mpc)
    K = (KX, KY, KZ)
    dk = np.fft.rfftn(delta)
    phi1_k = -dk / K2                                         # grad^2 phi1 = delta
    # second derivatives phi1_{,ij} in real space
    phi1_dd = {}
    for i in range(3):
        for j in range(i, 3):
            phi1_dd[(i, j)] = _irfft(-K[i] * K[j] * phi1_k, N)
    delta2 = (phi1_dd[(0, 0)] * phi1_dd[(1, 1)]
              + phi1_dd[(0, 0)] * phi1_dd[(2, 2)]
              + phi1_dd[(1, 1)] * phi1_dd[(2, 2)]
              - phi1_dd[(0, 1)] ** 2 - phi1_dd[(0, 2)] ** 2 - phi1_dd[(1, 2)] ** 2)
    phi2_k = -np.fft.rfftn(delta2) / K2
    Psi1 = tuple(_irfft(-1j * K[i] * phi1_k, N) for i in range(3))   # -grad phi1
    Psi2 = tuple(_irfft(-1j * K[i] * phi2_k, N) for i in range(3))   # -grad phi2
    return Psi1, Psi2, delta2


def lpt2_velocity(delta, L_mpc, a, H_a, f1, f2=None):
    """Peculiar velocity (km/s, proper) from a linear overdensity delta(a) via 2LPT.

      v = a H(a) [ f1 Psi1 + f2 (3/7) Psi2 ]
    with growth rates f1 = dlnD1/dlna (~Om(a)^0.55) and f2 = dlnD2/dlna (~2 Om(a)^(6/11);
    defaults to 2*f1, exact for D2 ~ D1^2). L_mpc comoving box; H_a in km/s/Mpc."""
    if f2 is None:
        f2 = 2.0 * f1
    Psi1, Psi2, _ = lpt2_fields(delta, L_mpc)
    aH = a * H_a
    v = tuple((aH * (f1 * Psi1[i] + f2 * (3.0 / 7.0) * Psi2[i])).astype(np.float32)
              for i in range(3))
    return v


def lpt2_displacement(delta, L_mpc):
    """2LPT comoving displacement field (Mpc): Psi1 + (3/7) Psi2. Returns (dx,dy,dz)."""
    Psi1, Psi2, _ = lpt2_fields(delta, L_mpc)
    return tuple((Psi1[i] + (3.0 / 7.0) * Psi2[i]).astype(np.float32) for i in range(3))


# ----------------------------- full IC directory -----------------------------
def write_grafic_ic(outdir, delta, velx, vely, velz, L_mpc_h, h,
                    astart, omega_m, omega_l, offset_mpc=(0.0, 0.0, 0.0)):
    """Write ic_deltab + ic_velc{x,y,z} into outdir (GRAFIC1, RAMSES-ready).

    delta, vel*: (N,N,N) grids (delta dimensionless; vel in km/s proper).
    L_mpc_h: box in Mpc/h. h: little-h. Converts to GRAFIC's Mpc units."""
    os.makedirs(outdir, exist_ok=True)
    N = delta.shape[0]
    dx = (L_mpc_h / h) / N                                    # comoving Mpc
    h0 = 100.0 * h
    off = tuple(float(o) for o in offset_mpc)
    args = (dx, off, astart, omega_m, omega_l, h0)
    write_grafic_field(os.path.join(outdir, "ic_deltab"), delta, *args)
    write_grafic_field(os.path.join(outdir, "ic_velcx"), velx, *args)
    write_grafic_field(os.path.join(outdir, "ic_velcy"), vely, *args)
    write_grafic_field(os.path.join(outdir, "ic_velcz"), velz, *args)
    return dict(outdir=outdir, N=N, dx_mpc=dx, box_mpc=L_mpc_h / h, h0=h0,
                astart=astart, omega_m=omega_m, omega_l=omega_l)


# ----------------------------- self-test -----------------------------
def _selftest():
    print("[grafic] self-test")
    N, L = 32, 128.0                                          # Mpc comoving
    rng = np.random.default_rng(0)
    # a smooth-ish random overdensity (low-pass white noise)
    w = rng.standard_normal((N, N, N)).astype(np.float32)
    kf = np.fft.fftfreq(N, d=L / N) * 2 * np.pi
    kr = np.fft.rfftfreq(N, d=L / N) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kf, kf, kr, indexing="ij")
    K = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
    env = np.exp(-(K * 8.0) ** 2)                             # suppress small scales
    delta = np.fft.irfftn(np.fft.rfftn(w) * env, s=(N, N, N), axes=(0, 1, 2)).astype(np.float32)
    delta -= delta.mean()

    # roundtrip write/read
    path = "/tmp/claude-10396/-home-kjhan-BACKUP-CF4/dec93330-1e57-4185-91a8-a65c274b6d90/scratchpad/ic_test"
    write_grafic_field(path, delta, 4.0, (0, 0, 0), 0.02, 0.31, 0.69, 74.6)
    back, meta = read_grafic_field(path)
    err = float(np.abs(back - delta).max())
    print(f"  roundtrip max|err|={err:.2e}  header={meta}")
    assert err < 1e-5, "roundtrip mismatch"

    # 2LPT: to first order div(v) -> -f1 a H delta (continuity); the 2nd-order term is
    # O(delta^2), so it must vanish as amplitude->0. Check both limits.
    f1, a, H_a = 0.5, 0.02, 74.6 / 0.02 ** 1.5               # crude EdS-ish H(a)
    (KX, KY, KZ), _ = _kgrid(N, L)
    def _divv(vx, vy, vz):
        return _irfft(1j * (KX * np.fft.rfftn(vx) + KY * np.fft.rfftn(vy)
                            + KZ * np.fft.rfftn(vz)), N)
    for amp, tol in ((1e-3, 5e-3), (1.0, None)):
        vx, vy, vz = lpt2_velocity(amp * delta, L, a, H_a, f1)
        divv = _divv(vx, vy, vz)
        target = -f1 * a * H_a * (amp * delta)               # first-order continuity
        rel = float(np.std(divv - target) / np.std(target))
        tag = "linear-limit" if amp < 1 else "full-amp (2nd-order visible)"
        print(f"  2LPT div(v)+f1 a H delta rel={rel:.2e}  amp={amp}  [{tag}] "
              f"v_rms={np.std(vx):.1f} km/s")
        if tol is not None:
            assert rel < tol, "2LPT first-order continuity broken in linear limit"
    # 2nd-order source must scale as delta^2
    _, _, d2a = lpt2_fields(1e-3 * delta, L)
    _, _, d2b = lpt2_fields(2e-3 * delta, L)
    ratio = np.std(d2b) / np.std(d2a)
    print(f"  delta2 scaling (expect ~4x for 2x amp): {ratio:.2f}")
    assert abs(ratio - 4.0) < 0.2, "2nd-order source not quadratic"
    vx, vy, vz = lpt2_velocity(delta, L, a, H_a, f1)

    # full IC dir
    outdir = "/tmp/claude-10396/-home-kjhan-BACKUP-CF4/dec93330-1e57-4185-91a8-a65c274b6d90/scratchpad/ic_dir"
    info = write_grafic_ic(outdir, delta, vx, vy, vz, L_mpc_h=96.0, h=0.746,
                           astart=0.02, omega_m=0.31, omega_l=0.69)
    files = sorted(os.listdir(outdir))
    print(f"  wrote {files} dx={info['dx_mpc']:.3f} Mpc box={info['box_mpc']:.1f} Mpc")
    assert files == ["ic_deltab", "ic_velcx", "ic_velcy", "ic_velcz"]
    print("[grafic] self-test PASS")


if __name__ == "__main__":
    _selftest()
