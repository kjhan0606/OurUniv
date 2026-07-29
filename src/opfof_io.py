#!/usr/bin/env python
"""Bridge between the pmwd forward and Juhan Kim's MPI FoF (OPFoF).

OPFoF scales the friends-of-friends halo finder to the full 768^3 (4.5e8) particle
box that the scipy pair search cannot handle. It reads z-ordered domain-slab files
and runs one MPI rank per contiguous z-slab.

I/O contract (reverse-engineered from opfof.c/read.c/fof.h, pflag='O'):
  input  SyncINITIAL.<nstep:05d><file:05d>
           header : float32 zstart, float32 zwidth, int32 mp
           mp x    : float32 x,y,z,vx,vy,vz ; int64 indx   (positions in GRID units [0,nx))
         params.<nstep:05d>  ('O' text format, 5 lines)
  output FoF_halo_cat.<nstep:05d>
           header : 48 bytes (7 float, 2 int, 3 float)
           Nh x   : int64 np ; float32 x,y,z,vx,vy,vz     (x..z in h^-1Mpc = grid*pscale)
         FoF_member_particle.<nstep:05d>
           sum(np) x : float32 x,y,z,vx,vy,vz ; int64 indx  (concat per halo, GRID units)

Constraints (user): nfile must be a multiple of n_mpi; files are z-slabs (file index
increases with z); rank r reads files [r*nfile/nid, (r+1)*nfile/nid). OPFoF keeps a
halo on rank r only if its z-extent lies in [zmin_r+link, zmax_r-link]; boundary halos
are passed to the neighbour rank, so the union over ranks is the full periodic FoF.

OPFoF's catalog velocity uses a GOTPM vscale that is wrong for pmwd, so we ignore it and
recompute each halo velocity from its member particles (their vx,vy,vz are stored raw).
"""
import os
import shutil
import struct
import subprocess
import numpy as np

# resolve mpirun (env override -> PATH -> known Intel oneAPI path on this cluster)
MPIRUN = (os.environ.get("MPIRUN") or shutil.which("mpirun")
          or "/opt/ohpc/pub/intel/oneapi/mpi/2021.17/bin/mpirun")

# particle record in the slab input and member-particle output (packed, 32 bytes)
REC = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                ('vx', '<f4'), ('vy', '<f4'), ('vz', '<f4'), ('indx', '<i8')])
# HaloQ record in the halo catalog (size_t np + 6 float, 32 bytes)
HALO = np.dtype([('np', '<i8'), ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                 ('vx', '<f4'), ('vy', '<f4'), ('vz', '<f4')])
_here = os.path.dirname(os.path.abspath(__file__))
# find the binary: env override, then common layouts (project src/opfof/opfof, repo ../opfof)
OPFOF_BIN = os.environ.get("OPFOF_BIN") or next(
    (p for p in (os.path.join(_here, "opfof", "opfof"), os.path.join(_here, "opfof"),
                 os.path.join(_here, "..", "opfof"))
     if os.path.exists(p)), os.path.join(_here, "opfof", "opfof"))


def write_input(pos, vel, nx, L, nstep, nfile, outdir, indx=None):
    """Write z-ordered slab files. pos (Np,3) in [0,L) h^-1Mpc, vel (Np,3) km/s."""
    Np = pos.shape[0]
    pscale = L / nx
    g = np.mod(pos / pscale, nx).astype(np.float32)               # grid units [0,nx)
    if indx is None:
        indx = np.arange(Np, dtype=np.int64)
    order = np.argsort(g[:, 2], kind="stable")                    # z-order for slabs
    g, v, ix = g[order], vel[order], indx[order]
    edges = np.linspace(0, Np, nfile + 1).astype(np.int64)
    os.makedirs(outdir, exist_ok=True)
    for f in range(nfile):
        a, b = edges[f], edges[f + 1]
        rec = np.empty(b - a, REC)
        rec['x'], rec['y'], rec['z'] = g[a:b, 0], g[a:b, 1], g[a:b, 2]
        rec['vx'], rec['vy'], rec['vz'] = v[a:b, 0], v[a:b, 1], v[a:b, 2]
        rec['indx'] = ix[a:b]
        fn = os.path.join(outdir, f"SyncINITIAL.{nstep:05d}{f:05d}")
        with open(fn, "wb") as fp:
            zst = float(g[a:b, 2].min()) if b > a else 0.0
            zw = float(g[a:b, 2].max() - zst) if b > a else 0.0
            fp.write(struct.pack("<ffi", zst, zw, int(b - a)))
            rec.tofile(fp)
    return nfile


def write_params(nstep, outdir, size, nx, nspace=1, hubble=0.746, npower=0.96,
                 omep=0.31, omepb=0.048, omepl=0.69, bias=1.0, smooth=0.0):
    """'O'-format params.<nstep> (size is the box in h^-1Mpc so pscale=size/nx)."""
    fn = os.path.join(outdir, f"params.{nstep:05d}")
    with open(fn, "w") as f:
        f.write(f"{size} {hubble}\n")
        f.write(f"{npower} {omep} {omepb} {omepl} {bias} {smooth}\n")
        f.write(f"{nx} {nx} {nx} {nspace}\n")
        f.write("4 4 0.3\n")            # ntree ntree1 theta (tree params, unused by FoF)
        f.write("49 0.25 1.0\n")        # zinit astep anow  (anow=1 -> z=0)
        f.write("1.0\n")                # amax (optional 6th value; a=anow -> vscale well-defined)
    return fn


def run(nstep, nfile, nid, outdir, bin=OPFOF_BIN, mpirun=None, timeout=7200, opts=None):
    """mpirun -np <nid> opfof <nstep> <nfile> O [key=value ...]  (cwd=outdir).

    opts: dict of extra OPFoF options, e.g. {"b":0.2,"nmin":20,"perrank":1,"ring":0}.
    """
    assert nfile % nid == 0, "nfile must be a multiple of n_mpi"
    cmd = [mpirun or MPIRUN, "-np", str(nid), bin, str(nstep), str(nfile), "O"]
    if opts:
        cmd += [f"{k}={v}" for k, v in opts.items()]
    env = dict(os.environ)                              # single node: fork launch + shared-mem fabric
    env.setdefault("I_MPI_HYDRA_BOOTSTRAP", "fork")
    env.setdefault("I_MPI_FABRICS", "shm")
    r = subprocess.run(cmd, cwd=outdir, capture_output=True, text=True, timeout=timeout, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"opfof failed ({r.returncode})\nSTDOUT:\n{r.stdout[-2000:]}\n"
                           f"STDERR:\n{r.stderr[-2000:]}")
    return r.stdout


def _read_one(hf, mf):
    with open(hf, "rb") as fp:
        fp.read(48)                                    # skip the 48-byte catalog header
        halos = np.fromfile(fp, HALO)
    members = np.fromfile(mf, REC)
    return halos, members


def read_halos(nstep, outdir, L, nx, nmin=20, prefix="FoF", per_rank=False):
    """Parse the catalog + member file(s) into a halo dict; velocity from member means (km/s).

    head/member index into pos_all (the original particle array) via the stored int64
    index, so the catalog plugs straight into hod.populate(mode='particles'). With
    per_rank=True, concatenate the per-rank output files <prefix>_halo_cat.<nstep>.<rank>.
    """
    if per_rank:
        hs, ms, r = [], [], 0
        while True:
            hf = os.path.join(outdir, f"{prefix}_halo_cat.{nstep:05d}.{r:04d}")
            mf = os.path.join(outdir, f"{prefix}_member_particle.{nstep:05d}.{r:04d}")
            if not os.path.exists(hf):
                break
            h, m = _read_one(hf, mf)
            hs.append(h); ms.append(m); r += 1
        if not hs:
            raise FileNotFoundError(f"no per-rank {prefix}_halo_cat.{nstep:05d}.* in {outdir}")
        halos = np.concatenate(hs); members = np.concatenate(ms)
    else:
        halos, members = _read_one(
            os.path.join(outdir, f"{prefix}_halo_cat.{nstep:05d}"),
            os.path.join(outdir, f"{prefix}_member_particle.{nstep:05d}"))
    counts = halos['np'].astype(np.int64)
    off = np.concatenate([[0], np.cumsum(counts)])
    assert off[-1] == members.shape[0], f"member count mismatch {off[-1]} vs {members.shape[0]}"
    pos = np.mod(np.stack([halos['x'], halos['y'], halos['z']], 1).astype(np.float64), L)
    mvel = np.stack([members['vx'], members['vy'], members['vz']], 1).astype(np.float64)
    memb_idx = members['indx'].astype(np.int64)

    kept = np.flatnonzero(counts >= nmin)
    n_kept = counts[kept]
    vel = np.stack([mvel[off[h]:off[h + 1]].mean(0) for h in kept]) if len(kept) \
        else np.zeros((0, 3))                          # km/s, bypasses OPFoF's GOTPM vscale
    head = np.concatenate([[0], np.cumsum(n_kept)])[:-1].astype(np.int64)
    member = (np.concatenate([memb_idx[off[h]:off[h + 1]] for h in kept])
              if len(kept) else np.zeros(0, np.int64))
    return dict(n=n_kept, pos=pos[kept], vel=vel, head=head, member=member,
                L=float(L), sigv=np.zeros(len(kept)), r_rms=np.zeros(len(kept)))


def fof_opfof(pos, vel, L, nx, nstep, nfile, nid, outdir, m_particle=None,
              nmin=20, opts=None, verbose=False):
    """One-call FoF via OPFoF: write -> run -> parse. Returns a catalog dict like fof.fof.

    opts: extra OPFoF options (e.g. {"perrank":1,"ring":0,"b":0.2}). With perrank=1 the
    per-rank output files are concatenated on read.
    """
    write_input(pos, vel, nx, L, nstep, nfile, outdir)
    write_params(nstep, outdir, size=L, nx=nx)
    if verbose:
        print(f"[opfof] {pos.shape[0]} ptcl -> {nfile} slabs, {nid} ranks opts={opts}", flush=True)
    out = run(nstep, nfile, nid, outdir, opts=opts)
    if verbose:
        print(out[-800:], flush=True)
    per_rank = bool(opts and int(opts.get("perrank", 0)))
    prefix = (opts or {}).get("outprefix", "FoF")
    cat = read_halos(nstep, outdir, L, nx, nmin=nmin, prefix=prefix, per_rank=per_rank)
    cat['mass'] = cat['n'].astype(np.float64) * (m_particle if m_particle else 1.0)
    if verbose:
        print(f"[opfof] {len(cat['n'])} halos (>= {nmin} mem), max {cat['n'].max() if len(cat['n']) else 0}",
              flush=True)
    return cat


if __name__ == "__main__":
    print(__doc__)
