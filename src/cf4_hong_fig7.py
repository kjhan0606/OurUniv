#!/usr/bin/env python
"""Observer-centred all-sky comparison of our present density with Hong et al. (2021) Fig 7.

Hong et al. 2021 (arXiv:2008.01738) applied a TNG300-trained CNN to the Cosmicflows-3
peculiar velocities and produced all-sky Mollweide maps of the local dark-matter column
density in radial shells out to 20 h^-1Mpc (their Fig 7, panels x13-x17). We forward our
CF4-constrained initial field with pmwd to z=0 and render the SAME quantity from the same
viewpoint (observer at the box centre): the column density integrated over each radial
shell, in supergalactic sky coordinates, so the two can be laid side by side.

Input: the +-35 h^-1Mpc present-density cube saved by cf4_ic_test.py (--dump-ptcl file).
Reference: refs/hong2021/x13..x17.png.
"""
import os
import numpy as np


SHELLS = [(0.7, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 16.0), (16.0, 20.0)]  # h^-1Mpc (Hong Fig 7)


def column_maps(cube, sp, shells, nlon=360, nlat=180, dr=0.2):
    """All-sky column density (integral of density over r in each shell), supergalactic.

    Observer = cube centre. Direction (SGL,SGB): rhat = (cos b cos l, cos b sin l, sin b),
    sampled onto the cube by trilinear interpolation. Returns list of (nlat,nlon) maps.
    """
    from scipy.ndimage import map_coordinates
    N = cube.shape[0]; c = (N - 1) / 2.0
    lon = np.linspace(-np.pi, np.pi, nlon)          # SGL
    lat = np.linspace(-np.pi / 2, np.pi / 2, nlat)  # SGB
    LON, LAT = np.meshgrid(lon, lat)
    rx = np.cos(LAT) * np.cos(LON); ry = np.cos(LAT) * np.sin(LON); rz = np.sin(LAT)
    maps = []
    for (r0, r1) in shells:
        rr = np.arange(r0, r1, dr)
        acc = np.zeros_like(LON)
        for r in rr:
            xi = c + r * rx / sp; yi = c + r * ry / sp; zi = c + r * rz / sp
            acc += map_coordinates(cube, [xi.ravel(), yi.ravel(), zi.ravel()],
                                   order=1, mode="nearest").reshape(LON.shape)
        maps.append(acc * dr)                        # column density (density*length)
    return maps, lon, lat


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptcl", default="recon/cf4_ptcl768.npz",
                    help="npz with dens_cube + spacing (from cf4_ic_test --dump-ptcl)")
    ap.add_argument("--hong", default="refs/hong2021", help="dir with x13..x17.png")
    ap.add_argument("--out", default="recon/cf4_hong_fig7.png")
    args = ap.parse_args()

    z = np.load(args.ptcl)
    cube = z["dens_cube"].astype(np.float64)
    sp = float(z["spacing"])
    print(f"[hong] density cube {cube.shape} @ {sp} h^-1Mpc, observer at centre", flush=True)
    maps, lon, lat = column_maps(cube, sp, SHELLS)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    hong_imgs = []
    for k in range(13, 18):
        p = os.path.join(args.hong, f"x{k}.png")
        hong_imgs.append(plt.imread(p) if os.path.exists(p) else None)

    fig = plt.figure(figsize=(12, 15))
    LON, LAT = np.meshgrid(lon, lat)
    for i, ((r0, r1), m) in enumerate(zip(SHELLS, maps)):
        axl = fig.add_subplot(5, 2, 2 * i + 1)
        if hong_imgs[i] is not None:
            axl.imshow(hong_imgs[i]); axl.set_title(f"Hong+21  {r0:.1f}<r<{r1:.0f}", fontsize=9)
        axl.axis("off")
        ms = gaussian_filter(np.log10(m / m.mean() + 0.3), 1.2)
        axr = fig.add_subplot(5, 2, 2 * i + 2, projection="mollweide")
        axr.pcolormesh(LON, LAT, ms, cmap="Greys", shading="auto")
        axr.set_title(f"ours (CF4->pmwd)  {r0:.1f}<r<{r1:.0f}", fontsize=9)
        axr.grid(True, lw=0.3, alpha=0.4); axr.set_xticklabels([]); axr.set_yticklabels([])
    fig.suptitle("Local column density, observer-centred: Hong+2021 (CF3+CNN) vs "
                 "ours (CF4 IC -> pmwd), supergalactic", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(args.out, dpi=110)
    print(f"[hong] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
