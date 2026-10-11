"""Integrate actual angular/LF selection on N256; never interpolate old exposure."""
from itertools import product
import json
from pathlib import Path
import time
import h5py
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord,CartesianRepresentation
from astropy import units as u
from cf4_actual_selection import base
from cf4_twompp_joint_information_budget_pilot_v1 import _cosmology_distance_table,schechter_fraction

ROOT=Path(__file__).resolve().parents[1]


def main():
    start=time.monotonic()
    cfg=json.loads((ROOT/"config/cf4_bundle_c_v1.json").read_text())
    tracer=json.loads((ROOT/cfg["tracer_program"]).read_text())
    native=Path(cfg["output_root"])/cfg["native_data_subdir"]
    if json.loads((native/"result.json").read_text())["status"]!="NATIVE_OBSERVATIONS_READY_NOT_FINE_DENSITY_POSTERIOR":
        raise ValueError("native inputs required")
    out=Path(cfg["output_root"])/"selection_1p5_v1"
    out.mkdir(parents=True,exist_ok=False)
    inp=tracer["inputs"]
    maps=[base.load_completeness_map(inp[k]["path"],512) for k in ("completeness_11_5","completeness_12_5")]
    rotation=SkyCoord(CartesianRepresentation(np.eye(3)),frame="supergalactic").icrs.cartesian.xyz.value
    check=np.random.default_rng(2026090711).normal(size=(3,128))
    check/=np.linalg.norm(check,axis=0)
    sky=SkyCoord(sgl=np.arctan2(check[1],check[0])*u.rad,sgb=np.arcsin(check[2])*u.rad,frame="supergalactic").icrs
    np.testing.assert_allclose(rotation@check,sky.cartesian.xyz.value,atol=1e-12)
    edges=np.array([5.,30.,60.,90.,120.,150.,180.])
    rtab=np.linspace(5.,180.,200001)
    cosmology=tracer["cosmology"]
    dl=_cosmology_distance_table(rtab,cosmology)*cosmology["h"]
    absolute=tracer["tracer_design"]["absolute_K_edges"]
    def fraction(distance,p):
        a,b=divmod(p,3)
        return schechter_fraction(distance,None if a==0 else 11.5,11.5 if a==0 else 12.5,
                                 absolute[b],absolute[b+1],-23.28,-.94)
    radial_tables=np.array([fraction(dl,p) for p in range(6)])
    rr=np.random.default_rng(2026090712).uniform(5.,180.,1024)
    exact_dl=_cosmology_distance_table(rr,cosmology)*cosmology["h"]
    lf_error=max(float(np.max(abs(np.interp(rr,rtab,radial_tables[p])-fraction(exact_dl,p)))) for p in range(6))
    if lf_error>1e-6:
        raise ValueError("LF tabulation exceeds fixed absolute tolerance")
    n,dx=cfg["background_N"],cfg["background_dx_cMpc_h"]
    axis=(np.arange(n)+.5)*dx-192
    nodes,weights=np.polynomial.legendre.leggauss(4)
    totals=np.zeros((6,6))
    with h5py.File(out/"selection.h5","x") as h:
        dataset=h.create_dataset("selection_shells",shape=(6,6,n,n,n),dtype="f4",
            chunks=(1,1,8,64,64),compression="gzip",compression_opts=1)
        h.attrs.update(count_origin_fraction=.5,dx_cMpc_h=dx,quadrature_order=4,
                       status="INCOMPLETE",LF_h=cosmology["h"])
        for lower in range(0,n,8):
            xyz=np.array(np.meshgrid(axis[lower:lower+8],axis,axis,indexing="ij")).reshape(3,-1)
            slab=np.zeros((6,6,xyz.shape[1]))
            for a,b,c in product(range(4),repeat=3):
                position=xyz+dx/2*nodes[[a,b,c],None]
                radius=np.linalg.norm(position,axis=0)
                active=(radius>=5)&(radius<=180)
                idx=np.flatnonzero(active)
                if idx.size==0: continue
                r=radius[idx]
                pixels=hp.vec2pix(512,*(rotation@position[:,idx]),nest=False)
                shell=np.clip(np.searchsorted(edges,r,side="right")-1,0,5)
                w=weights[a]*weights[b]*weights[c]/8
                for p in range(6):
                    slab[p,shell,idx]+=w*maps[p//3][pixels]*np.interp(r,rtab,radial_tables[p])
            totals+=slab.sum(axis=2)*dx**3
            dataset[:,:,lower:lower+8]=slab.reshape(6,6,8,n,n).astype(np.float32)
            h.flush()
            print(f"selection x={lower+8}/{n}, seconds={time.monotonic()-start:.1f}",flush=True)
        # Preserve evidence of any unresolved support; never invent exposure at a galaxy.
        with np.load(native/"counts_1p5_sparse.npz",allow_pickle=False) as f:
            keys=f["all_keys"]
        pop=keys//n**3
        cell=np.array(np.unravel_index(keys%n**3,(n,)*3)).T
        zero=[]
        for lower in range(0,n,8):
            take=np.flatnonzero((cell[:,0]>=lower)&(cell[:,0]<lower+8))
            block=dataset[:,:,lower:lower+8].sum(axis=1)
            exposure=block[pop[take],cell[take,0]-lower,cell[take,1],cell[take,2]]
            zero.extend(keys[take[exposure<=0]].tolist())
        h.attrs["status"]="INTEGRATION_COMPLETE_NOT_SELECTION_CALIBRATION"
    report=dict(status="READY_FOR_FINE_LIKELIHOOD_DEVELOPMENT" if not zero else "UNRESOLVED_POSITIVE_COUNT_SUPPORT",
        N=n,dx_cMpc_h=dx,quadrature_order=4,LF_table_max_absolute_error=lf_error,
        raw_effective_volumes_cMpc_h3=totals.tolist(),zero_exposure_positive_count_keys=zero,
        elapsed_seconds=time.monotonic()-start,
        limits="New selection integration, NOT a fine density posterior. Angular quadrature and scale-dependent survival/bias remain approximate; source masks are unchanged. No interpolated N32 exposure or data-dependent support patch.")
    (out/"result.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:report[k] for k in ("status","elapsed_seconds","LF_table_max_absolute_error")}),flush=True)


if __name__=="__main__":main()
