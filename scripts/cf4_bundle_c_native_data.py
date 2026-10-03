"""Keep actual native observations for fine inference; no pseudo-density/truth."""
import json
from pathlib import Path
import numpy as np
from cf4_actual_selection import base

ROOT=Path(__file__).resolve().parents[1]


def main():
    cfg=json.loads((ROOT/"config/cf4_bundle_c_v1.json").read_text())
    out=Path(cfg["output_root"])/cfg["native_data_subdir"]
    out.mkdir(parents=True,exist_ok=False)
    frozen=Path(cfg["frozen_inputs"])
    tracer=json.loads((ROOT/cfg["tracer_program"]).read_text())
    cat=base.load_catalog(tracer["inputs"]["twompp_catalog"]["path"])
    with np.load(frozen/"corrected_rows.npz",allow_pickle=False) as f:
        rows={k:f[k] for k in f.files}
    order=np.argsort(cat["recno"])
    ids=order[np.searchsorted(cat["recno"][order],rows["recno"])]
    np.testing.assert_array_equal(cat["recno"][ids],rows["recno"])
    distance,_=base.distance_and_absolute_magnitude(cat["Vcmb"],cat["Ksmag"],tracer["cosmology"])
    direction=base.supergalactic_unit_vectors(cat["RA"],cat["DEC"])[ids]
    position=distance[ids,None]*direction+cfg["box_cMpc_h"]/2
    cells=np.floor(position/cfg["background_dx_cMpc_h"]).astype(np.int32)
    if np.any(cells<0) or np.any(cells>=cfg["background_N"]):
        raise ValueError("observations outside finite box")
    used=rows["survives"] & ~rows["calibration"]
    # Count partition stays at zero-origin. It is NOT the density-cell partition.
    coarse=np.floor(position/(cfg["box_cMpc_h"]/cfg["parent_N"])).astype(int)
    np.testing.assert_array_equal(cells//8,coarse)
    flat=np.ravel_multi_index(coarse.T,(32,)*3)
    key=rows["population"].astype(np.int64)*32**3+flat
    with np.load(frozen/"corrected_counts.npz",allow_pickle=False) as f:
        for name,mask in (("counts_all",used),("counts_train",used&rows["train"]),
                          ("counts_holdout",used&rows["holdout"])):
            recovered=np.bincount(key[mask],minlength=6*32**3).reshape(6,32,32,32)
            np.testing.assert_array_equal(recovered,f[name])
    flatfine=np.ravel_multi_index(cells.T,(256,)*3)
    finekey=rows["population"].astype(np.int64)*256**3+flatfine
    sparse={}
    for name,mask in (("all",used),("train",used&rows["train"]),("holdout",used&rows["holdout"])):
        kk,nn=np.unique(finekey[mask],return_counts=True)
        sparse[f"{name}_keys"]=kk
        sparse[f"{name}_counts"]=nn.astype(np.int32)
    np.savez_compressed(out/"galaxy_native.npz",**rows,position_cMpc_h=position,
        direction=direction,radius_cMpc_h=distance[ids],count_cell=cells,
        RA_deg=cat["RA"][ids],DEC_deg=cat["DEC"][ids])
    np.savez_compressed(out/"counts_1p5_sparse.npz",**sparse)
    with np.load(frozen/"observations.npz",allow_pickle=False) as f:
        cf4={k:f[k] for k in f.files if k.startswith("CF4_") or k=="radial_observed"}
    required=("CF4_pos","CF4_holdout","CF4_variance","CF4_rhat","CF4_cz","CF4_pgc")
    if any(k not in cf4 for k in required):
        raise ValueError("incomplete frozen radial data")
    np.savez_compressed(out/"CF4_native.npz",**cf4)
    radii=np.linalg.norm(cf4["CF4_pos"]-192,axis=1)
    shell_radius=cfg["LG_direct_data_radius_cMpc_h"]
    lo,hi=cfg["LG_faces_cMpc_h"]
    in_patch=np.all((position>=lo)&(position<hi),axis=1)
    cf4_patch=np.all((cf4["CF4_pos"]>=lo)&(cf4["CF4_pos"]<hi),axis=1)
    report=dict(status="NATIVE_OBSERVATIONS_READY_NOT_FINE_DENSITY_POSTERIOR",
        count_dx_cMpc_h=1.5,LG_dx_cMpc_h=.1875,
        galaxy_training=int((used&rows["train"]).sum()),galaxy_holdout=int((used&rows["holdout"]).sum()),
        exact_coarse_integer_recovery=True,nonempty_fine_count_cells=int(len(sparse["all_keys"])),
        CF4_rows=len(radii),CF4_min_radius_cMpc_h=float(radii.min()),CF4_min_cz_km_s=float(cf4["CF4_cz"].min()),
        LG_radius_cMpc_h=shell_radius,
        LG_direct_CF4_rows=int((radii<shell_radius).sum()),
        LG_direct_count_rows=int((used&(distance[ids]<shell_radius)).sum()),
        LG_patch_count_rows=int((used&in_patch).sum()),LG_patch_CF4_rows=int(cf4_patch.sum()),
        density_lower_face_cMpc_h=-3.,count_lower_face_cMpc_h=0.,
        sources="Same corrected rows, masks and native frozen CF4 observations as A; no redrawn holdout or recalibration.",
        limits="Galaxy counts are not matter density. No fine selection integrals, posterior, high-k recovery or resolved LG halos supplied. No direct local rows does not remove shared calibration correlations.")
    (out/"result.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__": main()
