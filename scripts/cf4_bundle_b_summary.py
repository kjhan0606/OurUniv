"""One closure figure and compact results, reusing saved B products only."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path("/gpfs/kjhan/CF4/z0_density/bundle_b_v1")
bridge=root/"bridge_retry2"
report=json.loads((bridge/"result.json").read_text())
if len(report["cases"])!=2:
    raise ValueError("both fixed development cases required")
fig,axes=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
rows=[]
for ax,case in zip(axes,report["cases"]):
    with np.load(bridge/f"target_{case['target']}"/"candidate.npz",allow_pickle=False) as f:
        for panel,key,label in zip(ax[:2],["target_density","final_density"],["Target z=0 delta","Forward z=0 delta"]):
            im=panel.imshow((f[key]-1)[:,:,16].T,origin="lower",cmap="RdBu_r",vmin=-1,vmax=2,
                            extent=[-3,381,-3,381])
            panel.set(title=f"Mock {case['target']}: {label}",xlabel="x [cMpc/h]",ylabel="y [cMpc/h]")
        tv=f["target_velocity"].ravel()[::128]
        fv=f["final_velocity"].ravel()[::128]
        ax[2].scatter(tv,fv,s=3,alpha=.4)
        limits=[min(tv.min(),fv.min()),max(tv.max(),fv.max())]
        ax[2].plot(limits,limits,"k--",lw=.7)
        ax[2].set(title="Velocity components: fixed stride",xlabel="Target [km/s]",ylabel="Forward [km/s]")
    start,end=np.array(case["initial_linear_power"]),np.array(case["final_linear_power"])
    ax[3].loglog(start[:,0],start[:,1],label="Fixed prior start")
    ax[3].loglog(end[:,0],end[:,1],label="MAP candidate")
    ax[3].set(title="IC linear P(k); no rescaling",xlabel="k [h/cMpc]",ylabel="P [(cMpc/h)^3]")
    ax[3].legend(fontsize=8)
    b,a=case["before"],case["after"]
    rows.append(dict(target=case["target"],status=case["status"],
        density_RMS_before=b["density_RMS"],density_RMS_after=a["density_RMS"],
        velocity_RMS_before=b["velocity_RMS_km_s"],velocity_RMS_after=a["velocity_RMS_km_s"],
        density_residual_ratio=a["density_RMS"]/b["density_RMS"],
        velocity_residual_ratio=a["velocity_RMS_km_s"]/b["velocity_RMS_km_s"],
        density_correlation=a["density_correlation"],velocity_correlation=a["velocity_correlation"],
        white_second_moment=2*case["final_prior_penalty"]/64**3,
        linear_power_ratios=(end[:,1]/start[:,1]).tolist(),conservation=case["conservation_pass"]))
fig.colorbar(im,ax=axes[:,:2].ravel().tolist(),label="Delta; native 12 cMpc/h slice")
fig.suptitle("Development mocks only: z=0 -> regularized IC -> fresh forward z=0; not actual LG recovery")
fig.savefig(root/"bridge.png",dpi=150)
plt.close(fig)
result=dict(status="B_DELIVERABLES_AVAILABLE_NOT_SCIENTIFIC_PROMOTION",bridge=rows,
    environment="Model-conditional: clusters/Bootes aperture signs; Local Void incomplete and peak positions coarse.",
    LG="Source-backed interface, four tests passed; resolved operator and joint covariance calibration remain.",
    decision="No actual LG/IC posterior or global P(k) certification. Bundle C requires approval.")
(root/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
print(json.dumps(result,allow_nan=False),flush=True)
