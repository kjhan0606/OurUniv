"""Read-only terminal fit summary/illustration; never touch heldout data."""
import json
import os
from pathlib import Path

import numpy as np

BASE = Path('/gpfs/kjhan/CF4/z0_density/r2_v6_partial_map_v1')


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    report = json.loads((BASE/'result.json').read_text())
    output = BASE/'readout'
    output.mkdir(exist_ok=False)
    summary = {k: report.get(k) for k in ('status','error','iterations','evaluations',
        'optimizer_success','optimizer_message','initial_objective','final_objective',
        'final_gradient_inf','initial_adjoint','host_peak_GiB','elapsed_seconds')}
    summary.update(R2_complete=False,posterior_uncertainty=False,heldout_scored=False,
                   map_available=(BASE/'final_state.npz').is_file())
    if summary['map_available']:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(11,9),constrained_layout=True)
        for col,name in enumerate(('initial','final')):
            with np.load(BASE/f'{name}_state.npz',allow_pickle=False) as f:
                rho=f['rho']
                v=f['velocity_km_s'][2]
                slab=slice(62,67)
                density=rho[:,:,slab].mean(axis=2)
                den=rho[:,:,slab].sum(axis=2)
                bulk=np.divide((rho*v)[:,:,slab].sum(axis=2),den,
                               out=np.zeros_like(den),where=den>0)
                summary[name]=dict(mean_density=float(rho.mean()),
                    density_min=float(rho.min()),density_max=float(rho.max()),
                    white_IC_mean_square=float(np.mean(f['white_ic']**2)),
                    white_tracer=f['tracer'].tolist())
            # Log floor is for display only; saved physical fields unchanged.
            im=axes[0,col].imshow(np.log10(np.maximum(density,1e-5)).T,origin='lower',
                extent=(-192,192,-192,192),vmin=-1,vmax=1,cmap='magma')
            fig.colorbar(im,ax=axes[0,col],label='log10(rho / mean rho)')
            im=axes[1,col].imshow(bulk.T,origin='lower',extent=(-192,192,-192,192),
                vmin=-500,vmax=500,cmap='RdBu_r')
            fig.colorbar(im,ax=axes[1,col],label='mass-weighted v_SGZ (km/s)')
            axes[0,col].set_title('Unconditioned start' if col==0 else 'Training-only partial MAP iterate')
            for ax in axes[:,col]:
                ax.set_xlabel('SGX (cMpc/h)'); ax.set_ylabel('SGY (cMpc/h)')
                ax.plot(0,0,'+',color='lime',markersize=6)
        fig.suptitle('N128 / 384: 3 cMpc/h cells, central 15 cMpc/h slab\n'
                     'NOT a calibrated posterior; MW/M31/M33 not identified')
        fig.savefig(output/'field_comparison.png',dpi=150)
        plt.close(fig)
    if report.get('trace'):
        summary['accepted_steps']=len(report['trace'])
        summary['last_accepted']=report['trace'][-1]
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
