import json
import numpy as np
import jax.numpy as jnp
from cf4_q1_radial_shell_fourier_jax import predict_shell_kernel_convolution_jax
from cf4_q1_cell_integrated_convolution import cell_integrated_tsc_deposit, predict_selected_intensity_cell_integrated
from cf4_2mpp_joint_likelihood_local import observer_centred_spherical_rsd

def check_case(positions, vel, n=16):
    box=6.; observer=np.array([3.,3.,3.]); masses=np.full((6,len(positions)),.7); exposure=np.full((6,n,n,n),.8)
    kw=dict(observer=observer,box_size_cMpc_h=box,hubble_km_s_Mpc=74.6,little_h=.746,scale_factor=1.,sigma_fog_km_s=np.full(6,24.),sigma_redshift_km_s=np.full(6,11.))
    oracle=predict_selected_intensity_cell_integrated(positions,vel,masses,exposure,**kw)
    rsd=observer_centred_spherical_rsd(positions,vel,observer,box,74.6,little_h=.746,scale_factor=1.)
    rel=(rsd.positions-observer+box/2)%box-box/2; rh=rel/np.linalg.norm(rel,axis=1)[:,None]; disp=.746*np.hypot(24.,11.)/74.6
    kernels=[]
    for source,direction in enumerate(rh):
        center_idx=np.floor(rsd.positions[source]/(box/n)).astype(int)
        center=(center_idx+.5)*box/n
        kernel=cell_integrated_tsc_deposit(center[None,:],np.array([1.]),direction[None,:],np.array([disp]),n,box)
        kernels.append(np.fft.fftn(np.roll(kernel,-center_idx,axis=(0,1,2))))
    candidate=np.asarray(predict_shell_kernel_convolution_jax(jnp.asarray(rsd.positions),jnp.asarray(masses),jnp.asarray(exposure),jnp.arange(len(positions)),jnp.asarray(kernels),box_size_cMpc_h=box))
    return float(np.sum(np.abs(candidate-oracle))/np.sum(np.abs(oracle)))

box=6.; h=box/16.; base=np.array([[3.5*h,5.5*h,7.5*h]])
rows=[{'phase':phase,'relative_l1':check_case(base+phase*h*np.array([[1.,0.,0.]]),np.zeros((1,3)))} for phase in (0.,.1,.25,.5,.75)]
post_rsd=check_case(np.array([[3.5*h,5.5*h,7.5*h],[9.5*h,2.5*h,12.5*h]]),np.array([[85.,-20.,15.],[-60.,30.,-10.]]))
print(json.dumps({'status':'SHELL_PHASE_CHECK','post_rsd_relative_l1':post_rsd,'rows':rows},indent=2),flush=True)
