import numpy as np
import jax
import jax.numpy as jnp
from cf4_q1_radial_shell_fourier_jax import predict_shell_kernel_convolution_jax
from cf4_q1_cell_integrated_convolution import cell_integrated_tsc_deposit, predict_selected_intensity_cell_integrated

jax.config.update('jax_enable_x64', True)
n=16; box=6.; positions=np.array([[(3+0.5)*box/n]*3])
vel=np.zeros((1,3)); masses=np.full((6,1),.7); exposure=np.full((6,n,n,n),.8)
observer=np.array([3.,3.,3.]); sigma=np.full(6,np.hypot(24.,11.)); kw=dict(observer=observer,box_size_cMpc_h=box,hubble_km_s_Mpc=74.6,little_h=.746,scale_factor=1.,sigma_fog_km_s=np.full(6,24.),sigma_redshift_km_s=np.full(6,11.))
oracle=predict_selected_intensity_cell_integrated(positions,vel,masses,exposure,**kw)
from cf4_2mpp_joint_likelihood_local import observer_centred_spherical_rsd
rsd=observer_centred_spherical_rsd(positions,vel,observer,box,74.6,little_h=.746,scale_factor=1.)
rel=(rsd.positions-observer+box/2)%box-box/2; rh=rel/np.linalg.norm(rel,axis=1)[:,None]; disp=.746*np.hypot(24.,11.)/74.6
kernels=[]
for direction in rh:
    kernel=cell_integrated_tsc_deposit(positions[0:1],np.array([1.]),np.array([direction]),np.array([disp]),n,box)
    # FFT convolution uses index 0 as the impulse origin; the oracle kernel is
    # centred at n//2, so shift it back (the previous prototype used +n//2).
    kernels.append(np.fft.fftn(np.roll(kernel, -n//2, axis=(0,1,2))))
candidate=np.asarray(predict_shell_kernel_convolution_jax(jnp.asarray(positions),jnp.asarray(masses),jnp.asarray(exposure),jnp.array([0]),jnp.asarray(kernels),box_size_cMpc_h=box))
relerr=float(np.sum(np.abs(candidate-oracle))/np.sum(np.abs(oracle)))
print({'status':'ALIAS_KERNEL_CHECK','relative_l1':relerr,'mass':float(candidate.sum())},flush=True)
