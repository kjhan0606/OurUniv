import numpy as np
import jax.numpy as jnp
from cf4_q1_radial_shell_fourier_jax import predict_phase_basis_kernel_jax
from cf4_q1_cell_integrated_convolution import cell_integrated_tsc_deposit, predict_selected_intensity_cell_integrated
from cf4_2mpp_joint_likelihood_local import observer_centred_spherical_rsd

n=16; p=4; box=6.; h=box/n; obs=np.array([3.,3.,3.]); pos=np.array([[1.3125,2.0625,4.3125],[3.5625,4.0625,1.1875]]); vel=np.array([[70.,-15.,10.],[-55.,25.,-12.]])
masses=np.full((6,2),.7); exposure=np.full((6,n,n,n),.8); kw=dict(observer=obs,box_size_cMpc_h=box,hubble_km_s_Mpc=74.6,little_h=.746,scale_factor=1.,sigma_fog_km_s=np.full(6,24.),sigma_redshift_km_s=np.full(6,11.))
oracle=predict_selected_intensity_cell_integrated(pos,vel,masses,exposure,**kw); rsd=observer_centred_spherical_rsd(pos,vel,obs,box,74.6,little_h=.746,scale_factor=1.); rel=(rsd.positions-obs+box/2)%box-box/2; rh=rel/np.linalg.norm(rel,axis=1)[:,None]; disp=.746*np.hypot(24.,11.)/74.6
basis=np.empty((2,p,p,p,n,n,n),complex)
for s in range(2):
  cell=np.floor(rsd.positions[s]/h).astype(int)
  for ix in range(p):
   for iy in range(p):
    for iz in range(p):
     center=(cell+np.array([ix,iy,iz])/p)*h
     k=cell_integrated_tsc_deposit(center[None,:],np.array([1.]),rh[s:s+1],np.array([disp]),n,box)
     basis[s,ix,iy,iz]=np.fft.fftn(np.roll(k,-cell,axis=(0,1,2)))
pred=np.asarray(predict_phase_basis_kernel_jax(jnp.asarray(rsd.positions),jnp.asarray(masses),jnp.asarray(exposure),jnp.array([0,1]),jnp.asarray(basis),box_size_cMpc_h=box))
print({'status':'PHASE_SHELL_RSD_CHECK','relative_l1':float(np.sum(np.abs(pred-oracle))/np.sum(np.abs(oracle))),'mass':float(pred.sum())},flush=True)
