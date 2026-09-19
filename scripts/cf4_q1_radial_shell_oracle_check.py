import json
import numpy as np
import jax
import jax.numpy as jnp
from cf4_q1_radial_shell_fourier_jax import predict_selected_intensity_radial_shell_fourier_jax
from cf4_q1_cell_integrated_convolution import predict_selected_intensity_cell_integrated
from cf4_2mpp_joint_likelihood_local import observer_centred_spherical_rsd

jax.config.update('jax_enable_x64', True)
box=6.; n=16; observer=np.array([3.,3.,3.])
positions=np.array([[1.3,2.1,4.2],[5.7,.4,2.7]],dtype=np.float64)
vel=np.array([[20.,-5.,8.],[-12.,4.,9.]],dtype=np.float64); masses=np.full((6,2),.7)
exposure=np.full((6,n,n,n),.8); exposure[:,2,3,5]=.2; exposure[:,8,6,1]=1.3
sigma_fog=np.full(6,24.); sigma_red=np.full(6,11.)
kw=dict(observer=observer,box_size_cMpc_h=box,hubble_km_s_Mpc=74.6,little_h=.746,scale_factor=1.,sigma_fog_km_s=sigma_fog,sigma_redshift_km_s=sigma_red)
oracle=predict_selected_intensity_cell_integrated(positions,vel,masses,exposure,**kw)
rsd=observer_centred_spherical_rsd(positions,vel,observer,box,74.6,little_h=.746,scale_factor=1.)
relative=(rsd.positions-observer+box/2)%box-box/2; rhat=relative/np.linalg.norm(relative,axis=1)[:,None]
disp=.746*np.hypot(24.,11.)/74.6; shell_ids=jnp.array([0,1]);
candidate=np.asarray(predict_selected_intensity_radial_shell_fourier_jax(jnp.asarray(positions),jnp.asarray(vel),jnp.asarray(masses),jnp.asarray(exposure),shell_ids,jnp.asarray(rhat),jnp.array([disp,disp]),**kw))
rel=float(np.sum(np.abs(candidate-oracle))/np.sum(np.abs(oracle)))
print(json.dumps({'status':'RADIAL_SHELL_ORACLE_CHECK','relative_l1':rel,'mass_oracle':float(oracle.sum()),'mass_candidate':float(candidate.sum()),'finite':bool(np.isfinite(candidate).all())},indent=2),flush=True)
