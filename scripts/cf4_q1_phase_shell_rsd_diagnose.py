import numpy as np
import jax.numpy as jnp
from cf4_q1_radial_shell_fourier_jax import predict_phase_basis_kernel_jax
from cf4_q1_cell_integrated_convolution import cell_integrated_tsc_deposit, predict_selected_intensity_cell_integrated
from cf4_2mpp_joint_likelihood_local import observer_centred_spherical_rsd

n=16; p=8; box=6.; h=box/n; obs=np.array([3.,3.,3.])
pos=np.array([[1.3125,2.0625,4.3125],[3.5625,4.0625,1.1875]])
vel=np.array([[70.,-15.,10.],[-55.,25.,-12.]])
masses=np.full((6,2),.7); exposure=np.full((6,n,n,n),.8)
kw=dict(observer=obs,box_size_cMpc_h=box,hubble_km_s_Mpc=74.6,little_h=.746,scale_factor=1.,sigma_fog_km_s=np.full(6,24.),sigma_redshift_km_s=np.full(6,11.))
rsd=observer_centred_spherical_rsd(pos,vel,obs,box,74.6,little_h=.746,scale_factor=1.)
rel=(rsd.positions-obs+box/2)%box-box/2; rh=rel/np.linalg.norm(rel,axis=1)[:,None]
disp=.746*np.hypot(24.,11.)/74.6
print({'shifted_positions':rsd.positions.tolist(),'rhat':rh.tolist()},flush=True)
basis=np.empty((2,p,p,p,n,n,n),complex)
for s in range(2):
  cell=np.floor(rsd.positions[s]/h).astype(int)
  for ix in range(p):
   for iy in range(p):
    for iz in range(p):
     center=(cell+np.array([ix,iy,iz])/p)*h
     k=cell_integrated_tsc_deposit(center[None,:],np.array([1.]),rh[s:s+1],np.array([disp]),n,box)
     basis[s,ix,iy,iz]=np.fft.fftn(np.roll(k,-cell,axis=(0,1,2)))

def run(label, positions, velocities, ids):
  oracle=predict_selected_intensity_cell_integrated(positions,velocities,masses[:,:positions.shape[0]],exposure,**kw)
  shifted=observer_centred_spherical_rsd(positions,velocities,obs,box,74.6,little_h=.746,scale_factor=1.).positions
  pred=np.asarray(predict_phase_basis_kernel_jax(jnp.asarray(shifted),jnp.asarray(masses[:,:positions.shape[0]]),jnp.asarray(exposure),jnp.asarray(basis),jnp.asarray(ids),box_size_cMpc_h=box))
  print({'case':label,'relative_l1':float(np.sum(np.abs(pred-oracle))/np.sum(np.abs(oracle))),'oracle_mass':float(oracle.sum()),'pred_mass':float(pred.sum())},flush=True)

for s in range(2):
  q=rsd.positions[s]; cell=np.floor(q/h).astype(int)%n; ph=(q/h%1)*p; low=np.floor(ph).astype(int)%p; frac=ph-np.floor(ph)
  interp=np.zeros((n,n,n),complex)
  for ix in (0,1):
   for iy in (0,1):
    for iz in (0,1):
     w=(frac[0] if ix else 1-frac[0])*(frac[1] if iy else 1-frac[1])*(frac[2] if iz else 1-frac[2])
     interp += w*basis[s,(low[0]+ix)%p,(low[1]+iy)%p,(low[2]+iz)%p]
  direct=np.real(np.fft.ifftn(np.fft.fftn(np.eye(n)[cell[0]][:,None,None]*0 + np.zeros((n,n,n))).astype(complex))) if False else None
  delta=np.zeros((n,n,n)); delta[tuple(cell)]=1.
  approx=np.real(np.fft.ifftn(np.fft.fftn(delta)*interp))
  exact=cell_integrated_tsc_deposit(q[None,:],np.array([1.]),rh[s:s+1],np.array([disp]),n,box)
  print({'kernel_source':s,'relative_l1':float(np.sum(np.abs(approx-exact))/np.sum(np.abs(exact)))},flush=True)

# Decisive exact-kernel translation test: no phase interpolation.
for s in range(2):
  q=rsd.positions[s]; cell=np.floor(q/h).astype(int)%n
  k=cell_integrated_tsc_deposit(q[None,:],np.array([1.]),rh[s:s+1],np.array([disp]),n,box)
  kernel=np.fft.fftn(np.roll(k,-cell,axis=(0,1,2)))
  delta=np.zeros((n,n,n)); delta[tuple(cell)]=1.
  approx=np.real(np.fft.ifftn(np.fft.fftn(delta)*kernel))
  exact=cell_integrated_tsc_deposit(q[None,:],np.array([1.]),rh[s:s+1],np.array([disp]),n,box)
  print({'exact_kernel_source':s,'relative_l1':float(np.sum(np.abs(approx-exact))/np.sum(np.abs(exact)))},flush=True)

run('two_source_rsd',pos,vel,np.array([0,1]))
run('source0_rsd',pos[:1],vel[:1],np.array([0]))
run('source1_rsd',pos[1:],vel[1:],np.array([1]))
run('two_source_zero_velocity',pos,np.zeros_like(vel),np.array([0,1]))
