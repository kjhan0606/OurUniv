import numpy as np
import jax
import jax.numpy as jnp
from cf4_q1_radial_shell_fourier_jax import predict_phase_basis_kernel_jax
from cf4_q1_cell_integrated_convolution import cell_integrated_tsc_deposit

jax.config.update('jax_enable_x64', True)
n=16; box=6.; h=box/n; pcount=8; cell=np.array([3,5,7]); direction=np.array([[1.,0.,0.]]); disp=.746*np.hypot(24.,11.)/74.6
exposure=np.full((6,n,n,n),.8); masses=np.full((6,1),.7)
basis=np.empty((pcount,pcount,pcount,n,n,n),dtype=np.complex128)
for ix in range(pcount):
  for iy in range(pcount):
    for iz in range(pcount):
      pos=(cell+np.array([ix,iy,iz])/pcount)*h
      kernel=cell_integrated_tsc_deposit(pos[None,:],np.array([1.]),direction,np.array([disp]),n,box)
      basis[ix,iy,iz]=np.fft.fftn(np.roll(kernel,-cell,axis=(0,1,2)))
rows=[]
for phase in (.1,.25,.5,.75):
  pos=(cell+np.array([phase,.37,.63]))*h
  oracle=cell_integrated_tsc_deposit(pos[None,:],np.array([.7]),direction,np.array([disp]),n,box)[None]*exposure
  pred=np.asarray(predict_phase_basis_kernel_jax(jnp.asarray(pos[None,:]),jnp.asarray(masses),jnp.asarray(exposure),jnp.asarray(basis),box_size_cMpc_h=box))
  rows.append({'phase':phase,'relative_l1':float(np.sum(np.abs(pred-oracle))/np.sum(np.abs(oracle)))})
print({'status':'PHASE_BASIS_CHECK','rows':rows},flush=True)
