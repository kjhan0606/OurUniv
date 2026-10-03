import json
import numpy as np
from cf4_q3_source_compressed_operator import evaluate_chunked_q1_operator_shared_geometry

rng=np.random.default_rng(20261004); m=64; n=64; box=1000.; obs=np.full(3,box/2.); pos=rng.uniform(100.,900.,size=(m,3)); rel=(pos-obs+box/2)%box-box/2; los=rel/np.linalg.norm(rel,axis=1)[:,None]; vr=rng.normal(0.,300.,m); base=(pos+(vr/100.)[:,None]*los)%box
raw=rng.normal(size=(m,3)); tang=raw-(np.sum(raw*los,axis=1)[:,None]*los); tang*=200./np.linalg.norm(tang,axis=1)[:,None]; full_velocity=vr[:,None]*los+tang; radial_recovered=np.sum(full_velocity*los,axis=1); masses=rng.uniform(.5,1.5,size=(6,m)); sigma=np.full(m,5.)
a=evaluate_chunked_q1_operator_shared_geometry(base,los,sigma,masses,n,box,chunk_size=8); b=evaluate_chunked_q1_operator_shared_geometry((pos+(radial_recovered/100.)[:,None]*los)%box,los,sigma,masses,n,box,chunk_size=8)
print(json.dumps({'status':'TANGENTIAL_VELOCITY_GATE','relative_l1':float(np.sum(np.abs(a-b))/np.sum(np.abs(a))),'radial_projection_error':float(np.max(np.abs(radial_recovered-vr))),'radial_only_contract':'PASS' if np.allclose(base,(pos+(radial_recovered/100.)[:,None]*los)%box,atol=1e-12) else 'FAIL','note':'tangential prior is marginalized because the operator consumes only v_r'}),flush=True)
