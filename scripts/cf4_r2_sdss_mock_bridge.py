"""Bounded first official SDSS mock acquisition and host/member readout.

Never extracts archive paths or downloads the full 10.6GB archive. The first
regular catalogue is a fixed source fixture, not selected for fit quality.
"""
import hashlib
import argparse
import io
import json
import os
from pathlib import Path
import tarfile
from urllib.request import urlopen

import numpy as np
from scipy.integrate import cumulative_trapezoid

BASE = Path('/gpfs/kjhan/CF4')
OUT = BASE/'z0_density/r2_sdss_mock_bridge_v1'
URL = 'https://zenodo.org/api/records/6824749/files/mocks.tar.gz/content'


class BoundedReader:
    def __init__(self, stream, limit=64*1024**2):
        self.stream, self.limit, self.count = stream, limit, 0

    def read(self, n=-1):
        n = min(n if n >= 0 else 65536, self.limit-self.count+1)
        data = self.stream.read(n)
        self.count += len(data)
        if self.count > self.limit:
            raise RuntimeError('compressed source exceeded64MiB bound')
        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-source',action='store_true')
    args = parser.parse_args()
    out = BASE/'z0_density/r2_sdss_mock_bridge_v3' if args.resume_source else OUT
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    if out.exists():
        raise FileExistsError(out)
    if args.resume_source:
        binding = json.loads((OUT/'source.json').read_text())
        raw = (OUT/'first_mock.dat').read_bytes()
        if hashlib.sha256(raw).hexdigest() != binding['member_sha256']:
            raise ValueError('saved member changed')
    else:
        with urlopen(URL,timeout=90) as response:
            stream = BoundedReader(response)
            with tarfile.open(fileobj=stream,mode='r|gz') as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    if not member.name.endswith('_err_corr') or member.size > 128*1024**2:
                        raise ValueError(f'unexpected first source member: {member.name}, {member.size}')
                    raw = archive.extractfile(member).read()
                    if len(raw) != member.size:
                        raise ValueError('truncated mock member')
                    member_name = member.name
                    break
                else:
                    raise ValueError('no regular mock member')
        # Preserve this fixed input even if its schema needs an explicit adapter.
        out.mkdir(parents=True)
        (out/'first_mock.dat').write_bytes(raw)
        binding = dict(url=URL,archive_member=member_name,compressed_bytes_read=stream.count,
                       member_bytes=len(raw),member_sha256=hashlib.sha256(raw).hexdigest(),
                       full_archive_checksum_verified=False)
        (out/'source.json').write_text(json.dumps(binding,indent=2)+'\n')
    first = raw.splitlines()[0].decode()
    print(json.dumps(dict(source=binding,header=first)),flush=True)
    data = np.genfromtxt(io.BytesIO(raw),names=True,delimiter=',',encoding=None)
    needed = ('z_true','z_obs','z_obs_cen','cen_flag','logdist_true','logdist',
              'logdist_err','logdist_alpha','parenthalomass')
    if any(k not in data.dtype.names for k in needed):
        raise ValueError(f'unsupported schema: {data.dtype.names}')
    z = np.linspace(0,.2,40001)
    d = 2997.92458*cumulative_trapezoid(1/np.sqrt(.31*(1+z)**3+.69),z,initial=0)
    for key in ('z_true','z_obs','z_obs_cen'):
        if np.any(~np.isfinite(data[key])|(data[key]<=0)|(data[key]>.2)):
            raise ValueError(f'unsupported redshift range: {key}')
    dist = lambda key: np.interp(data[key],z,d)
    shift = np.log10(dist('z_obs_cen')/dist('z_obs'))
    # Both observed and truth eta move to the SAME group-redshift numerator.
    # This is a reference-variable conversion, NOT a new FP fit or correction
    # for Tempel selection/richness. Published mock FP fit is uncorrected.
    truth_group = data['logdist_true']+shift
    measured_group = data['logdist']+shift
    redshift_v = 299792.458*(data['z_obs']-data['z_obs_cen'])/(1+data['z_obs_cen'])
    observer_id = int(binding['archive_member'].split('.')[-1].split('_')[0])
    observers = np.array([[250,260,300],[250,260,1200],[250,1160,300],[250,1160,1200],
                          [1150,260,300],[1150,260,1200],[1150,1160,300],[1150,1160,1200]])
    offset = np.column_stack([data[k] for k in ('x','y','z')])-observers[observer_id]
    direction = offset/np.linalg.norm(offset,axis=1)[:,None]
    relative_v = np.column_stack([data[k]-data[k+'cen'] for k in ('vx','vy','vz')])
    internal_v = np.sum(relative_v*direction,axis=1)
    central = data['cen_flag']==1
    if np.any(~np.isin(data['cen_flag'],[0,1])):
        raise ValueError('invalid central flag')
    summaries = {}
    for name,mask in (('central',central),('satellite',~central)):
        v = internal_v[mask]
        summaries[name] = dict(n=int(mask.sum()),internal_velocity_rms_km_s=float(np.sqrt(np.mean(v*v))),
            redshift_offset_equivalent_rms_km_s=float(np.sqrt(np.mean(redshift_v[mask]**2))),
            redshift_vs_velocity_rms_difference_km_s=float(np.sqrt(np.mean((redshift_v[mask]-v)**2))),
            internal_velocity_abs_quantiles_km_s=np.quantile(np.abs(v),[.5,.9,.99]).tolist(),
            eta_reference_shift_quantiles=np.quantile(shift[mask],[.01,.5,.99]).tolist())
    residual = data['logdist_true']-np.log10(dist('z_obs')/dist('z_true'))
    result = dict(classification='ONE_SELECTED_SDSS_MOCK_HOST_MEMBER_BRIDGE_NOT_CALIBRATION',
        job_id=os.environ['SLURM_JOB_ID'],source=binding,rows=len(data),columns=list(data.dtype.names),
        populations=summaries,eta_truth_distance_max_abs_residual=float(np.max(np.abs(residual))),
        eta_residual_preserved_max_abs=float(np.max(np.abs((measured_group-truth_group)-(data['logdist']-data['logdist_true'])))),
        full_parent_membership_available=False,Tempel_regrouping_available=False,
        group_selection_calibrated=False,group_COM_vs_PM_calibrated=False,R2_posterior=False,
        new_gravity_runs=0)
    if args.resume_source:
        out.mkdir(parents=True)
        (out/'source.json').write_text(json.dumps(binding,indent=2)+'\n')
    np.savez_compressed(out/'host_member_inputs.npz',z_true=data['z_true'],z_observed=data['z_obs'],
        z_host_observed=data['z_obs_cen'],central=central,parent_logmass=data['parenthalomass'],
        member_host_velocity_km_s=internal_v,redshift_offset_equivalent_km_s=redshift_v,
        eta_group_truth=truth_group,eta_group_measured=measured_group,
        eta_std=data['logdist_err'],eta_alpha=data['logdist_alpha'])
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
