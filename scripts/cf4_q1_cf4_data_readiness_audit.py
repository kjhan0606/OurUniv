import csv, json
from collections import Counter
from pathlib import Path
import numpy as np

path=Path('data/cf4_galaxies.csv'); error_cols=('e_DM','e_DMsnIa','e_DMtf','e_DMfp','e_DMsbf','e_DMsnII','e_DMtrgb','e_DMceph','e_DMmas'); n=0; valid_geom=0; valid_v=0; valid_err=0; methods=Counter(); min_dist=1e99; max_dist=0.
with path.open(newline='') as f:
    reader=csv.DictReader(f); headers=reader.fieldnames or []
    for row in reader:
        n+=1
        try:
            dm,sgl,sgb=float(row['DM']),float(row['SGL']),float(row['SGB'])
            if np.all(np.isfinite([dm,sgl,sgb])):
                valid_geom+=1; d=.746*10**((dm-25)/5); min_dist=min(min_dist,d); max_dist=max(max_dist,d)
        except (TypeError,ValueError): pass
        try:
            if np.isfinite(float(row['Vcmb'])): valid_v+=1
        except (TypeError,ValueError): pass
        found=None
        for col in error_cols:
            try:
                value=float(row[col])
                if np.isfinite(value) and value>=0: found=col; break
            except (TypeError,ValueError,KeyError): continue
        if found is not None: valid_err+=1; methods[found]+=1
selection=[h for h in headers if any(x in h.lower() for x in ('comple','selection','weight','flux','mag'))]
print(json.dumps({'status':'CF4_DATA_READINESS','rows':n,'valid_geometry':valid_geom,'valid_vcmb':valid_v,'valid_distance_error':valid_err,'error_method_counts':dict(methods),'distance_range_cMpc_h':[min_dist,max_dist],'selection_like_columns':selection,'tangential_velocity_columns':[],'selection_map_in_catalog':False},sort_keys=True),flush=True)
