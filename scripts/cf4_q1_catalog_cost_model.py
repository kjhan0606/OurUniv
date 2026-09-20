import json
import math
rows=55877
sec_per_source_256=11.528248753398657/16.0
for cores in (1,4,8,16,32):
    serial=rows*sec_per_source_256
    ideal=serial/cores
    print(json.dumps({'sources':rows,'grid':256,'seconds_per_source':sec_per_source_256,'cores':cores,'ideal_seconds':ideal,'ideal_hours':ideal/3600.,'chunk_size':8,'peak_chunk_mib':973.0},sort_keys=True),flush=True)
