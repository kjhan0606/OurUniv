import json
rows=55877
chunk_size=64
chunks=[{'chunk_id':i,'start':i*chunk_size,'stop':min((i+1)*chunk_size,rows),'count':min((i+1)*chunk_size,rows)-i*chunk_size} for i in range((rows+chunk_size-1)//chunk_size)]
print(json.dumps({'source_rows':rows,'chunk_size':chunk_size,'chunk_count':len(chunks),'chunks':chunks},separators=(',',':')),flush=True)
