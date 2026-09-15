"""Additive v6-open staged authorization contract; shipped fully fail-closed."""
from __future__ import annotations
import hashlib, json, os, stat
from pathlib import Path
from typing import Any
import cf4_aggregate_evidence_smc_execution as base_execution
import cf4_aggregate_evidence_smc_shared_annealing_v6 as shared

ROOT=Path(__file__).resolve().parents[1]
DESIGN=ROOT/'config/cf4_aggregate_evidence_smc_execution_authorization_design_v6_open.json'
PROGRAM=ROOT/'config/cf4_aggregate_evidence_smc_execution_authorization_program_v6_open.json'
GRANT=ROOT/'config/cf4_aggregate_evidence_smc_execution_grant_v6_open.json'
RELEASE=Path('/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_execution_authorization_v6_open_release.json')
MANIFEST=Path('/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_execution_authorization_v6_open_manifest.json')
RECEIPTS=Path('/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_receipts')
PILOT=Path('/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_disposable_pilot')
DATA=Path('/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open')
STATE=Path('/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_run')
DESIGN_SHA='162ee1122ad0d756420021a6855872dde0643d54a0d21131fd08a82da4cdca1e'
AUTH={"implementation_authorized":True,"grant_creation_authorized":False,"external_release_or_manifest_creation_authorized":False,"receipt_creation_authorized":False,"pilot_stage_authorized":False,"production_stage_authorized":False,"cache_population_authorized":False,"downstream_execution_authorized":False,"automatic_follow_on_authorized":False}
COMPLETE={'complete_pass_production_smc','complete_scientific_fail_production_smc'}

def sha256_file(p:Path)->str:
 d=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): d.update(b)
 return d.hexdigest()
def _json(p:Path,label:str)->dict[str,Any]:
 if not p.is_file(): raise PermissionError(f'{label} is absent')
 try: x=json.loads(p.read_text())
 except Exception as e: raise PermissionError(f'{label} is invalid') from e
 if not isinstance(x,dict): raise PermissionError(f'{label} is invalid')
 return x
def _v6open(p:Path,expected:Path,label:str)->None:
 s=str(p).lower()
 if 'v4' in s or 'v5' in s or 'v6_open' not in s or Path(p).resolve()!=expected.resolve(): raise PermissionError(f'{label} is not canonical v6-open')
def load_program()->dict[str,Any]:
 if sha256_file(DESIGN)!=DESIGN_SHA: raise PermissionError('v6-open design hash mismatch')
 x=_json(PROGRAM,'v6-open program')
 if x.get('schema')!='ouruniv-cf4-aggregate-evidence-smc-execution-authorization-program-v6-open' or x.get('authorization')!=AUTH or x.get('design')!={'path':str(DESIGN.relative_to(ROOT)),'sha256':DESIGN_SHA}: raise PermissionError('v6-open program changed')
 if x.get('shared_v6',{}).get('source_sha256')!=sha256_file(ROOT/'src/cf4_aggregate_evidence_smc_shared_annealing_v6.py') or x.get('base',{}).get('program_sha256')!=sha256_file(ROOT/'config/cf4_aggregate_evidence_smc_production_program.json'): raise PermissionError('v6-open hard pin changed')
 for raw,expected,label in ((x['paths']['grant'],GRANT,'grant'),(x['paths']['release'],RELEASE,'release'),(x['paths']['manifest'],MANIFEST,'manifest'),(x['paths']['receipt_root'],RECEIPTS,'receipt')):
  actual=Path(raw); _v6open(actual if actual.is_absolute() else ROOT/actual,expected,label)
 shared.validate_frozen_v6_parameters(); return x
def require_execution_authorization(program:dict[str,Any],stage:str)->dict[str,Any]:
 if stage not in {'pilot','production'}: raise PermissionError('unknown v6-open stage')
 load_program()
 if not GRANT.is_file(): raise PermissionError('v6-open sealed grant is absent; all stages unauthorized')
 if not RELEASE.is_file() or not MANIFEST.is_file(): raise PermissionError('v6-open paired external objects are absent')
 raise PermissionError('v6-open future grant is unreachable in this implementation')
def _snapshot(stage:str)->dict[str,Any]:
 st=RELEASE.stat()
 if st.st_mode&(stat.S_IWUSR|stat.S_IWGRP|stat.S_IWOTH): raise PermissionError('release must be read-only')
 return {'schema':'ouruniv-cf4-v6-open-stage-receipt-v1','stage':stage,'release':{'path':str(RELEASE),'sha256':sha256_file(RELEASE),'stat':{'dev':st.st_dev,'ino':st.st_ino,'size':st.st_size,'nlink':st.st_nlink}},'manifest_sha256':sha256_file(MANIFEST),'grant_sha256':sha256_file(GRANT),'program_sha256':sha256_file(PROGRAM)}
def create_stage_receipt(stage:str,receipt:Path,program:dict[str,Any])->dict[str,Any]:
 require_execution_authorization(program,stage); receipt=Path(receipt)
 if receipt.parent.resolve()!=RECEIPTS.resolve() or receipt.name!=stage: raise PermissionError('stage receipt path is not canonical')
 receipt.mkdir(mode=0o700); os.link(RELEASE,receipt/'release.anchor'); value=_snapshot(stage); (receipt/'snapshot.json').write_text(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n'); return value
def revalidate_stage_receipt(stage:str,receipt:Path,snapshot:dict[str,Any])->None:
 anchor=Path(receipt)/'release.anchor'
 if not anchor.is_file() or anchor.stat().st_ino!=RELEASE.stat().st_ino or _snapshot(stage)!=snapshot: raise PermissionError('stage receipt provenance changed')
def read_only_postcheck(data:Path)->dict[str,Any]:
 data=Path(data).resolve(); _v6open(data,DATA,'data')
 manifest=data/'manifest.json'
 if not manifest.is_file() or manifest.stat().st_mode&(stat.S_IWUSR|stat.S_IWGRP|stat.S_IWOTH): raise PermissionError('manifest is absent or not read-only')
 value=base_execution.validate_published_bundle(data)
 if value.get('status') not in COMPLETE or value.get('valid_scientific_complete') is not True: raise RuntimeError('postcheck is not allowed COMPLETE')
 return value
def run_authorized_v6_open(program_path:Path,stage:str)->None:
 if Path(program_path).resolve()!=PROGRAM.resolve(): raise PermissionError('v6-open accepts only canonical program')
 require_execution_authorization(load_program(),stage)
 raise AssertionError('unreachable: future staged runner not installed')
