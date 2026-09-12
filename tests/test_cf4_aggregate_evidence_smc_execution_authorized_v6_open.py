import json
from pathlib import Path
import pytest
import cf4_aggregate_evidence_smc_execution_authorized_v6_open as auth

def test_open_program_is_pinned_v6_only_and_all_stages_closed():
 p=auth.load_program()
 assert p['authorization']==auth.AUTH
 assert all('v4' not in str(v).lower() and 'v5' not in str(v).lower() for v in p['paths'].values())
 assert p['fixed_science']['worker_processes']==8 and p['fixed_science']['threads_per_worker']==1
 assert p['fixed_science']['replicates_sequential'] is True
 assert not auth.GRANT.exists()

def test_public_gate_rejects_wrong_path_and_absent_grant_before_stage_work(tmp_path):
 with pytest.raises(PermissionError,match='canonical'):
  auth.run_authorized_v6_open(tmp_path/'wrong.json','pilot')
 with pytest.raises(PermissionError,match='grant is absent'):
  auth.run_authorized_v6_open(auth.PROGRAM,'pilot')
 with pytest.raises(PermissionError,match='unknown'):
  auth.require_execution_authorization(auth.load_program(),'other')

def test_v6_open_rejects_old_namespace_strings():
 with pytest.raises(PermissionError,match='canonical'):
  auth._v6open(Path('/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v5'),auth.DATA,'data')

def test_open_design_freezes_receipt_schedule_and_completion_contract():
 d=json.loads(auth.DESIGN.read_text())
 assert d['staged_contract']['each_stage_receipt_has_release_hardlink_and_canonical_snapshot'] is True
 assert d['staged_contract']['pilot_posterior_cache_or_result_reuse'] is False
 assert d['completion_contract']['allowed_scientific_complete_statuses']==sorted(auth.COMPLETE)
