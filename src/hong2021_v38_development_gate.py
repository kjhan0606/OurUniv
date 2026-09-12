#!/usr/bin/env python
"""Integrity-bound three-domain gate for V38 Gaussian-copula innovation."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from hong2021_v6_gate import field_gate
from hong2021_v15_development_gate import _load_metrics, canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v20_development_gate import marginal_diagnostics
from hong2021_v28_empirical import DOMAIN_KEYS, DOMAIN_ORDER
from hong2021_v38_gaussian_copula import ARMS, ENSEMBLE_SCHEMA, PREFLIGHT_SCHEMA, PROGRAM_SHA256, load_program


SCHEMA = "hong2021-v38-gaussian-copula-innovation-three-domain-decision-v1"


def _validate(path: Path, arm: str, domain: str, parent: Path, gate_commit: str) -> dict[str, Any]:
    with h5py.File(path,"r") as h, h5py.File(parent,"r") as old:
        exact={"schema":ENSEMBLE_SCHEMA,"method":"train_only_query_conditioned_gaussian_copula_innovation","arm":arm,"v38_program_sha256":PROGRAM_SHA256,"parent_selection_sha256":sha256_file(parent),"diagnostic_k_h_mpc":1.0,"ensemble_members":16,"donor_translation":False,"donor_reselection":False,"validation_truth_used_for_fit_or_sampling":False,"density_field_clipping":False,"posthoc_Ak_used":False,"worktree_clean_at_sampling":True,"Astrid_accessed":False,"historical_EAGLE_accessed":False,"complete":True}
        for key,expected in exact.items():
            actual=h.attrs.get(key); actual=actual.item() if isinstance(actual,np.generic) else actual
            if actual!=expected: raise ValueError(f"V38 {domain} {arm} metadata differs: {key}")
        reused=("source_index","donor_source","donor_index","donor_isometry","donor_distance","predicted_residual_dc","predicted_band_scales")
        if tuple(h["sample"].shape)!=(16,16,1,64,64,64) or any(not np.array_equal(h[n][:],old[n][:]) for n in reused): raise ValueError("V38 selection or shape differs")
        residual=np.asarray(h["sample"],dtype=np.float32)-np.asarray(h["conditional_mean"],dtype=np.float32)[:,None]; maximum=float(np.max(np.abs(residual.mean(axis=(-3,-2,-1)))))
        if maximum>1e-7: raise ValueError("V38 residual DC differs")
        model=Path(str(h.attrs["model"])); report=Path(str(h.attrs["report"])); preflight=Path(str(h.attrs["preflight"])); model_sha=str(h.attrs["model_sha256"]); report_sha=str(h.attrs["report_sha256"]); preflight_sha=str(h.attrs["preflight_sha256"])
        if sha256_file(model)!=model_sha or sha256_file(report)!=report_sha or sha256_file(preflight)!=preflight_sha: raise ValueError("V38 artifact binding differs")
        pf=json.loads(preflight.read_text()); commit=str(h.attrs["sampling_code_commit"])
        if pf.get("schema")!=PREFLIGHT_SCHEMA or pf.get("status")!="pass" or pf.get("code_commit")!=commit or pf.get("model_sha256")!=model_sha: raise ValueError("V38 preflight differs")
        if subprocess.run(["git","merge-base","--is-ancestor",commit,gate_commit],capture_output=True).returncode: raise ValueError("V38 sampling commit is not ancestor")
    return {"sampling_code_commit":commit,"model":str(model.resolve()),"model_sha256":model_sha,"report":str(report.resolve()),"report_sha256":report_sha,"preflight":str(preflight.resolve()),"preflight_sha256":preflight_sha,"maximum_absolute_sample_residual_dc":maximum,"donor_selection_exactly_reused":True}


def q_pass(domains: dict[str,Any]) -> tuple[bool,bool]:
    return (all(abs(r["mechanism_Q3_Q4"]["delta_q99_999_dex"])<=.1 and r["mechanism_Q3_Q4"]["generated_max_above_truth_max_dex"]<=.3 for r in domains.values()),all(r["mechanism_Q3_Q4"]["generated_over_truth_mean_delta_squared"]<=1.5 for r in domains.values()))


def classify(primary: bool,q3: bool,q4: bool,material: bool)->tuple[str,str]:
    if primary:return "linear_query_conditioned_gaussian_copula_sufficient","seal_v38_and_await_explicit_approval_before_independent_gate"
    if q3 and q4:return "linear_gaussian_copula_repairs_tails_but_not_morphology","freeze_a_multiscale_nonlinear_query_conditioned_copula_without_changing_the_marginal_inverse"
    if material:return "linear_query_conditioning_is_causal_but_insufficient","replace_bandwise_linear_predictor_by_train_only_multiscale_nonlinear_conditional_copula"
    return "linear_gaussian_query_conditioning_is_not_a_common_domain_repair","freeze_a_train_only_nonparametric_local_patch_copula_with_bijective_overlap_consistency"


def evaluate(root:Path,program_path:Path,repo:Path,commit:str)->dict[str,Any]:
    program,v35=load_program(program_path,repo); arms={}
    for arm in ARMS:
        domains={}
        for domain in DOMAIN_ORDER:
            domain_root=root/arm/"development_candidate"/DOMAIN_KEYS[domain]; ensemble=domain_root/"ensemble16.h5"; parent=Path(v35["development_domains"][domain]["phase_object_selection"]); provenance=_validate(ensemble,arm,domain,parent,commit); metrics_path=domain_root/"ensemble_evaluation/metrics.json"; metrics=_load_metrics(metrics_path)
            if Path(metrics["path"]).resolve()!=ensemble.resolve():raise ValueError("V38 metrics point elsewhere")
            domains[domain]={"ensemble":str(ensemble.resolve()),"ensemble_sha256":sha256_file(ensemble),"metrics":str(metrics_path.resolve()),"metrics_sha256":sha256_file(metrics_path),"field_gate":field_gate(metrics),"mechanism_Q3_Q4":marginal_diagnostics(ensemble),"provenance":provenance}
        q3,q4=q_pass(domains); arms[arm]={"domains":domains,"Q3_all_domains":q3,"Q4_all_domains":q4,"all_three_field_pass":all(r["field_gate"]["pass"] for r in domains.values())}
    inherited=program["inherited_inputs"]; v31=_verified_record((repo/inherited["v31_record"]).resolve(),inherited["v31_record_sha256"]); comparison={}; causal=[]
    for domain in DOMAIN_ORDER:
        candidate=arms["query_conditioned"]["domains"][domain]["mechanism_Q3_Q4"]; control=arms["zero_predictor_control"]["domains"][domain]["mechanism_Q3_Q4"]; old=v31["paired_v29_to_v31"][domain]; oq=float(old["Q4"][1]); cq=float(candidate["generated_over_truth_mean_delta_squared"]); zq=float(control["generated_over_truth_mean_delta_squared"]); causal.append(cq/oq<=.75 and cq/zq<=.90); comparison[domain]={"Q3_delta_q99_999_dex_V31_candidate_control":[float(old["Q3_delta_q99_999_dex"][1]),candidate["delta_q99_999_dex"],control["delta_q99_999_dex"]],"Q3_maximum_excess_dex_V31_candidate_control":[float(old["Q3_maximum_excess_dex"][1]),candidate["generated_max_above_truth_max_dex"],control["generated_max_above_truth_max_dex"]],"Q4_V31_candidate_control":[oq,cq,zq],"candidate_Q4_over_V31":cq/oq,"candidate_Q4_over_control":cq/zq}
    selected=arms["query_conditioned"]; material=all(causal); primary=selected["Q3_all_domains"] and selected["Q4_all_domains"] and selected["all_three_field_pass"]; classification,next_step=classify(bool(primary),bool(selected["Q3_all_domains"]),bool(selected["Q4_all_domains"]),material)
    payload={"schema":SCHEMA,"experiment":"v38_query_conditioned_gaussian_copula_innovation","program":str(program_path.resolve()),"program_sha256":PROGRAM_SHA256,"gate_code_commit":commit,"worktree_clean_at_gate":True,"arms":arms,"comparison_to_v31_and_zero_predictor":comparison,"conditional_innovation_material":material,"development_pass":bool(primary),"classification":classification,"next":next_step,"validation_used_for_fit_or_hyperparameter_choice":False,"donor_translation":False,"donor_reselection":False,"density_field_clipping":False,"posthoc_Ak_used":False,"Astrid_accessed":False,"historical_EAGLE_accessed":False};payload["decision_digest_sha256"]=canonical_digest(payload);return payload


def _verified_record(path:Path,digest:str)->dict[str,Any]:
    if sha256_file(path)!=digest:raise ValueError("V38 V31 record differs at gate")
    return json.loads(path.read_text())


def main()->None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True);p.add_argument("--program",type=Path,required=True);p.add_argument("--repo",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args();commit,clean=git_state(a.repo.resolve())
    if not clean:raise RuntimeError("V38 gate requires clean worktree")
    result=evaluate(a.root.resolve(),a.program.resolve(),a.repo.resolve(),commit)
    if a.out.exists():raise RuntimeError("V38 refuses existing decision")
    partial=a.out.with_suffix(a.out.suffix+".partial");partial.write_text(json.dumps(result,indent=2)+"\n");os.replace(partial,a.out);print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
