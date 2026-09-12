#!/usr/bin/env python
"""Frozen V38 query-conditioned Gaussian-copula innovation model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
from scipy.special import ndtr, ndtri

from hong2021_augmentation import CUBE_ISOMETRIES, apply_cube_isometry
from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_KEYS, DOMAIN_ORDER
from hong2021_v30_backbone_audit import fourier_masks
from hong2021_v31_copula import conditional_forward, conditional_inverse, load_model as load_v31
from hong2021_v35_spectrum_phase import _backbone, _open_split, load_program as load_v35
from hong2021_v37_query_alignment import _selection_arrays


PROGRAM_SCHEMA = "hong2021-v38-query-conditioned-gaussian-copula-innovation-development-program-v1"
PROGRAM_SHA256 = "d2eeff96a9365ef300866978a36146faa10a579dd5f4c66b5b657da1a69b70f8"
MODEL_SCHEMA = "hong2021-v38-train-only-gaussian-copula-wiener-v1"
PREFLIGHT_SCHEMA = "hong2021-v38-gaussian-copula-innovation-hard-preflight-v1"
ENSEMBLE_SCHEMA = "hong2021-v38-gaussian-copula-innovation-ensemble-v1"
ARMS = ("query_conditioned", "zero_predictor_control")
FEATURES = (
    "log1p_count",
    "asinh_mean_velocity_over_100",
    "log1p_intrinsic_dispersion_over_100",
    "backbone",
    "backbone_squared",
    "log1p_count_times_backbone",
    "radius_over_10_mpc_h",
)
EDGES = np.asarray((0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, np.inf))
EPSILON = 1.0 / (2.0 * 64**3)
RIDGE_FRACTION = 1.0e-3


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"{label} hash differs")
    return json.loads(path.read_text())


def load_program(path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    program = _verified_json(path.resolve(), PROGRAM_SHA256, "V38 program")
    if program.get("schema") != PROGRAM_SCHEMA or not str(program.get("status", "")).startswith("frozen_"):
        raise ValueError("V38 program schema or status differs")
    parent = program["parent_evidence"]
    record = _verified_json((repo / parent["v37_record"]).resolve(), parent["v37_record_sha256"], "V38 V37 record")
    decision = record.get("decision", {})
    if (
        decision.get("classification") != parent["required_classification"]
        or decision.get("next") != parent["required_next"]
        or record.get("firewall", {}).get("Astrid_accessed") is not False
        or record.get("firewall", {}).get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V38 parent conclusion or firewall differs")
    inherited = program["inherited_inputs"]
    v35_path = (repo / inherited["v35_program"]).resolve()
    if sha256_file(v35_path) != inherited["v35_program_sha256"]:
        raise ValueError("V38 V35 program hash differs")
    v35, _ = load_v35(v35_path, repo)
    if sha256_file((repo / inherited["v31_record"]).resolve()) != inherited["v31_record_sha256"]:
        raise ValueError("V38 V31 record hash differs")
    if sha256_file(Path(inherited["conditional_copula_artifact"])) != inherited["conditional_copula_artifact_sha256"]:
        raise ValueError("V38 V31 copula hash differs")
    return program, v35


def radius_field() -> np.ndarray:
    coordinate = (np.arange(64, dtype=np.float64) + 0.5) * 0.3125 - 10.0
    return np.sqrt(
        np.square(coordinate[:, None, None])
        + np.square(coordinate[None, :, None])
        + np.square(coordinate[None, None, :])
    ) / 10.0


RADIUS = radius_field().astype(np.float32)


def feature_cube(data: h5py.File, cache: h5py.File, index: int) -> np.ndarray:
    count = np.asarray(data["input"][index, 0], dtype=np.float32)
    velocity = np.asarray(data["input"][index, 1], dtype=np.float32)
    dispersion = np.asarray(data["input"][index, 2], dtype=np.float32)
    backbone = _backbone(cache, index).astype(np.float32)
    logcount = np.log1p(count)
    result = np.stack(
        (
            logcount,
            np.arcsinh(velocity / 100.0),
            np.log1p(dispersion / 100.0),
            backbone,
            np.square(backbone),
            logcount * backbone,
            RADIUS,
        )
    ).astype(np.float32)
    if result.shape != (len(FEATURES), 64, 64, 64) or not np.isfinite(result).all():
        raise ValueError("V38 feature cube differs")
    return result


def gaussian_score(residual: np.ndarray, backbone: np.ndarray, copula: Mapping[str, Any]) -> np.ndarray:
    uniform = conditional_forward(residual, backbone, copula).astype(np.float64)
    mid = EPSILON + (1.0 - 2.0 * EPSILON) * uniform
    score = ndtri(mid)
    score -= score.mean(axis=(-3, -2, -1), keepdims=True)
    if not np.isfinite(score).all():
        raise ValueError("V38 Gaussian score is nonfinite")
    return score


def source_balanced_normalization(rows: Mapping[str, tuple[np.ndarray, np.ndarray, int]]) -> tuple[np.ndarray, np.ndarray]:
    if tuple(rows) != DOMAIN_ORDER:
        raise ValueError("V38 normalization source order differs")
    means, seconds = [], []
    for domain in DOMAIN_ORDER:
        total, second, count = rows[domain]
        means.append(np.asarray(total, dtype=np.float64) / count)
        seconds.append(np.asarray(second, dtype=np.float64) / count)
    mean = np.mean(means, axis=0)
    std = np.sqrt(np.maximum(np.mean(seconds, axis=0) - np.square(mean), 1.0e-12))
    if mean.shape != (len(FEATURES),) or np.any(std <= 0) or not np.isfinite(std).all():
        raise ValueError("V38 normalization is invalid")
    return mean, std


def feature_transforms(value: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    standardized = (np.asarray(value, dtype=np.float64) - mean[:, None, None, None]) / std[:, None, None, None]
    standardized -= standardized.mean(axis=(1, 2, 3), keepdims=True)
    return np.fft.fftn(standardized, axes=(-3, -2, -1))


def fit_model(program_path: Path, repo: Path, artifact: Path, report_path: Path) -> dict[str, Any]:
    program, v35 = load_program(program_path, repo)
    commit, clean = git_state(repo.resolve())
    if not clean:
        raise RuntimeError("V38 fit requires a clean worktree")
    copula = load_v31(Path(program["inherited_inputs"]["conditional_copula_artifact"]), program["inherited_inputs"]["conditional_copula_artifact_sha256"])
    normalization: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for domain in DOMAIN_ORDER:
        row = v35["development_domains"][domain]
        total = np.zeros(len(FEATURES)); second = np.zeros(len(FEATURES)); count = 0
        data, cache = _open_split(row, "train")
        try:
            for index in range(int(row["train_objects"])):
                value = feature_cube(data, cache, index).astype(np.float64)
                total += value.sum(axis=(1, 2, 3)); second += np.square(value).sum(axis=(1, 2, 3)); count += 64**3
                if (index + 1) % 64 == 0 or index + 1 == int(row["train_objects"]):
                    print(f"[v38-normalize] {domain} {index + 1}/{row['train_objects']}", flush=True)
        finally:
            data.close(); cache.close()
        normalization[domain] = (total, second, count)
    mean, std = source_balanced_normalization(normalization)
    masks = fourier_masks(64, 0.3125, EDGES)
    domain_stats: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for domain in DOMAIN_ORDER:
        row = v35["development_domains"][domain]
        cxx = np.zeros((len(masks), len(FEATURES), len(FEATURES)))
        cxz = np.zeros((len(masks), len(FEATURES)))
        pz = np.zeros(len(masks)); objects = int(row["train_objects"])
        data, cache = _open_split(row, "train")
        try:
            for index in range(objects):
                x = feature_transforms(feature_cube(data, cache, index), mean, std)
                backbone = _backbone(cache, index)[None]
                truth = np.asarray(data["target"][index], dtype=np.float32)
                z = gaussian_score(truth - backbone, backbone, copula)[0]
                zf = np.fft.fftn(z)
                for band, mask in enumerate(masks):
                    xv = x[:, mask]; zv = zf[mask]; modes = max(int(mask.sum()), 1)
                    cxx[band] += np.real(xv @ np.conj(xv.T)) / modes
                    cxz[band] += np.real(np.conj(xv) @ zv) / modes
                    pz[band] += float(np.square(np.abs(zv)).mean())
                if (index + 1) % 32 == 0 or index + 1 == objects:
                    print(f"[v38-covariance] {domain} {index + 1}/{objects}", flush=True)
        finally:
            data.close(); cache.close()
        domain_stats[domain] = (cxx / objects, cxz / objects, pz / objects)
    cxx = np.mean([domain_stats[d][0] for d in DOMAIN_ORDER], axis=0)
    cxz = np.mean([domain_stats[d][1] for d in DOMAIN_ORDER], axis=0)
    pz = np.mean([domain_stats[d][2] for d in DOMAIN_ORDER], axis=0)
    beta = np.zeros_like(cxz); pp = np.zeros(len(masks)); pe = np.zeros(len(masks)); ridge = np.zeros(len(masks))
    for band in range(len(masks)):
        ridge[band] = RIDGE_FRACTION * np.trace(cxx[band]) / len(FEATURES)
        beta[band] = np.linalg.solve(cxx[band] + ridge[band] * np.eye(len(FEATURES)), cxz[band])
        pp[band] = float(beta[band] @ cxx[band] @ beta[band])
        pe[band] = float(pz[band] - 2.0 * beta[band] @ cxz[band] + pp[band])
    if np.any(pz <= 0) or np.any(pe <= 0) or not np.isfinite(beta).all():
        raise RuntimeError("V38 fitted covariance is invalid")
    candidate_scale = np.sqrt(np.maximum(pz - pp, 0.0) / pe)
    control_scale = np.sqrt(pz / pe)
    metadata = {"schema": MODEL_SCHEMA, "program_sha256": PROGRAM_SHA256, "code_commit": commit, "features": list(FEATURES), "epsilon": EPSILON}
    if artifact.exists() or report_path.exists():
        raise RuntimeError("V38 refuses to overwrite fit outputs")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    partial = artifact.with_suffix(artifact.suffix + ".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, feature_mean=mean, feature_std=std, edges=EDGES, beta=beta, candidate_scale=candidate_scale, control_scale=control_scale, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    os.replace(partial, artifact)
    report: dict[str, Any] = {
        **metadata, "status": "complete_train_only_source_balanced_fit", "worktree_clean": clean,
        "artifact": str(artifact.resolve()), "artifact_sha256": sha256_file(artifact),
        "feature_mean": mean.tolist(), "feature_std": std.tolist(), "ridge": ridge.tolist(),
        "target_score_power": pz.tolist(), "predicted_score_power": pp.tolist(), "innovation_score_power": pe.tolist(),
        "explained_power_fraction": (pp / pz).tolist(), "candidate_innovation_scale": candidate_scale.tolist(), "zero_predictor_scale": control_scale.tolist(),
        "validation_opened": False, "posthoc_Ak_used": False, "Astrid_accessed": False, "historical_EAGLE_accessed": False,
    }
    report["decision_digest_sha256"] = canonical_digest(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    partial_report = report_path.with_suffix(report_path.suffix + ".partial")
    partial_report.write_text(json.dumps(report, indent=2) + "\n"); os.replace(partial_report, report_path)
    print(json.dumps(report, indent=2), flush=True)
    return report


def load_model(path: Path, digest: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError("V38 model hash differs")
    with np.load(path, allow_pickle=False) as handle:
        model = {key: np.asarray(handle[key]) for key in handle.files}
    metadata = json.loads(str(model.pop("metadata_json").item()))
    if metadata.get("schema") != MODEL_SCHEMA or metadata.get("program_sha256") != PROGRAM_SHA256:
        raise ValueError("V38 model metadata differs")
    model["metadata"] = metadata
    return model


def predict_score(features: np.ndarray, model: Mapping[str, Any]) -> np.ndarray:
    x = feature_transforms(features, np.asarray(model["feature_mean"]), np.asarray(model["feature_std"]))
    masks = fourier_masks(64, 0.3125, np.asarray(model["edges"]))
    output = np.zeros((64, 64, 64), dtype=np.complex128)
    beta = np.asarray(model["beta"])
    for band, mask in enumerate(masks):
        output[mask] = np.sum(beta[band, :, None] * x[:, mask], axis=0)
    output[0, 0, 0] = 0
    result = np.fft.ifftn(output).real
    return result - result.mean()


def scale_bands(field: np.ndarray, scales: np.ndarray, edges: np.ndarray) -> np.ndarray:
    transform = np.fft.fftn(np.asarray(field, dtype=np.float64))
    output = np.zeros_like(transform)
    for band, mask in enumerate(fourier_masks(64, 0.3125, edges)):
        output[mask] = scales[band] * transform[mask]
    output[0, 0, 0] = 0
    result = np.fft.ifftn(output).real
    return result - result.mean()


def density_sample(score: np.ndarray, backbone: np.ndarray, copula: Mapping[str, Any]) -> tuple[np.ndarray, float]:
    residual = conditional_inverse(ndtr(score), backbone, copula).astype(np.float64)
    residual -= residual.mean(axis=(-3, -2, -1), keepdims=True)
    dc = float(np.max(np.abs(residual.mean(axis=(-3, -2, -1)))))
    sample = np.asarray(backbone, dtype=np.float64) + residual
    if not np.isfinite(sample).all():
        raise RuntimeError("V38 generated nonfinite density")
    return sample.astype(np.float32), dc


def preflight(program_path: Path, repo: Path, model_path: Path, model_sha: str, report_path: Path, report_sha: str, output: Path) -> dict[str, Any]:
    program, v35 = load_program(program_path, repo); commit, clean = git_state(repo.resolve())
    if not clean: raise RuntimeError("V38 preflight requires clean worktree")
    model = load_model(model_path, model_sha); report = _verified_json(report_path, report_sha, "V38 fit report")
    if report.get("artifact_sha256") != model_sha or report.get("code_commit") != commit: raise ValueError("V38 fit binding differs")
    selection = _selection_arrays(v35); domain = DOMAIN_ORDER[0]; query_index = int(selection[domain]["source_index"][0]); donor_source = DOMAIN_ORDER[int(selection[domain]["donor_source"][0,0])]; donor_index = int(selection[domain]["donor_index"][0,0]); iso = int(selection[domain]["donor_isometry"][0,0])
    qd,qc=_open_split(v35["development_domains"][domain],"validation"); dd,dcache=_open_split(v35["development_domains"][donor_source],"train")
    copula=load_v31(Path(program["inherited_inputs"]["conditional_copula_artifact"]),program["inherited_inputs"]["conditional_copula_artifact_sha256"])
    try:
        db=_backbone(dcache,donor_index)[None]; truth=np.asarray(dd["target"][donor_index],dtype=np.float32); z=gaussian_score(truth-db,db,copula)[0]; pd=predict_score(feature_cube(dd,dcache,donor_index),model); innovation=z-pd
        reconstruction=float(np.sqrt(np.mean(np.square(z-(pd+innovation)))))
        perm,ref=CUBE_ISOMETRIES[iso]; innovation=apply_cube_isometry(innovation,perm,ref); qb=_backbone(qc,query_index)[None]; pq=predict_score(feature_cube(qd,qc,query_index),model)
        candidate=pq+scale_bands(innovation,np.asarray(model["candidate_scale"]),np.asarray(model["edges"])); control=scale_bands(innovation,np.asarray(model["control_scale"]),np.asarray(model["edges"])); cs,cdc=density_sample(candidate[None],qb,copula); zs,zdc=density_sample(control[None],qb,copula)
    finally: qd.close(); qc.close(); dd.close(); dcache.close()
    if reconstruction>1e-6 or cdc>1e-7 or zdc>1e-7: raise RuntimeError("V38 real preflight failed")
    payload={"schema":PREFLIGHT_SCHEMA,"status":"pass","program_sha256":PROGRAM_SHA256,"code_commit":commit,"worktree_clean":clean,"model":str(model_path.resolve()),"model_sha256":model_sha,"report":str(report_path.resolve()),"report_sha256":report_sha,"real_reconstruction_rms":reconstruction,"candidate_maximum_dc":cdc,"control_maximum_dc":zdc,"validation_truth_used":False,"donor_translation":False,"donor_reselection":False,"posthoc_Ak_used":False,"Astrid_accessed":False,"historical_EAGLE_accessed":False}; payload["decision_digest_sha256"]=canonical_digest(payload)
    if output.exists(): raise RuntimeError("V38 refuses pre-existing preflight")
    output.parent.mkdir(parents=True,exist_ok=True); partial=output.with_suffix(output.suffix+".partial"); partial.write_text(json.dumps(payload,indent=2)+"\n"); os.replace(partial,output); print(json.dumps(payload,indent=2)); return payload


def _new_ensemble(handle: h5py.File) -> dict[str,h5py.Dataset]:
    return {"sample":handle.create_dataset("sample",shape=(16,16,1,64,64,64),dtype="f4",chunks=(1,1,1,64,64,64),compression="lzf"),"conditional_mean":handle.create_dataset("conditional_mean",shape=(16,1,64,64,64),dtype="f4",compression="lzf"),"truth":handle.create_dataset("truth",shape=(16,1,64,64,64),dtype="f4",compression="lzf")}


def sample_all(program_path: Path, repo: Path, model_path: Path, model_sha: str, report_path: Path, report_sha: str, preflight_path: Path, preflight_sha: str, output_root: Path) -> None:
    program,v35=load_program(program_path,repo); commit,clean=git_state(repo.resolve())
    if not clean: raise RuntimeError("V38 sampling requires clean worktree")
    model=load_model(model_path,model_sha); report=_verified_json(report_path,report_sha,"V38 report"); pf=_verified_json(preflight_path,preflight_sha,"V38 preflight")
    if report.get("artifact_sha256")!=model_sha or pf.get("code_commit")!=commit or pf.get("model_sha256")!=model_sha: raise ValueError("V38 sampling bindings differ")
    if output_root.exists(): raise RuntimeError("V38 refuses pre-existing output root")
    copula=load_v31(Path(program["inherited_inputs"]["conditional_copula_artifact"]),program["inherited_inputs"]["conditional_copula_artifact_sha256"]); selection=_selection_arrays(v35)
    train={d:_open_split(v35["development_domains"][d],"train") for d in DOMAIN_ORDER}; donor_cache:dict[tuple[str,int],tuple[np.ndarray,np.ndarray]]={}
    try:
        for domain in DOMAIN_ORDER:
            row=v35["development_domains"][domain]; indices=np.asarray(selection[domain]["source_index"],dtype=np.int64); qd,qc=_open_split(row,"validation")
            handles={}; datasets={}; partials={}; maxdc={a:0.0 for a in ARMS}
            try:
                for arm in ARMS:
                    path=output_root/arm/"development_candidate"/DOMAIN_KEYS[domain]/"ensemble16.h5"; path.parent.mkdir(parents=True,exist_ok=True); partial=path.with_suffix(path.suffix+".partial"); partials[arm]=partial; handles[arm]=h5py.File(partial,"w"); datasets[arm]=_new_ensemble(handles[arm]);
                    for name,value in selection[domain].items(): handles[arm].create_dataset(name,data=value)
                for oi,qi in enumerate(indices):
                    qb=_backbone(qc,int(qi))[None]; pq=predict_score(feature_cube(qd,qc,int(qi)),model)
                    for member in range(16):
                        ds=DOMAIN_ORDER[int(selection[domain]["donor_source"][oi,member])]; di=int(selection[domain]["donor_index"][oi,member]); iso=int(selection[domain]["donor_isometry"][oi,member]); key=(ds,di)
                        if key not in donor_cache:
                            dd,dc=train[ds]; db=_backbone(dc,di)[None]; truth=np.asarray(dd["target"][di],dtype=np.float32); z=gaussian_score(truth-db,db,copula)[0]; donor_cache[key]=(z,predict_score(feature_cube(dd,dc,di),model))
                        z,pd=donor_cache[key]; perm,ref=CUBE_ISOMETRIES[iso]; innovation=apply_cube_isometry(z-pd,perm,ref)
                        scores={"query_conditioned":pq+scale_bands(innovation,np.asarray(model["candidate_scale"]),np.asarray(model["edges"])),"zero_predictor_control":scale_bands(innovation,np.asarray(model["control_scale"]),np.asarray(model["edges"]))}
                        for arm in ARMS:
                            sample,dcvalue=density_sample(scores[arm][None],qb,copula); datasets[arm]["sample"][oi,member]=sample; maxdc[arm]=max(maxdc[arm],dcvalue)
                    for arm in ARMS: datasets[arm]["conditional_mean"][oi]=qb; datasets[arm]["truth"][oi]=np.asarray(qd["target"][int(qi)],dtype=np.float32)
                    print(f"[v38-sample] {domain} {oi+1}/16",flush=True)
                for arm in ARMS:
                    handles[arm].attrs.update({"schema":ENSEMBLE_SCHEMA,"method":"train_only_query_conditioned_gaussian_copula_innovation","arm":arm,"v38_program_sha256":PROGRAM_SHA256,"model":str(model_path.resolve()),"model_sha256":model_sha,"report":str(report_path.resolve()),"report_sha256":report_sha,"preflight":str(preflight_path.resolve()),"preflight_sha256":preflight_sha,"parent_selection":str(Path(row["phase_object_selection"]).resolve()),"parent_selection_sha256":row["phase_object_selection_sha256"],"diagnostic_k_h_mpc":1.0,"maximum_absolute_residual_dc":maxdc[arm],"ensemble_members":16,"donor_translation":False,"donor_reselection":False,"validation_truth_used_for_fit_or_sampling":False,"density_field_clipping":False,"posthoc_Ak_used":False,"worktree_clean_at_sampling":clean,"sampling_code_commit":commit,"Astrid_accessed":False,"historical_EAGLE_accessed":False,"complete":True})
            finally:
                for h in handles.values(): h.close()
                qd.close(); qc.close()
            for arm in ARMS: os.replace(partials[arm],partials[arm].with_suffix(""))
    finally:
        for dd,dc in train.values(): dd.close(); dc.close()


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="command",required=True)
    fit=sub.add_parser("fit"); fit.add_argument("--program",type=Path,required=True); fit.add_argument("--repo",type=Path,required=True); fit.add_argument("--artifact",type=Path,required=True); fit.add_argument("--report",type=Path,required=True)
    pf=sub.add_parser("preflight"); pf.add_argument("--program",type=Path,required=True); pf.add_argument("--repo",type=Path,required=True); pf.add_argument("--model",type=Path,required=True); pf.add_argument("--model-sha256",required=True); pf.add_argument("--report",type=Path,required=True); pf.add_argument("--report-sha256",required=True); pf.add_argument("--out",type=Path,required=True)
    sample=sub.add_parser("sample"); sample.add_argument("--program",type=Path,required=True); sample.add_argument("--repo",type=Path,required=True); sample.add_argument("--model",type=Path,required=True); sample.add_argument("--model-sha256",required=True); sample.add_argument("--report",type=Path,required=True); sample.add_argument("--report-sha256",required=True); sample.add_argument("--preflight",type=Path,required=True); sample.add_argument("--preflight-sha256",required=True); sample.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="fit": fit_model(args.program,args.repo,args.artifact,args.report)
    elif args.command=="preflight": preflight(args.program,args.repo,args.model,args.model_sha256,args.report,args.report_sha256,args.out)
    else: sample_all(args.program,args.repo,args.model,args.model_sha256,args.report,args.report_sha256,args.preflight,args.preflight_sha256,args.out)


if __name__=="__main__": main()
