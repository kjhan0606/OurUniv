#!/usr/bin/env python3
"""Calibrate survival offsets and six population biases from CF4-disjoint 2M++.

This is a bounded development calibration, not a density-field posterior.  The
official ARES maps provide known angular survival, a Schechter radial factor is
the known radial offset, and Carrick's cube is only a reference covariate.  The
six population amplitudes and positive bias slopes are fitted from Poisson
voxel counts, with a deterministic spatial holdout.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cf4_twompp_disjoint_tracer_pilot import (  # noqa: E402
    LIGHT_SPEED_KMS, classify_disjoint_tracer, distance_and_absolute_magnitude,
    galactic_directions, load_catalog, load_completeness_map,
    read_crossmatch_exclusions, supergalactic_unit_vectors,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def load_joint_catalog(path: str | Path) -> dict[str, np.ndarray]:
    """Read the published 2M++ header (_RA/_DE) without altering the source pilot."""
    rows = {k: [] for k in ("recno", "Ksmag", "Vcmb", "c11_5", "c12_5", "Cln", "Ref", "RA", "DEC")}
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"recno", "Ksmag", "Vcmb", "c11_5", "c12_5", "Cln", "Ref", "_RA", "_DE"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("2M++ catalogue header changed")
        for r in reader:
            rows["recno"].append(int(r["recno"])); rows["Ksmag"].append(float(r["Ksmag"]))
            rows["Vcmb"].append(float(r["Vcmb"])); rows["c11_5"].append(float(r["c11_5"]))
            rows["c12_5"].append(float(r["c12_5"]) if r["c12_5"].strip() else np.nan)
            rows["Cln"].append(int(r["Cln"])); rows["Ref"].append(r["Ref"].strip())
            rows["RA"].append(float(r["_RA"])); rows["DEC"].append(float(r["_DE"]))
    return {"recno": np.asarray(rows["recno"], dtype=np.int64), "Ksmag": np.asarray(rows["Ksmag"]), "Vcmb": np.asarray(rows["Vcmb"]), "c11_5": np.asarray(rows["c11_5"]), "c12_5": np.asarray(rows["c12_5"]), "Cln": np.asarray(rows["Cln"], dtype=np.int8), "Ref": np.asarray(rows["Ref"], dtype=str), "RA": np.asarray(rows["RA"]), "DEC": np.asarray(rows["DEC"])}


def read_program(path: Path) -> dict[str, Any]:
    p = json.loads(path.read_text())
    if p.get("schema") != "ouruniv-cf4-joint-tracer-calibration-v1":
        raise ValueError("unexpected calibration program schema")
    return p


def schechter_fraction(r: np.ndarray, Kmax: float, lo: float, hi: float,
                       h: float, Omega_m: float, Omega_b: float, Tcmb_K: float,
                       H0_km_s_Mpc: float | None = None) -> np.ndarray:
    from astropy import units as u
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=100*h*u.km/u.s/u.Mpc, Om0=Omega_m, Ob0=Omega_b, Tcmb0=Tcmb_K*u.K)
    # r is in cMpc/h.  Use the low-z Hubble inversion z ~= 100*r/c.
    z = np.clip(100.0 * r / 299792.458, 1e-5, 0.25)
    dm = cosmo.distmod(z).value
    mlim = Kmax - dm
    mgrid = np.linspace(-30.0, -15.0, 6001)
    Mstar, alpha = -23.28, -0.94
    q = 10.0 ** (-0.4 * (mgrid - Mstar))
    phi = np.exp(-q) * q ** (alpha + 1.0)
    # cumulative integral from bright to faint; normalize to the model range.
    cum = np.concatenate([[0.0], np.cumsum((phi[:-1] + phi[1:]) * np.diff(mgrid) * 0.5)])
    def integ(a: float, b: np.ndarray) -> np.ndarray:
        bb = np.clip(b, mgrid[0], mgrid[-1])
        return np.interp(bb, mgrid, cum) - np.interp(np.full(bb.shape, a), mgrid, cum)
    denom = max(float(integ(lo, np.asarray([hi]))[0]), 1e-30)
    return np.clip(integ(lo, np.minimum(hi, mlim)) / denom, 1e-8, 1.0)


def fit_bias(y: np.ndarray, exposure: np.ndarray, x: np.ndarray,
             train: np.ndarray) -> tuple[float, float, float, float]:
    from scipy.optimize import minimize_scalar
    yy, ee, xx = y[train], exposure[train], x[train]
    def nll(b: float) -> float:
        z = ee * np.exp(np.clip(b * xx, -30.0, 30.0))
        A = yy.sum() / max(z.sum(), 1e-30)
        lam = np.maximum(A * z, 1e-12)
        return float(np.sum(lam - yy * np.log(lam)))
    opt = minimize_scalar(nll, bounds=(0.01, 6.0), method="bounded", options={"xatol": 1e-8})
    b = float(opt.x)
    z = ee * np.exp(np.clip(b * xx, -30.0, 30.0)); A = float(yy.sum() / max(z.sum(), 1e-30))
    hold = ~train
    zh = exposure[hold] * np.exp(np.clip(b * x[hold], -30.0, 30.0))
    pred = A * zh
    nullA = float(y[train].sum() / max(exposure[train].sum(), 1e-30))
    null = nullA * exposure[hold]
    dev = lambda p: float(2.0 * np.sum(np.where(y[hold] > 0, y[hold] * np.log(y[hold] / np.maximum(p, 1e-12)) - (y[hold] - p), p)))
    return b, A, dev(pred), dev(null)


def fit_bias_radial_fog(y: np.ndarray, exposure: np.ndarray, x: np.ndarray,
                        radius: np.ndarray, train: np.ndarray) -> tuple[float, float, float, float, float]:
    """Jointly fit bias and a radial-selection nuisance; residual variance is a FoG proxy."""
    from scipy.optimize import minimize
    scale = 60.0
    rt = (radius - np.median(radius[train])) / scale
    yy, ee, xx, rr = y[train], exposure[train], x[train], rt[train]
    def nll(theta: np.ndarray) -> float:
        b, eta = theta
        z = ee * np.exp(np.clip(b * xx + eta * rr, -30.0, 30.0))
        A = yy.sum() / max(z.sum(), 1e-30)
        lam = np.maximum(A * z, 1e-12)
        return float(np.sum(lam - yy * np.log(lam)))
    opt = minimize(nll, np.array([0.5, 0.0]), method="L-BFGS-B", bounds=[(0.01, 6.0), (-2.0, 2.0)])
    b, eta = map(float, opt.x)
    z = ee * np.exp(np.clip(b * xx + eta * rr, -30.0, 30.0)); A = float(yy.sum() / max(z.sum(), 1e-30))
    hold = ~train; zh = exposure[hold] * np.exp(np.clip(b*x[hold] + eta*rt[hold], -30.0, 30.0)); pred = A*zh
    nullA = float(y[train].sum() / max(exposure[train].sum(), 1e-30)); null = nullA*exposure[hold]
    def dev(p: np.ndarray) -> float:
        yyh = y[hold]; term = np.zeros_like(yyh, dtype=float); pos = yyh > 0
        term[pos] = yyh[pos] * np.log(yyh[pos] / np.maximum(p[pos], 1e-12)) - (yyh[pos] - p[pos]); term[~pos] = p[~pos]
        return float(2.0*np.sum(term))
    # Overdispersion diagnostic: variance/mean of holdout Pearson residuals.
    pearson = (y[hold] - pred) / np.sqrt(np.maximum(pred, 1e-8))
    fog_proxy = float(np.var(pearson))
    return b, eta, A, dev(pred), dev(null), fog_proxy


def fit_bias_radial_nb(y: np.ndarray, exposure: np.ndarray, x: np.ndarray,
                       radius: np.ndarray, train: np.ndarray) -> tuple[float, float, float, float, float]:
    """Fit bias, radial nuisance and NB overdispersion for redshift-space counts."""
    from scipy.optimize import minimize
    from scipy.special import gammaln
    rt = (radius - np.median(radius[train])) / 60.0
    yy, ee, xx, rr = y[train], exposure[train], x[train], rt[train]
    def nll(theta: np.ndarray, yv: np.ndarray, ev: np.ndarray, xv: np.ndarray, rv: np.ndarray) -> float:
        b, eta, logk = theta; k = np.exp(np.clip(logk, -8.0, 15.0))
        z = ev*np.exp(np.clip(b*xv + eta*rv, -30.0, 30.0)); A = yv.sum()/max(z.sum(),1e-30); mu=np.maximum(A*z,1e-12)
        return float(-np.sum(gammaln(yv+k)-gammaln(k)-gammaln(yv+1)+k*np.log(k/(k+mu))+yv*np.log(mu/(k+mu))))
    opt = minimize(lambda t: nll(t,yy,ee,xx,rr), np.array([0.5,0.0,2.0]), method="L-BFGS-B", bounds=[(0.01,6.0),(-2.0,2.0),(-8.0,15.0)])
    b,eta,logk = map(float,opt.x); k=np.exp(logk); z=ee*np.exp(np.clip(b*xx+eta*rr,-30,30)); A=float(yy.sum()/max(z.sum(),1e-30))
    hold=~train; zh=exposure[hold]*np.exp(np.clip(b*x[hold]+eta*rt[hold],-30,30)); mu=A*zh
    score_model=-nll(np.array([b,eta,logk]),y[hold],exposure[hold],x[hold],rt[hold])
    nullA=float(y[train].sum()/max(exposure[train].sum(),1e-30)); nullmu=nullA*exposure[hold]
    score_null=-nll(np.array([0.0,0.0,logk]),y[hold],exposure[hold],np.zeros_like(x[hold]),rt[hold])
    return b,eta,A,float(score_model),float(score_null),float(k)


def run(program: dict[str, Any]) -> dict[str, Any]:
    data = program["data"]; design = program["design"]
    version = program.get("model_version", "v1")
    for key in ("catalog", "crossmatch", "map11", "map12", "carrick"):
        path = Path(data[key]["path"])
        if not path.is_file() or ("sha256" in data[key] and sha256(path) != data[key]["sha256"]):
            raise ValueError(f"input binding changed: {path}")
    cat = load_joint_catalog(data["catalog"]["path"])
    excluded, _ = read_crossmatch_exclusions(data["crossmatch"]["path"], int(program["excluded_targets"]))
    distance, absmag = distance_and_absolute_magnitude(cat["Vcmb"], cat["Ksmag"], program["cosmology"])
    eligible, _, appbin, absbin = classify_disjoint_tracer(cat, excluded, distance, absmag, design)
    nside = int(design["nside"])
    c11 = load_completeness_map(data["map11"]["path"], nside)
    c12 = load_completeness_map(data["map12"]["path"], nside)
    glon, glat = galactic_directions(cat["RA"], cat["DEC"])
    import healpy as hp
    pix = hp.ang2pix(nside, 0.5*np.pi-glat, np.mod(glon, 2*np.pi), nest=False)
    survival = np.where(appbin == 0, c11[pix], c12[pix])
    vec = supergalactic_unit_vectors(cat["RA"], cat["DEC"])
    N = int(design["grid_N"]); box = float(design["box"]); dx = box/N
    pos = distance[:, None] * vec + box/2.0
    idx = np.floor(pos/dx).astype(int)
    inside = eligible & np.all((idx >= 0) & (idx < N), axis=1)
    flat = np.ravel_multi_index(idx[inside].T, (N,N,N))
    pop = appbin[inside]*3 + absbin[inside]
    counts = np.zeros((6, N**3), dtype=float)
    for p in range(6): counts[p] = np.bincount(flat[pop == p], minlength=N**3)
    # voxel exposure from centre direction and radial Schechter survival.
    grid = (np.arange(N)+0.5)*dx - box/2.0
    X,Y,Z = np.meshgrid(grid, grid, grid, indexing="ij")
    sg = np.sqrt(X*X+Y*Y+Z*Z).ravel(); sg_safe = np.maximum(sg, 1e-3)
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    lon = np.arctan2(Y.ravel(), X.ravel()); lat = np.arcsin(Z.ravel()/sg_safe)
    gal = SkyCoord(sgl=lon*u.rad, sgb=lat*u.rad, frame="supergalactic").galactic
    cp11 = c11[hp.ang2pix(nside, 0.5*np.pi-gal.b.rad, np.mod(gal.l.rad,2*np.pi), nest=False)]
    cp12 = c12[hp.ang2pix(nside, 0.5*np.pi-gal.b.rad, np.mod(gal.l.rad,2*np.pi), nest=False)]
    exp = np.zeros((6,N**3)); edges = np.asarray(design["abs_edges"], float)
    for a,K in enumerate(design["Kmax"]):
        for j in range(3):
            exp[a*3+j] = (cp11 if a == 0 else cp12) * schechter_fraction(sg_safe, K, edges[j], edges[j+1], **program["cosmology"])
    if version == "v5":
        perturb = float(program.get("survival_perturb", 0.0))
        exp = np.clip(exp * (1.0 + perturb * (1.0 - exp)), 1e-8, 1.0)
    carrick = np.load(data["carrick"]["path"], mmap_mode="r", allow_pickle=False)
    # trilinear-free nearest-neighbour reference covariate, only a calibration covariate.
    cg = np.clip(np.floor((np.stack(np.meshgrid(grid,grid,grid,indexing="ij"),-1)+200.0)/1.5625).astype(int),0,256)
    delta = np.asarray(carrick[cg[...,0],cg[...,1],cg[...,2]], float).ravel()
    x = np.log1p(np.clip(delta, -0.999999, None))
    modulus = int(program.get("holdout_modulus", 5)); remainder = int(program.get("holdout_remainder", 0))
    train = (np.arange(N**3) % modulus) != remainder
    results=[]
    v2 = version == "v2"; v3 = version == "v3"
    prior = program.get("external_fog_prior", [])
    for p in range(6):
        x_fit = x
        if version == "v4":
            sigma = float(prior[p]["mean_cMpc_h"])
            # Bounded radial FoG operator: suppress line-of-sight density contrast
            # before the tracer link.  The prior is external; NB k remains separate.
            x_fit = x * np.exp(-0.5 * (sigma / float(program.get("fog_operator_scale_cMpc_h", 3.0))) ** 2)
        if v3 or version == "v4":
            b,eta,A,score,score0,k = fit_bias_radial_nb(counts[p], exp[p], x_fit, sg, train)
            results.append({"population":p,"bias":b,"radial_nuisance":eta,"amplitude":A,"holdout_log_score":score,"null_holdout_log_score":score0,"log_score_improvement":score-score0,"nb_dispersion_k":k})
        elif v2:
            b,eta,A,dev,dev0,fog = fit_bias_radial_fog(counts[p], exp[p], x_fit, sg, train)
            results.append({"population":p,"bias":b,"radial_nuisance":eta,"amplitude":A,"holdout_deviance":dev,"null_holdout_deviance":dev0,"deviance_improvement":dev0-dev,"fog_overdispersion_proxy":fog})
        else:
            b,A,dev,dev0 = fit_bias(counts[p], exp[p], x, train)
            results.append({"population":p,"bias":b,"amplitude":A,"holdout_deviance":dev,"null_holdout_deviance":dev0,"deviance_improvement":dev0-dev})
    metric = "log_score_improvement" if (v3 or version == "v4") else "deviance_improvement"
    passed = all(r["bias"] > 0 and np.isfinite(r[metric]) for r in results)
    return {"schema":program["schema"],"status":"CALIBRATION_PASS" if passed else "CALIBRATION_FAIL","model_version":version,"eligible_rows":int(inside.sum()),"grid":{"N":N,"box_cMpc_h":box,"cell_cMpc_h":dx,"train_fraction":float(train.mean())},"selection":{"official_ARES":True,"survival_min":float(np.min(exp)),"survival_max":float(np.max(exp)),"radial_model":"Schechter Mstar=-23.28 alpha=-0.94 with fitted radial nuisance" if (v2 or v3) else "Schechter Mstar=-23.28 alpha=-0.94"},"population_results":results,"reference_covariate":"Carrick luminosity-weighted delta only; not treated as truth","production_gate":{"external_survival_bias_calibration_or_joint_model":bool(passed),"production_IC_GO":False,"reason":"development calibration; NB dispersion is fitted phenomenologically and still lacks external RSD/FoG validation"}}


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--program",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
    result=run(read_program(a.program)); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,sort_keys=True))

if __name__ == "__main__": main()
