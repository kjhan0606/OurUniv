"""Identity-preserving LG observation interface, deliberately not enabled on N32."""
import numpy as np

# ICRS -> Galactic rotation (J2000 convention). Vectors here use its transpose.
ICRS_TO_GALACTIC = np.array([
    [-.0548755604,-.8734370902,-.4838350155],
    [.4941094279,-.4448296300,.7469822445],
    [-.8676661490,-.1980763734,.4559837762]])


def basis(ra_deg,dec_deg):
    a,d=np.deg2rad([ra_deg,dec_deg])
    return np.array([[np.cos(d)*np.cos(a),np.cos(d)*np.sin(a),np.sin(d)],
                     [-np.sin(a),np.cos(a),0],
                     [-np.sin(d)*np.cos(a),-np.sin(d)*np.sin(a),np.cos(d)]])


def solar_reference(contract):
    ref=contract["solar_reference"]
    return (-ref["R0_kpc"]*ICRS_TO_GALACTIC[0],
            ICRS_TO_GALACTIC.T @ np.array(ref["U_Vtotal_W_km_s"]))


def predict(catalog,contract,*,h,solar_position_kpc,solar_velocity_km_s):
    if not catalog.get("resolved_halos",False):
        raise ValueError("MW/M31/M33 require resolved halo/subhalo operators, not N32 peaks")
    if catalog.get("frame") != contract["frame"]:
        raise ValueError("explicit ICRS/kpc/peculiar-km/s frame required")
    if not np.isfinite(h) or h<=0:
        raise ValueError("positive finite h required")
    mw=catalog["MW"]
    output=[]
    for name in contract["galaxy_order"]:
        item=catalog[name]  # M33 must exist; no anonymous/best pair selection.
        dr=np.asarray(item["position_kpc"])-np.asarray(mw["position_kpc"])
        dv=np.asarray(item["peculiar_velocity_km_s"])-np.asarray(mw["peculiar_velocity_km_s"])
        for v in (dr,dv,solar_position_kpc,solar_velocity_km_s):
            if np.shape(v)!=(3,) or not np.isfinite(v).all():
                raise ValueError("finite three-vectors required")
        heliopos=dr-np.asarray(solar_position_kpc)
        distance=np.linalg.norm(heliopos)
        if distance<=0:
            raise ValueError("positive heliocentric distance required")
        # At a=1: physical relative velocity = peculiar difference + H0*dr.
        # solar velocity is TOTAL solar motion relative to MW, not peculiar.
        heliovel=dv+100*h*dr/1000-np.asarray(solar_velocity_km_s)
        ra=np.rad2deg(np.arctan2(heliopos[1],heliopos[0]))%360
        dec=np.rad2deg(np.arcsin(heliopos[2]/distance))
        observed=contract["measurements"][name]
        delta_angle=np.rad2deg(np.arccos(np.clip(np.dot(heliopos/distance,
            basis(observed["ra_deg"],observed["dec_deg"])[0]),-1,1)))
        if delta_angle>contract["sky_conditioning_rounding_tolerance_deg"]:
            raise ValueError(f"{name} does not satisfy conditioned observed sky direction")
        radial,east,north=basis(ra,dec)@heliovel
        output.extend([5*np.log10(distance)+10,radial,
                       east*1000/(4.74047*distance),north*1000/(4.74047*distance)])
    return np.array(output)


def approximate_covariance(contract):
    """Includes shared ladder terms; unknown PM/cross terms are NOT measured zero."""
    sigma=np.concatenate([contract["measurements"][g]["sigma"] for g in contract["galaxy_order"]])
    cov=np.diag(sigma**2)
    c=contract["covariance"]
    common=c["distance_common_LMC_DEB_sigma_mag"]**2
    common+=(5/np.log(10))**2*np.prod(c["distance_common_LMC_PLR_fractional_sigma"])
    cov[0,4]=cov[4,0]=common
    return cov


def log_likelihood(prediction,contract,*,observation_covariance=None,
                   discrepancy_covariance=None,allow_approximate=False):
    """8-dimensional observational likelihood; mass/environment priors excluded.

    Full covariance is supplied in the declared mixed observable units, enabling
    PM, distance-ladder and numerical-discrepancy correlations without duplicate
    distance/derived-speed constraints. Marginalize shared solar nuisances outside.
    """
    if observation_covariance is None:
        if not allow_approximate:
            raise ValueError("unreported joint COM covariance: supply covariance or opt into development approximation")
        observation_covariance=approximate_covariance(contract)
    obs=np.asarray(observation_covariance,float)
    if obs.shape!=(8,8) or not np.isfinite(obs).all() or not np.allclose(obs,obs.T):
        raise ValueError("finite symmetric 8x8 observation covariance required")
    np.linalg.cholesky(obs)
    covariance=obs.copy()
    if discrepancy_covariance is not None:
        extra=np.asarray(discrepancy_covariance,float)
        if extra.shape!=(8,8) or not np.isfinite(extra).all() or not np.allclose(extra,extra.T) or np.linalg.eigvalsh(extra).min() < -1e-12:
            raise ValueError("model discrepancy must be separately declared positive semidefinite covariance")
        covariance+=extra
    mean=np.concatenate([contract["measurements"][g]["value"] for g in contract["galaxy_order"]])
    prediction=np.asarray(prediction,float)
    if prediction.shape!=(8,) or not np.isfinite(prediction).all():
        raise ValueError("finite 8-observable vector required")
    factor=np.linalg.cholesky(covariance)
    residual=np.linalg.solve(factor,prediction-mean)
    return float(-.5*np.dot(residual,residual)-np.log(np.diag(factor)).sum()-4*np.log(2*np.pi))
