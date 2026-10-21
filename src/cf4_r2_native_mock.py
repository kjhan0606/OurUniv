"""Native galaxy observation mock for an explicitly defined source window.

External calibration geometry only. Disjoint native halves are translated
near/far so each native galaxy appears once; no gravity or LG assignment.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid

OBS_EDGES = np.array([-25., -23.-2./3., -22.-1./3., -21.])
TRUE_EDGES = np.array([-np.inf, -25., -23.-2./3., -22.-1./3., -21., np.inf])


def distance_tables(omega_m):
    z = np.linspace(0., .1, 20001)
    radius = cumulative_trapezoid(299792.458/100./np.sqrt(
        omega_m*(1+z)**3+1-omega_m), z, initial=0.)
    table = np.linspace(.001,192.,20001)
    redshift = np.interp(table,radius,z)
    return table, redshift, 5*np.log10(table*(1+redshift))+25.


def place_native_halves(positions):
    positions = np.asarray(positions,dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
        raise ValueError('finite native positions required')
    native = positions % 75.
    mapped = native + np.array([154.5,154.5,154.5])
    mapped[:,0] += np.where(native[:,0] >= 37.5,84.,0.)
    return mapped


def observe(positions,velocities,magnitude_h,radial_table,z_table,modulus_table):
    """Spherical RSD with h/H0=.01 and the SAME K correction as the target."""
    positions, velocities = np.asarray(positions), np.asarray(velocities)
    magnitude_h = np.asarray(magnitude_h)
    if positions.shape != velocities.shape or positions.shape != (len(magnitude_h),3):
        raise ValueError('aligned native observation arrays required')
    relative = (positions-192.+192.)%384.-192.
    rt = np.linalg.norm(relative,axis=1)
    direction = relative/np.where(rt>0,rt,1.)[:,None]
    vlos = np.sum(velocities*direction,axis=1)
    observed = (positions+.01*vlos[:,None]*direction)%384.
    ro = np.linalg.norm((observed-192.+192.)%384.-192.,axis=1)
    mt = np.interp(rt,radial_table,modulus_table)
    mo = np.interp(ro,radial_table,modulus_table)
    zt = np.interp(rt,radial_table,z_table)
    zo = np.interp(ro,radial_table,z_table)
    correction = 1.16*2.9*(zo-zt)-1.6*np.log10((1+zo)/(1+zt))
    apparent = magnitude_h+mt+correction
    absolute = apparent-mo
    absolute_bin = np.searchsorted(OBS_EDGES,absolute,side='right')-1
    selected = ((rt>0)&(ro>=5.)&(ro<180.)&(apparent<=12.5)&
                (absolute_bin>=0)&(absolute_bin<3))
    population = 3*(apparent>11.5).astype(int)+absolute_bin
    true_bin = np.searchsorted(TRUE_EDGES[1:-1],magnitude_h,side='right')
    cell = np.floor(observed/3.).astype(int)%128
    spatial_key = np.ravel_multi_index(tuple(cell.T),(128,128,128))
    key = population[selected]*128**3+spatial_key[selected]
    keys,counts = np.unique(key,return_counts=True)
    return dict(keys=keys,counts=counts,selected=selected,population=population,
        true_bin=true_bin,positions=observed,true_radius=rt,observed_radius=ro,
        apparent_K=apparent,observed_absolute_K=absolute,direction=direction,
        coherent_vlos=vlos)


def exposure_and_radial_bins():
    flat = np.arange(128**3)
    x, y, z = flat//128**2, (flat//128)%128, flat%128
    radius = np.sqrt(((x+.5)*3-192.)**2+((y+.5)*3-192.)**2+((z+.5)*3-192.)**2)
    rbin = np.clip((radius//12).astype(int),0,15)
    dy = (y+.5)*3-192.
    return dy < -6., dy > 6., rbin


def aggregate(keys,counts,exposure,rbin):
    keep = exposure[keys%128**3]
    index = keys[keep]//128**3*16+rbin[keys[keep]%128**3]
    return np.bincount(index,weights=counts[keep],minlength=96).reshape(6,16)
