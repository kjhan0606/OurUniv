# Bundle B: LG observations and the fine-field interface

This is a source-backed **development contract**, not an evaluated LG
posterior. N32 cannot resolve these objects. M33 is mandatory, MW fixes the
observer identity, and there is no anonymous best-pair search.

| Identity | Conditioned ICRS sky (degrees) | Distance modulus (mag) | Heliocentric LOS (km/s) | (mu_alpha*, mu_delta), microarcsec/yr |
| --- | --- | --- | --- | --- |
| MW | Observer halo; Sun offset explicitly supplied | Solar R0 reference8.122 kpc, not an external galaxy distance datum | Solar reflex nuisance, not zero heliocentric halo speed | Not a duplicate external proper-motion datum |
| M31 | (10.68,41.27) | 24.407 +/-0.032 | -300.8 +/-1.2 | (45.9 +/-8.1, -20.5 +/-6.6) |
| M33 | (23.46,30.66) | 24.622 +/-0.030 | -179.5 +/-0.6 | (45.3 +/-9.7, 26.3 +/-7.3) |

Distances: [Li et al.2021, section4.1 and Table4](https://arxiv.org/pdf/2107.08029)
and [Breuval et al.2023, sectionIV.5 and Table4](https://arxiv.org/html/2304.00037v1).
The two Cepheid ladders share LMC calibration. The partial covariance includes
the common0.026 mag DEB term and shared LMC PLR terms (fractional0.42%/0.41%);
these are already within marginal errors, not extra diagonal inflation.

Sky/motion entries use the5p results, including published zero-point error,
from [Wu et al.2025, sections2.2/3.1 and Tables3–4](https://arxiv.org/html/2508.01127v1).
Their solar reference is(U,V_c+V,W)=(7.01,244.17,4.95)km/s. Their PM reductions
assume distances784/840 kpc; changing distance must eventually propagate into
the disk-rotation reduction, not only the transverse-speed conversion.
COM component/cross-galaxy covariances are not tabulated here. The callable
therefore requires a joint covariance or explicit opt-in to a **partial
development approximation**. It does not silently declare unknown terms zero.
Earlier Gaia estimates, combined Gaia/VLBA estimates, and derived total speeds
are alternatives, not additional independent factors.

## Physical conventions and remaining calibration

`src/cf4_lg_observation_contract.py` accepts identity-labelled positions in
physical kpc at z=0 and peculiar velocities in km/s, both along ICRS axes.
Caller supplies Sun-minus-MW position and TOTAL solar velocity in the same axes.
For separation dr from MW, v_total = v_pec,relative +100*h*dr/1000. Subtract
solar velocity, then project on the actual heliocentric sky basis. PM uses
v_t=4.74047*mu(mas/yr)*D(kpc). There is no independent derived v_rad/v_t factor.
The numerical roundtrip tests include the Hubble term and solar reflex.
Conditioned sky precision0.01deg only accommodates the rounded source centres;
it is not a manufactured measurement error or permission to move the observer.

The R0 reference and illustrative mass assumptions come from
[Sawala et al.2025, LG model and Methods](https://www.nature.com/articles/s41550-025-02563-1):
MW1.0+/-0.2, M311.3+/-0.4, M330.3+/-0.1, in10^12 physical Msun, labelled
M200c. These are **astrophysical priors**, not direct mass measurements. MW
excludes a separately treated LMC. M33's isolated-halo mass assumption cannot
be substituted for its stripped bound subhalo mass. Mass priors are recorded
but disabled pending a resolved mass operator and LMC/reflex treatment.

Do not inherit the old80 km/s screening widths or isolation mass cut. Numerical
PM discrepancy, stellar-disk versus halo COM offset, solar uncertainties,
LMC-induced motion, tracer systematics, and environmental priors are separate
model ingredients. No automatic first-infall/binding/isolation likelihood is
inferred from an orbit paper. Calibration and sensitivity tests remain needed.
Before combining with the parent posterior, audit any overlapping CF4 distance
data/calibration; this interface alone does not authorize counting them twice.

## Coarse/fine connection for the next bundle

Fine density must be positive and satisfy sum(rho_f*V_f)=rho_c*V_c in each
parent cell; sum(rho_f*v_f*V_f)=rho_c*v_c*V_c similarly fixes momentum. These
constraints do not determine every fine mode. Update that conditional fine
field using the LG observations above and an explicitly defined halo/subhalo
readout. HOP on N32 and painting three arbitrary peaks are not that readout.
Report information gain and uncertainty at LG<=0.3 cMpc/h; do not confuse this
with final zoom particle/force resolution. No actual LG conditioning ran in B.
