# R2 — Tully raw-HV frame recovery and source discrepancy (2026-10-03)

## Scope and source correction

The previous source note was too broad. Tully (2015) table4 publishes adjusted
`Vcmba`, which must not be treated as ordinary raw CMB velocity or inverted
without the distance-error inputs. However, table5 also publishes individual
heliocentric velocity `HV` and Galactic longitude/latitude. With the standard
heliocentric-to-CMB dipole correction, table5 therefore does provide a route
to each member's *unadjusted* CMB-frame velocity. The official ReadMe lists
these fields and separately labels `Vcmba` as adjusted.

The 2M++ ReadMe lists both heliocentric `HV` and CMB-frame `Vcmb`, and says its
frame conversion follows Kogut et al. (1993) and Tully et al. (2008). To
calibrate the exact vector used by this frozen 2M++ snapshot, fit
`Vcmb - HV = v_sun · n(l,b)` by least squares over its 67,320 rows with valid
velocities/coordinates, `Cln=0`, and non-ZOA `Ref`. Here
`n=(cos(b)cos(l), cos(b)sin(l), sin(b))`. The recovered Galactic Cartesian
vector is `(-24.9956, -246.0032, 276.9978) km/s`, amplitude 371.309 km/s,
toward `(l,b)=(264.1983°,48.2454°)`. Residuals against the catalogued 2M++
corrections have median `-0.0004`, 90th percentile absolute `0.615`, and
maximum absolute `0.999 km/s`, consistent with integer-km/s catalog rounding.
No fitted astrophysical parameter is carried into the field likelihood.

Primary source descriptions: [Tully (2015) CDS ReadMe](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/AJ/149/171?format=html&tex=true),
[2M++ CDS ReadMe](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/MNRAS/416/2840?format=html&tex=true),
[Tully et al. (2008) CDS ReadMe](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/ApJ/676/184?format=html&tex=true),
and [Huchra et al. (2012) 2MRS CDS ReadMe](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/ApJS/199/26?format=html&tex=true).
The calculation uses the existing frozen local inputs: Tully table5 gzip SHA256
`337b9f24484ae34c973a7187fa002807b345e47e76ffabce066549783e2b2844`, 2M++
catalog CSV `05d39f49af58caa7aa199420cc7354b3aa9fe3dbacf8d6c33222479c288fb23d`,
CF4 individuals `28e7b8bd386f53716ed84cddd67a6f7602f98bc1394923a312f906555da7f709`,
and CF4 groups `bfdc0cfc0f172b48468e3a8fd05e87978c1ec68c341fb2d929fc1200f0123334`.

## Frozen-v6 results

The source cohort is the same 272 training groups / 828 secure links as the
preceding audit. Its 441 members whose Tully table4 Nest matches the Tully
parent of their CF4 `1PGC` all have a table5 raw `HV` and coordinates. For each,
reconstruct ordinary CMB velocity from table5 `HV` and the dipole vector above,
then compare with the matched 2M++ point's raw CMB `Vcmb`:

- absolute difference: median `11.54`, p90 `61.41`, maximum `5598.95 km/s`;
- 63/441 exceed 50, 22 exceed 100, 7 exceed 200, and 1 exceeds 500 km/s.

The extreme row is PGC33946 in Tully Nest100020. Tully table5 has
`HV=5160 km/s`, giving reconstructed raw `Vcmb=5463.05 km/s`; the matched 2M++
record `11104662+2816428` has `HV=10759`, `Vcmb=11062 km/s`. The frozen CF4
individual has `Vcmb=11081 km/s`, only 19 km/s from 2M++. The official 2MRS
table3 row for this same 2MASS identifier lists barycentric `cz=10768 km/s`
and reference `2011SDSS8.C...0000:`. That barycentric `cz` is recorded as
source metadata only; it is not numerically compared to CMB-frame velocities.
The 2M++ row's `Ref` is instead `2004ApJ...607..202M`, and Tully table5 does
not provide a per-row source reference for its `HV`. Available evidence shows
a large archived redshift disagreement, but does not adjudicate which
measurement should be treated as correct. Tully `Vcmba=11393` is the
cosmologically adjusted catalog field, not the raw `HV` measurement.

For the 60 selected Nests in which every table4 member is represented by the
frozen 2M++ training links (42 singletons and 18 pairs), the arithmetic mean
of reconstructed table5 raw member `Vcmb` values agrees with the local CF4
group `Vcmb`: absolute residual median `5.33`, p90 `21.37`, maximum
`73.996 km/s`. This is consistent with the EDD description of CF4 group
velocity aggregation for this small selected subset. It is not evidence for a
group covariance law, full-sample representativeness, group inclusion
probability, or independence from the 2M++ counts. The severe PGC33946 row is
in a 27-member Nest, not among these 60 fully linked singleton/pair controls.

## Driver review and decision

- Q-GOAL: yes. Correct member-velocity semantics and the PGC33946 discrepancy
  affect whether CF4 group marks and 2M++ counts can be combined as dependent
  observations of the same z=0 field. This creates no new field constraint.
- Q-LEAN: yes. One algebraic frame calibration plus the existing 441-row
  cohort and 60 complete-Nest check is proportionate; no new fit, likelihood,
  heldout read, posterior, simulation, or source-residual ladder was run.
- Review: driver-reviewed. An external review here would immediately repeat
  the preceding driver source audit; under the user's reviewer-routing rule,
  the driver performs this check.

Next R2 work remains the shared-group observation-law/calibration branch.
It must preserve member-level source disagreement/outlier uncertainty and
group-inclusion ambiguity rather than turning these residuals into an assumed
covariance or multiplying a second redshift factor. The current moment target
still uses its explicitly documented single-mark association assumption; its
shared-group kernels still lack calibrated covariance/inclusion inputs and are
not wired into the target. No posterior or production field is promoted.
MW/M31 remain role-ambiguous and M33 unresolved; all three must eventually
constrain the same NEW field at LG resolution <=0.3 cMpc/h, with native truth
identities restricted to calibration/evaluation.
