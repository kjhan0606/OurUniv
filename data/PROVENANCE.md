# CF4 data provenance

Downloaded 2026-07-07 from VizieR (CDS) via TAP, catalog **J/ApJ/944/94**
(Tully et al. 2023, "Cosmicflows-4", ApJ 944, 94).

- `cf4_groups.csv`   — 38053 rows, table `J/ApJ/944/94/groups` (grouped catalog:
  distances DMzp/Dist, peculiar velocities Vpec, SG coords).
- `cf4_galaxies.csv` — 55877 rows, table `J/ApJ/944/94/table2` (individual galaxies).

TAP endpoint: https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync
Query: `SELECT * FROM "J/ApJ/944/94/<table>"` (format=csv, MAXREC=200000).

Note: the cdsarc FTP host (cdsarc.cds.unistra.fr) is blocked from this cluster;
only vizier.cds.unistra.fr / tapvizier.cds.unistra.fr are reachable. ReadMe not
retrieved (all routes 302→cdsarc). Field semantics taken from CSV headers + TAP schema.

Units: Dist [Mpc], V* [km/s], DM* [mag], SGX/Y/Z [km/s CMB-ish, |SG|~Vls],
angles [deg]. H0(CF4)=74.6.
