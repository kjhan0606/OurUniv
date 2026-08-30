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

## 2M++ galaxy-density source

Downloaded 2026-08-30 from the official CDS/VizieR catalog
**J/MNRAS/416/2840**. The source publication is Lavaux & Hudson (2011),
MNRAS 416, 2840–2856, DOI
[`10.1111/j.1365-2966.2011.19233.x`](https://doi.org/10.1111/j.1365-2966.2011.19233.x).
The VizieR dataset DOI is
[`10.26093/cds/vizier.74162840`](https://doi.org/10.26093/cds/vizier.74162840).

The observed local CSV files were obtained from
`https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync` with `MAXREC=100000`
using unordered queries:

```sql
SELECT * FROM "J/MNRAS/416/2840/catalog"
SELECT * FROM "J/MNRAS/416/2840/group"
```

Their observed local raw SHA256 values below are byte hashes of those specific
TAP responses and therefore depend on response serialization and row order.
They are provenance observations, not the reproducibility binding. The
reproducibility contract is the order-invariant canonical recno-sorted CSV/LF
SHA256.

The reproducible downloader `scripts/download_2mpp_v1.sh` instead issues:

```sql
SELECT * FROM "J/MNRAS/416/2840/catalog" ORDER BY recno
SELECT * FROM "J/MNRAS/416/2840/group" ORDER BY recno
```

The ReadMe endpoint was
`https://cdsarc.cds.unistra.fr/ftp/J/MNRAS/416/2840/ReadMe`.

- `2mpp_catalog.csv` — 72,973 rows; observed local raw SHA256
  `05d39f49af58caa7aa199420cc7354b3aa9fe3dbacf8d6c33222479c288fb23d`;
  order-invariant canonical recno-sorted CSV/LF SHA256
  `e761a9973f92e74520b81d36c5c7e76f739e47e2279da0567cdb2a92cf9d02ce`.
- `2mpp_groups.csv` — 4,002 rows and 4,002 unique group IDs; observed local
  raw SHA256
  `e83bcad0ebe97f6048f36b3235f66babaadf995fc2257302313160f35980057a`;
  order-invariant canonical recno-sorted CSV/LF SHA256
  `c2959d7fbda188ae2496ce76743a9243b6fb260099550025db988d1df381f6fa`.
- `2mpp_ReadMe.txt` — SHA256
  `0a4206da2c0e9997ff508909434816dfa095595d23eb6be4bee85fb8aee6c2d9`.

The ReadMe declares 69,160 real galaxies and 3,813 synthetic Zone-of-Avoidance
galaxies. The local validator finds exactly 1,840 rows with `Cln=1`. Rows with
normalized `Ref.strip()='zoa'` and cloned-redshift rows are imputed/latent components and are
forbidden as independent observations. Catalog-to-group linkage has one known
source anomaly: catalog `GID=5000` occurs in exactly three rows, while the
group table has no `GID=5000`; no other orphan group ID is accepted.

The raw galaxy catalog is not the published 4 Mpc/h 2M++ density map. A usable
`D_galaxy_density` likelihood still requires an explicit angular/radial
selection function, redshift-space-distortion treatment, luminosity-dependent
bias, model discrepancy, and uncertainty propagation. It must also crossmatch
against CF4 galaxies/groups and use a joint or explicitly de-duplicated
likelihood with cross-covariance so shared galaxies and redshift information
are not counted twice.

`src/cf4_2mpp_validate.py` validates the exact schemas and semantic counts and
binds downloads to the order-invariant canonical hashes. The reproducible
downloader is `scripts/download_2mpp_v1.sh`; it validates a temporary download
before atomically installing the directory and refuses to overwrite a target.
