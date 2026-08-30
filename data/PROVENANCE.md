# CF4 data provenance

Downloaded 2026-07-07 from VizieR (CDS) via TAP, catalog **J/ApJ/944/94**
(Tully et al. 2023, "Cosmicflows-4", ApJ 944, 94).

- `cf4_groups.csv`   — 38053 rows, table `J/ApJ/944/94/groups` (grouped catalog:
  distances DMzp/Dist, peculiar velocities Vpec, SG coords); raw SHA256
  `bfdc0cfc0f172b48468e3a8fd05e87978c1ec68c341fb2d929fc1200f0123334`.
- `cf4_galaxies.csv` — 55877 rows, table `J/ApJ/944/94/table2` (individual
  galaxies); raw SHA256
  `28e7b8bd386f53716ed84cddd67a6f7602f98bc1394923a312f906555da7f709`.

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

## CF4–2M++ crossmatch

The driver published the deterministic crossmatch on 2026-08-30 using
`src/cf4_2mpp_crossmatch.py`. Matching uses bidirectional nearest neighbours on
the unit sphere. A row is marked `secure_joint_mark` only when it is an exact
mutual one-to-one nearest pair, its angular separation is at most 3.0 arcsec,
and `abs(CF4 Vcmb - 2M++ Vcmb)` is at most 300 km/s. Normalized 2M++
`Ref.strip() == 'zoa'` rows are excluded before matching; the `Cln` flag is
retained and its cloned radial redshift remains latent rather than an
independent datum.

- `data/cf4_2mpp_crossmatch_v1.csv` — canonical mapping sorted by integer CF4
  `recno`; SHA256
  `64e4f8a1a8a612a19788ac759062930991a8ffe52bfa203635845fa1ad7a83bf`.
  This is an ignored local data artifact, not a repository source file.
- `config/cf4_2mpp_crossmatch_v1_result.json` — COMPLETE result bound to the
  mapping hash above; raw SHA256
  `3e2e5841d62e9581c7437a28f07d6f5c3423b023749f99a678546d1d7d29752a`.

The COMPLETE result contains 55,877 CF4 rows and 69,160 eligible real 2M++
rows. Its mutually exclusive classifications are 16,584 secure joint marks,
188 coordinate/redshift conflicts, 5 nonreciprocal collisions, 273 extended
review candidates, and 38,827 unmatched rows. Secure matches cover 11,610
unique CF4 `1PGC` groups. Conflicts, collisions, and extended-review candidates
remain quarantined; neither automatic promotion nor manual truth editing is
allowed.

The ignored mapping is reproducible in a new ignored directory. Plain `mkdir`
is intentional: reproduction fails rather than reusing or overwriting an
existing directory or artifact.

```bash
mkdir data/cf4_2mpp_reproduction_v1
python src/cf4_2mpp_crossmatch.py \
  --cf4-galaxies data/cf4_galaxies.csv \
  --twompp-catalog data/2mpp_catalog.csv \
  --output data/cf4_2mpp_reproduction_v1/mapping.csv \
  --summary data/cf4_2mpp_reproduction_v1/result.json
sha256sum \
  data/cf4_2mpp_reproduction_v1/mapping.csv \
  data/cf4_2mpp_reproduction_v1/result.json
```

The printed mapping and result hashes can be compared respectively with the
canonical values
`64e4f8a1a8a612a19788ac759062930991a8ffe52bfa203635845fa1ad7a83bf` and
`3e2e5841d62e9581c7437a28f07d6f5c3423b023749f99a678546d1d7d29752a`.

Publication of the mapping closes only the catalog crossmatch artifact step.
The de-duplicated factorization, shared-redshift treatment, and remaining
likelihood blockers are authoritative in
`config/cf4_2mpp_joint_likelihood_v1.json`; the crossmatch does not authorize
joint-likelihood, KF-EXPAND, all-D mock, or production execution.
