# COSMOSIM MDPL2 access and scope follow-up — 2026-10-05

## Question

Can the physically based MDPL2/SAG/SAGE 2M++-like mock candidate supply the
bounded R2 tracer-bias/count validation contemplated in
`CF4_R2_2MPP_MOCK_APPLICABILITY_20261006.md`?

## Driver checks

- COSMOSIM TAP metadata queries work as SQL/ADQL. The z=0 `availhalos` row is
  `snapnum=125`, with 127,388,160 Rockstar halos and 101,557,320 FOF groups.
- The inspected SAG/SAGE columns include positions, peculiar velocities,
  stellar-component masses, and SDSS rest-frame magnitudes. Galacticus exposes
  SDSS-band luminosities. The inspected columns contain no apparent-K/2MASS
  values or documented 2M++ K assignment.
- A three-row SAG smoke query returned promptly. A synchronous full-table
  `redshift=0` filter timed out; TAP metadata marks `redshift` unindexed.
  COSMOSIM documents asynchronous queues for longer selections, but an async
  query would not provide a bounded matter cutout or the missing K-selection.
- The official COSMOSIM files page separately lists MDPL2 raw particle
  snapshots. Its z=0 `snapdir_130` manifest exposes 1,920 GADGET-1 shards whose
  listed sizes sum to 1,716.2 decimal GB. The 912-byte README says `aexp=1`,
  `z=0`, GADGET-1 format, and total size `1.7 Tb`. This is a direct file
  download, not a SQL query result. The files page requires a registered-user
  API token for download; none was requested or retrieved.
- The database redshift table identifies z=0 as `snapnum=125`; the raw file
  directory is numbered 130. Both state z=0, but correspondence was not
  established.
- The user supplied existing COSMOSIM homepage credentials and identified
  their scope. They were used only to retrieve the small README. No secret was
  written to project files, no API token was requested or retrieved, and no
  particle or galaxy catalogue data were downloaded.

## Independent review and driver disposition

Fable5 first returned **CONDITIONAL PASS** on the TAP-schema evidence and
correctly warned not to claim that MDPL2 has no matter truth anywhere. After
the raw-snapshot evidence, its addendum recommended closing the candidate for
R2 in practical terms: truth exists in principle, but the complete snapshot
is too large for this narrow validation; there is no documented spatial
cutout, the 2M++ apparent-K selection is not established in the inspected
SAM products, and the database/raw snapshot numbering is unreconciled.

The driver adopts that source disposition. The addendum's statement that
credentialed README access lacked user authority is rejected because the user
provided the existing COSMOSIM account information before this limited read.
That does not authorize the 1.7-TB particle download. No further bulk or
asynchronous catalogue query is justified for this candidate.

The decision is scoped: **matter truth exists in the complete MDPL2 snapshot,
but MDPL2 is not an eligible current R2 mock source under the disk budget and
selection requirements.** Reopen only with a documented bounded spatial
cutout (or an explicitly accepted storage plan), a reproducible K-selection
mapping, and verified correspondence between the z=0 galaxy and particle
outputs.

## Q-GOAL and Q-LEAN

**Q-GOAL:** MDPL2 is an unconstrained random realization; it cannot create
the CF4-conditioned present-day field or determine MW/M31 roles. M33 remains
unresolved. MW, M31, and M33 observables must constrain those same roles on the
same NEW evolved LG field at `<=0.3 cMpc/h`; native truth identities remain
calibration/evaluation-only and may not seed or select generated-field roles.
This mock contributes no z=0 posterior or field result. R2 remains NO-GO.

**Q-LEAN:** no full snapshot download, shard probing, large SQL extraction, or
sampler/gravity run. The 3-row query was only a connectivity check. Resume the
already-planned R2 observation/exposure work without another external mock
search unless new, bounded matter-truth and K-selection evidence appears.

## Sources

- [COSMOSIM MDPL2 metadata/table list](https://www.cosmosim.org/metadata/mdpl2/)
- [COSMOSIM files and particle snapshot listings](https://www.cosmosim.org/cms/files/)
- [COSMOSIM MDPL2 z=0 DataLink manifest](https://www.cosmosim.org/datalink/MDPL2_snapdir_130/)
- [COSMOSIM TAP tutorial and async-query guidance](https://www.cosmosim.org/cms/tech-docs/access-via-tap/)
- Fable5 read-only addendum audit, 2026-10-05, captured in the session; no credentials were included.
