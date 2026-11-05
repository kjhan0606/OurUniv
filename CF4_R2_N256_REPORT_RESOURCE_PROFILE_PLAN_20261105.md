# R2 N256 four-chain report resource profile — 2026-11-05

## Question and bounded scope

Measure peak RSS and end-to-end elapsed/read-write time for the existing
four-chain field-MCSE report at its native `N=256` array shapes. This is a
synthetic resource profile only. It must not read or score held-out outcomes,
load CF4 target states, run an R2 chain, make a field map, or promote a
posterior.

The profile will generate four fresh, job-specific chain fixture directories
using the report's current sufficient-statistics and selected-map schemas.
The statistics arrays have the native `N=256`, two-scale layout, 512 retained
states, and complete batches at sizes 32/64. Their deterministic synthetic
moments will be algebraically consistent, spatially varying, predominantly
occupied, and have positive density/velocity variance. They are not sampled
posterior chains or CF4 data. `np.savez` will preserve the current uncompressed
archive behavior. The selected map archive contains only the density,
velocity, and occupancy members the report actually reads. Fixture generation
must finish in a separate process before the real report CLI starts. The
surviving resource JSON must prominently identify all inputs/results as
synthetic and non-promoting.

## Resource estimate and guard

- Four statistics archives: `4 x 9.25 GiB = 37.0 GiB` raw array bytes.
- Selected-map members: approximately `1.30 GiB` per chain, or `5.2 GiB`
  total; expected temporary fixture storage is about `42.2 GiB`.
- During sequential merge, the expected report peak is dominated by two live
  chain accumulators, `18.5 GiB` raw. Allow up to `2.5 GiB` additional Python,
  array-decoding, and bookkeeping overhead, for a conservative `21 GiB`
  expected peak. Request `28 GiB` (about 33% headroom, exceeding the required
  20%). Verify this estimate against the merge/finalize implementation before
  submission; increase the request if that review finds a larger expected
  peak.
- The report reads all four statistics archives, but only the selected map
  members from the smallest batch scale: ten float64 scalar-grid equivalents
  plus three boolean grids, `1.296875 GiB` per chain. Total raw fixture arrays
  are therefore about `42.1875 GiB`. Immediate reads follow fixture writes,
  so observed report I/O may be page-cache warm; record generation/write time
  separately and do not claim cold-storage throughput.
- One CPU on Syntax `a10`, 2-hour wall cap, CPU-only/no GRES. Fail closed on
  memory/time exhaustion; no automatic retry or larger follow-up.
- Before submission, check available space on the exact GPFS output mount and
  refuse to run unless at least `55 GiB` is free for the temporary fixtures,
  report, logs, and margin. GPFS remains ordinary shared storage; this is a
  capacity check only, not a filesystem diagnostic. Repeat the check at job
  start to account for queue delay.
- Create under a new path keyed by `SLURM_JOB_ID`; never overwrite. Preserve
  the JSON result and logs. Validate the four-chain diagnostic status, both
  pooled scales, retained-state counts, finite/defined regional summaries,
  and selected-member CRC checks before cleanup. After those checks succeed,
  remove only this job's manifest-listed synthetic fixture subdirectory and
  record the removed byte count. On failure, preserve all artifacts for
  diagnosis. A measured report-step peak above `23.333 GiB` means the 28-GiB
  allocation did not demonstrate 20% measured-peak headroom; do not claim the
  memory gate passed or auto-retry with a larger request.

## Interpretation and gates

Report Slurm `MaxRSS`, elapsed time, job/node/partition, source commit,
statistics/map member byte counts, report exit status, and whether the
job-specific fixtures were cleaned. A successful profile establishes only the
resource behavior of this report code path on synthetic arrays; it does not
certify production sampler memory, I/O, MCSE, convergence, or field validity.
No N256 target, posterior, held-out score, or GPU calculation is authorized by
this profile.

**Q-GOAL:** this is a necessary resource prerequisite for evaluating the
requested CF4+galaxy-conditioned `z=0` density/velocity posterior at
`1.5 cMpc/h`; it produces none of that posterior, its uncertainty, or a
held-out prediction. MW/M31 remain role-ambiguous and M33 unresolved. Their
observables must ultimately constrain those same roles on the same NEW evolved
LG field at `<=0.3 cMpc/h`; native truth identities remain calibration and
evaluation labels only.

**Q-LEAN:** one direct N256 report profile using existing schemas; no new
simulation, target evaluation, sampler screen, source search, or diagnostic
ladder. The approximately `42.2 GiB` synthetic artifacts are temporary and
job-scoped; only the small resource report and logs remain after success.

R2 remains NO-GO: the source-matched CF4/Tempel/SDSS-PV mark-availability and
group-selection law is unresolved; spatial MCSE calibration and the target
sampler's N256 cost/convergence contract are also unfinished. No held-out
outcome may be opened during this profile.

## Astra advice and driver disposition — 2026-11-05

Astra's read-only advisory verdict was **CONDITIONAL**. The driver independently
checked the cited implementation. The 18.5-GiB floor is two 9.25-GiB
accumulators during sequential chain merge; restoration adopts saved arrays and
merge is in-place. Finalization has a separate peak (about 9.25 GiB pooled
accumulator plus up to 8.375 GiB of scalar-grid float64 temporaries and masks),
not 18.5 plus that workspace. The 21-GiB planning envelope remains plausible;
28 GiB is retained, with actual report-step RSS required to be at most
23.333 GiB to demonstrate 20% measured headroom.

Adopted clarifications: nondegenerate internally consistent spatial fixtures;
generation and report in separate Slurm steps; distinguish fixture-write time
from report-read time and flag cache-warm uncertainty; explicitly label the
surviving result synthetic; recheck capacity at job start; clean only the
exact fixture directory after validating both scales, counts, and report
integrity. No job was launched by the audit. This advisory does not authorize
or qualify an R2 posterior.

## Completed profile — 2026-11-05

Syntax job 450056 completed successfully on `a10` / `syn04` with one CPU,
28 GiB requested, and wall time 7:09. The report step (`450056.2`) took 122 s
and had MaxRSS 19,552,148 KiB = 18.64 GiB. This is below the predeclared
23.333 GiB measured-peak ceiling, so the 28 GiB allocation exceeds the
measured report-step peak by more than 20%. The separate fixture-generation
step (`450056.1`) took 298 s and peaked at 17,299,644 KiB (16.50 GiB).

The four synthetic statistics/map inputs totaled 45,298,569,814 bytes
(42.19 GiB). CRC verification passed for all four statistics and selected-map
archives; the pooled report had four chains, 2,048 retained synthetic states,
64 complete batch-32 batches and 32 complete batch-64 batches, with both
scales and all required regional summaries defined. The job validated these
conditions, removed only its manifest-listed `fixtures/` directory, and left
the 41 KiB `resource_profile.json` and Slurm logs. The job-start free-space
guard measured 157,980,439,871,488 bytes on `/gpfs/kjhan/CF4`, above the
55 GiB minimum.

Authoritative resource result:
`/gpfs/kjhan/CF4/z0_density/r2_n256_report_profile_20261105/450056/resource_profile.json`;
Slurm stdout/stderr:
`/gpfs/kjhan/CF4/logs/cf4_r2_n256_profile_450056.out` and `.err`.
The executed source commit was
`f8816f15c1d34fe3528a64dbbd4b3635030c8453`.

Interpretation is limited to CPU memory and elapsed time of this report code
path on synthetic N256-shaped inputs. Fixture reads followed writes
immediately, so the observed 122 s is not cold-storage throughput. The run
does not measure production sampler RSS, validate the MCSE precision gate,
establish posterior convergence, or support a field/posterior claim. No
held-out outcome or N256 target state was read; `R2_complete=false` and
`posterior_promoted=false` remain in the saved profile.

Q-GOAL: this removes one report-resource uncertainty on the path to the
CF4+galaxy-conditioned 1.5 cMpc/h z=0 posterior, but delivers none of its
field, uncertainty, or held-out prediction. MW/M31 remain role-ambiguous and
M33 unresolved; their observables must constrain those same roles on the same
NEW evolved LG field at `<=0.3 cMpc/h`, with truth IDs only for
calibration/evaluation.

Q-LEAN: one bounded resource run completed, with exact job-scoped cleanup.
Do not repeat it or infer production chain memory/I/O from it. The selection
law, target sampler qualification, and calibrated joint field-precision gate
remain open; R2 remains NO-GO.
