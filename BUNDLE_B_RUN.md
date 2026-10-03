# Bundle B execution

User approved 2026-09-07. Scope and stopping rules: BUNDLE_B_DESIGN.md.
Three deliverables: actual environment diagnosis, explicit LG observation
contract, two bounded mock dynamics bridges. Bundle C is not approved.

Environment334398 completed: Virgo/Coma and secondary clusters show positive
coarse aperture means; Bootes negative. Local Void probes are uncertain or
positive, especially at24 cMpc/h. This is not a calibrated environment pass.

Bridge334399 failed after1m09s at its first gradient (before any optimizer
step). Unchanged-forward regression had passed. An initial hypothesis was an
acc=None versus array tree mismatch. A constant initial acc array did NOT
resolve the error: retry334400 failed after1m06s at the same first gradient.
Do not present that hypothesis as the established cause. A trace-only CPU
diagnostic now exposes JAX's underlying message, which PMWD's constructor
otherwise masks by attempting to cast type-description strings to arrays.
Total GPU use so far2m15s; no optimizer evaluations completed. No installed
library, physics, weights or seeds changed. Preserve both failed directories.

LG source table and callable identity/frame/covariance interface are implemented.
Partial covariance explicitly requires opt-in; production mass/environment
priors are disabled pending resolved operators and astrophysical calibration.
Tests are submitted with the trace diagnosis, not on the login node.

CPU334401 passed all4 focused tests and PM gradient tracing. Source inspection
established that JAX custom_derivatives.py:862 reconstructs string-valued type
trees BEFORE checking whether outputs agree; PMWD tries to convert those
strings to numbers. Thus the string constructor itself is the established
failure, not evidence of a hidden physical/type mismatch. The process-local
adapter extends PMWD's placeholder guard to all-string metadata trees only.
No numerical array conversion or installed library files are changed.
Retry2 must pass unchanged-forward and finite-difference checks again;
time limit3h57m plus prior2m15s stays below4 GPU-hours.

## Calculation record and available products

- GPU retry334402: completed4m58s from e4834d4, 1 GPU/4 CPUs/12000 MiB,
  partitions a40,a100,h100,h200, exclude syn06; Slurm time limit3h57m.
  Output `/gpfs/kjhan/CF4/z0_density/bundle_b_v1/bridge_retry2`.
  Check this fixed job ID and its scalar logs/results. No scan/monitor loop.
- Environment334398: completed29s. Products at
  `/gpfs/kjhan/CF4/z0_density/bundle_b_v1/environment/`:
  `environment.png`, `structures.csv`, `apertures.npz`, `result.json`.
- LG contract: LG_OBSERVATION_CONTRACT.md and
  config/cf4_lg_observation_contract_v1.json. Four CPU tests passed334401
  (15s including trace check). The interface is not actual LG conditioning.

At R12 cMpc/h, Virgo delta mean0.585 (95% interval0.422..0.765),
Coma0.648 (0.516..0.783), Bootes-0.315 (-0.540..-0.054).
All six secondary cluster apertures have positive model-conditional signs.
Local Void Lacerta/Andromeda/Aquila R12 intervals cross zero; UrsaMinor R12
is positive (0.123..0.498). Larger apertures mix surrounding material:
three Local Void R24 probes have positive intervals and Lacerta is uncertain.
This does not establish the absence of small voids in the real universe.
Virgo/Coma native maximum offsets within the fixed R24 searches are typically
18/22 cMpc/h, so positive aperture mass is NOT precise cluster localization.
Bootes R31 coverage is97%; missing coverage is not observed zero density.
Gauss aperture volume quadrature differs from analytic sphere volume by up
to about5% for R12; windows are normalized and uncertainty is from draws.
Interpret borderline small-aperture signs at this numerical/coarse-field limit.

## Closure judgement

Both targets passed the fixed development gate at the200-evaluation cap:

| Target | Density RMS start -> final | Velocity RMS start -> final (km/s) | Final density/velocity correlation |
| --- | --- | --- | --- |
| 0 | 0.964423 ->0.117889 | 332.732 ->17.766 | 0.98472 /0.99715 |
| 5 | 0.971773 ->0.118383 | 300.068 ->18.115 | 0.98490 /0.99656 |

Both final density fields are positive/unit mean, and conservative mass and
momentum readout errors were zero at reported precision. Directional adjoint
relative errors1.50e-6/2.23e-5 pass the2% engineering tolerance. Each final
forward run was recomputed from the fitted white field. Optimization hit its
cap; this is not a claim of a unique or fully converged MAP solution.
GPU time including failed starts:7m13s, below4h. Successful job MaxRSS
3300384 KiB; host memory request12000 MiB included the required20% margin.

Saved candidates retain LCDM transfer/cosmology but are not posterior draws.
Initial/final prior penalties and linear P(k) are saved without rescaling.
The white second moments are about0.896/0.903; some resolved k bins have
P(k) about30% lower than the fixed initial realization (not an ensemble/theory
comparison). Do not hide this with amplitude boosting. Optimization does not preserve
an unconditional Gaussian realization distribution. This is not high-k phase
recovery or statistical certification of LCDM IC samples. Actual z=0 draws
from the empirical field model still need a correctly defined dynamical
joint target/proposal correction; multiplying CF4 likelihood again is invalid.

Summary/visualization job334405 completed5s, reading saved outputs only. Products:
`/gpfs/kjhan/CF4/z0_density/bundle_b_v1/bridge.png` and `summary.json`.

**B is closed at development/diagnostic level.** The bridge supports continuing
the z=0-first route; no evidence here promotes the actual coarse posterior or
recovers the LG. Stop N32 sampling/bridge extensions. Next substantive work is
the LG-conditioned multiresolution z=0 field, retaining coarse mass/momentum,
not another direct-CF4 IC search. Its resolved object operator, shared-data
covariance and numerical discrepancy must be explicit. Surroundings1–2 and
LG<=0.3 cMpc/h remain unachieved; zoom force/particle resolution is separate.
**No Bundle C calculations submitted; wait for the user's next-bundle approval.**
All calculations use Slurm. No new training ensemble or full RAMSES outputs.
