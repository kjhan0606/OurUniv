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
All calculations use Slurm. No new training ensemble or full RAMSES outputs.
