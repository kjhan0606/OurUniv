# Bundle B execution

User approved 2026-09-07. Scope and stopping rules: BUNDLE_B_DESIGN.md.
Three deliverables: actual environment diagnosis, explicit LG observation
contract, two bounded mock dynamics bridges. Bundle C is not approved.

Environment334398 completed: Virgo/Coma and secondary clusters show positive
coarse aperture means; Bootes negative. Local Void probes are uncertain or
positive, especially at24 cMpc/h. This is not a calibrated environment pass.

Bridge334399 failed after1m09s at its first gradient (before any optimizer
step). Unchanged-forward regression had passed. PMWD LPT supplied acc=None
but its custom reverse returned an acc array; JAX's error formatting also
hit PMWD's constructor, hiding the original tree mismatch. The local adapter
now initializes a constant zero acc array, overwritten by PMWD's initial
force as before. No installed library, physics, weights or seeds changed.
Retry must recheck forward equivalence and finite differences, use a fresh
bridge_retry1 directory, and request at most3h58m to keep total below4h.

Implementation of the LG contract is in progress.
All calculations use Slurm. No new training ensemble or full RAMSES outputs.
