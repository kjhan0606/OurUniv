# R2 terminal-state re-anchoring disposition (2026-10-01)

R1 → **R2 current-state posterior: ongoing, not promoted** → R3 same-field
MW/M31/M33 identification → R4 precise evolution → R5 zoom IC.

## Job 409024

The bounded two-endpoint test completed on Slurm in 1:38:33, exit 0, with
48 GiB host memory reserved and 21,015,188 KiB (about 20.0 GiB) MaxRSS.
The four focused corrected-split-HMC tests passed in 13 s. The same N256 GL2
target, .08 step, eight split steps and fixed metric were kept. One fresh GL2
gradient at each predetermined chain endpoint defined a separate affine
tangent for its single proposal; this was never a production chain.

| Endpoint | Tangent-gradient residual at anchor | ΔH | Formal exp(min(0,-ΔH)) | Anchor + trajectory cost |
|---|---:|---:|---:|---:|
| A | 1.11e-16 max abs | -130.596 | 1.0 | 3,022 s |
| B | 1.39e-17 max abs | -130.827 | 1.0 | 2,825 s |

The GL2 tangent matches by construction at each start, but the proposed paths
have a large-magnitude fine-Hamiltonian error. The negative errors are not a
convergence signal; this job did not decompose the error into integrator and
surrogate-correction terms. More importantly, the anchor is a function of
the start state. These separate local trajectories do not define one
state-independent reversible proposal kernel, so the reported formal
acceptance values and acceptance-weighted movement are **not valid
posterior-transition evidence**. Do not use the proposed states as posterior
draws. This fails to establish an economical reusable affine force over the
tested path and closes this local re-anchoring repair; no further
affine-anchor ladder is planned.

## Driver disposition and next bounded test

The exact GL2 force was state-independent and valid, with ΔH +.734 (A) and
+.597 (B) for the eight-step trials, but cost about 7,085 s per trajectory.
The next proportionate sampler check is one exact GL2 **one-step** trajectory
per same predetermined endpoint, using the same momentum seeds, target,
metric and .08 step. This is a valid short exact-force proposal; compare
movement/cost with the saved eight-step references. It tests whether fewer
expensive force evaluations improve useful transport, not stationarity.
Bound the Slurm run to 4 h application / 4 h wall time, 48 GiB host RAM and
one H100. If movement/cost is poor, reassess sampler conditioning before any
long allocation; do not claim R2 completion.

Q-GOAL: still targets actual N256 z=0 posterior exploration. Q-LEAN: two
fixed starts, one short exact-force trajectory each, existing target and
diagnostics only; no new production cosmological simulation or survey.
MW/M31 remain ambiguous and M33 unresolved. Later LG candidates and their
observables must arise from and constrain the **same NEW field**; truth IDs
may evaluate/calibrate but never seed/select generated components.

Even a favorable one-step result cannot finish R2. Stationary posterior draws
and MC-error assessment, GL4 numerical sensitivity, common-scale prior
sensitivity, information-support maps and the single untouched-v6 predictive
check remain outstanding and require usable draws.
