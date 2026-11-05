# R2 terminal-force comparison (2026-09-30)

R1 → **R2 current-state posterior: ongoing, not promoted** → R3 same-field
MW/M31/M33 identification → R4 precise evolution → R5 zoom IC.

## Completed calculation

Slurm job 408858 completed on syn08/H100 in 5:34:12, exit 0, with 48 GiB
host-memory reservation and 27.8 GiB observed MaxRSS. It ran one common-random-
momentum trajectory per force at each predetermined A/B terminal state. The
GL2 fine target, step 0.08, 8 integration steps, metric and nuisance inverse
mass were unchanged. This is a local proposal comparison, not posterior
sampling, a PM production simulation, or a stationarity test.

| Endpoint | Proposal force | ΔH | MH acceptance probability | elapsed | acceptance-weighted IC RMS / second |
|---|---|---:|---:|---:|---:|
| A | old frozen GL1 affine | +19.603 | 3.07e-9 | 1,567 s | 1.02e-12 |
| A | exact GL2 | +0.734 | 0.480 | 7,087 s | 3.54e-5 |
| B | old frozen GL1 affine | +2.152 | 0.116 | 1,551 s | 3.91e-5 |
| B | exact GL2 | +0.597 | 0.551 | 7,084 s | 4.06e-5 |

The same momentum and acceptance uniform were used within each endpoint pair.
Both A proposals happened to reject (the exact-force acceptance probability
was nevertheless 0.480); both B proposals happened to accept. Exact GL2
forces reduce the tested Hamiltonian error at both endpoints, but take about
4.5 times longer. The per-cost movement proxy is dramatically better at A,
where the old affine proposal is effectively stuck, and only about 4% better
at B. One momentum at each of two states cannot establish average efficiency,
mixing or posterior convergence.

## Driver decision and next bounded test

Do not start another long chain using the unchanged old affine anchor, and do
not choose exact GL2 for every production force call based on this pair alone.
The next smallest useful test is one **new fixed affine tangent per terminal
state**: compute the GL2 gradient once at that saved state, correct the GL1
force there, freeze the anchor/correction for the whole one-trajectory test,
and retain the existing GL2 Metropolis target, metric, step, length and paired
momentum. Reuse the old chain's force cache only after checking its exact
state/target provenance; the GL2 gradients from 408858 were not persisted, so
this test must reevaluate them. Compare anchor setup cost separately and use
acceptance-weighted movement per total/amortized cost. No seed or step-size
search. If results are mixed or fail to reduce correction error economically,
close this local-affine repair rather than adding more diagnostic tiers.

This disposition follows Astra's backup advisory after Fable was unavailable;
the advice is not an automatic verdict. Q-GOAL: the test addresses affordable
exploration of the actual R2 z=0 field target. Q-LEAN: only two predetermined
states and one trajectory each; no new survey, cosmological production run,
or heldout score.

R2 remains incomplete: usable stationary posterior draws/MC error, GL4
numerical sensitivity, common-scale prior sensitivity, information-support
maps and the single untouched-v6 predictive check still require usable draws.
MW/M31 remain ambiguous and M33 unresolved. Later LG candidates and their
observables must come from and constrain the **same NEW field**; truth
identities may label calibration/evaluation only, never seed or select
generated components.
