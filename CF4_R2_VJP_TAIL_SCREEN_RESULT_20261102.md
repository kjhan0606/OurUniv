# R2 intensity-VJP tail screen result — 2026-11-02

**Disposition: numerical repeat/decomposition screen passed; sketch work may be
considered, but this is not a sampler or posterior result. R2 remains NO-GO.**

Job 441842 completed on Syntax with `h100`, typed `gpu:H100:1`, exit 0,
18:33 Slurm elapsed, and 6.42 GiB host peak under the 20 GiB request. The
application reported 1,076.70 s. Source commit was
`406b6a97a8ca2e60e9567fc4f202543bec10b997`. The machine-readable record is
`/gpfs/kjhan/CF4/z0_density/r2_intensity_vjp_tail_screen_20261030/441842/result.json`.

The fixed conditional-v6 anchor and N256/384 cMpc/h count operator were used
with training rows only: 47,121 count occurrences, 37,951 occupied keys,
11,009,805 exposed population-cells, of which 1,993,850 had positive
intensity. The lowest-intensity 1% of positive exposed rows was 19,939 rows;
no intensity floor was added and no row was removed from the target. The
minimum positive intensity remained `1.9562372e-37`, with maximum
`1/sqrt(lambda)=2.2609411e18`.

| Probe | Coordinate | Full norm | Tail norm | Rest norm | Tail squared-norm share | Doubled cross term |
|---|---|---:|---:|---:|---:|---:|
| seed 20261029 | white IC | 957.57319 | 5.7561e-4 | 957.57314 | 3.6133e-13 | 0.08382 |
| seed 20261029 | tracer | 230.06283 | 2.1984e-5 | 230.06281 | 9.1308e-15 | 0.00607 |
| seed 20261030 | white IC | 965.38590 | 5.1771e-4 | 965.38596 | 2.8758e-13 | -0.11555 |
| seed 20261030 | tracer | 148.15844 | 2.3411e-5 | 148.15843 | 2.4969e-14 | 0.00203 |

All four full = tail + rest decomposition checks passed the predeclared
`1e-8` relative tolerance; observed relative errors were `1.2e-15` to
`8.7e-15`. Repeating the seed-20261029 full pullback gave relative errors
`7.21e-15` for white IC and zero for tracer. Peak device memory was
41.13 decimal GB (38.31 GiB); the device-reported limit was 69.82 GiB.

**Interpretation.** For these two Gaussian probes at this one anchor, the
lowest-intensity 1% contributes negligibly to the measured VJP norms despite
the very large inverse-square-root weight. This resolves the specific
arithmetic-tail concern for these probes; it does not prove the tail is
negligible for all probes, estimate a trace/rank or reliable spectrum, or
validate the likelihood score. The result supports only considering a
separately budgeted training-only discovery/independent-validation sketch,
subject to its own predeclared uncertainty, residual-information, cost and
stability gates. It does not authorize a sampler by itself.

Held-out outcomes loaded: 0. Held-out scores: 0. Full-target sampler run: no.
Posterior sample, uncertainty, field and R2 completion: none. The prior
Gaussian-reference design's remaining requirements still apply, including
the separate 15-dimensional raw-FP curvature `F_p`, fixed-reference
construction, unchanged full-target correction, and useful movement criteria.

**Q-GOAL:** This is a bounded feasibility result on the route to the first
conditional-v6 z=0 density/velocity posterior; it does not supply that
posterior. MW/M31 remain role-ambiguous and M33 unresolved. Future observables
must constrain those roles on the same NEW evolved LG field at `<=0.3
cMpc/h`; native truth identities are calibration/evaluation labels only.

**Q-LEAN:** The single seven-pullback screen was proportionate. Do not add a
tail threshold/seed ladder, read held-out values, or launch a gravity
simulation. A separate sketch calculation needs an explicit fixed budget and
falsifiable validation criteria before submission.

## Independent advisory and driver disposition

Astra's read-only review returned **CONDITIONAL GO** for one capped
discovery/independent-validation geometry experiment and **NO-GO** for a
sampler launch from this result alone. Astra flagged and the driver corrected
the device-memory unit: the recorded peak is41.13 decimal GB, or38.31 GiB.
The driver adopts Astra's requirement to gate the **absolute** residual
count-information, not only its fractional trace share, with a predeclared
minimum useful ellipse angle. The concrete thresholds and resource plan are
recorded in `CF4_R2_VJP_GEOMETRY_SKETCH_PLAN_20261102.md`. Astra emphasized
that passing these geometry screens is neither a posterior nor an R2 exit
approval.
