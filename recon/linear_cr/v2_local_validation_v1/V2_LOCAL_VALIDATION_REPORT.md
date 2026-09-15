# V2 local-likelihood N64 validation

- Verdict: **DO NOT PROMOTE TO N192**
- Fresh split seed: `20260809`
- CR seeds: `71--78`
- Hybrid rows: 19,226 WF15 + 252 local-direct
- Local held-out rows: 51

## Declared tuning split

| local sigma_nl [km/s] | local z mean | local z std | cov68 | cov95 |
|---:|---:|---:|---:|---:|
| 50 | `-0.683` | `1.809` | `0.392` | `0.686` |
| 100 | `-0.474` | `1.015` | `0.647` | `0.922` |
| 150 | `-0.433` | `0.722` | `0.784` | `0.980` |

The frozen selection rule chose `100 km/s` by minimizing
`abs(local z_std - 1)`. Split `20260805` was then declared spent, and the
following gates use only the new split.

## Fresh validation gates

| metric | result | acceptance | gate |
|---|---:|---:|:---:|
| adjoint relative error | `1.579e-6` | `<5e-5` | PASS |
| maximum CG relative residual | `3.756e-5` | `<1e-4` | PASS |
| global z mean | `+0.011` | `abs <0.1` | PASS |
| global z std | `1.027` | `0.85--1.15` | PASS |
| global 68% coverage | `0.708` | `0.62--0.75` | PASS |
| global 95% coverage | `0.945` | `0.92--0.98` | PASS |
| local z mean | `-0.158` | `abs <0.2` | PASS |
| local z std | `1.303` | `0.8--1.2` | **FAIL** |
| local 68% coverage | `0.784` | `0.55--0.8` | PASS |
| local 95% coverage | `0.922` | `0.88--1.0` | PASS |

The preregistered promotion rule requires every numerical, global, and local
gate to pass. No N192 hybrid parent will be generated from this model.

## What was learned

The direct nearby likelihood removes the central cavity very strongly. For a
Gaussian `R=5 Mpc/h` linear-density probe, the original matched-seed control
had sample mean `-0.941`, the first hybrid had `+0.059`, and this independent
validation has:

- posterior mean: `+0.403`
- CR sample mean/std: `+0.397 +/- 0.091`

Thus local CF4 rows carry the missing observer-environment information, but the
simple Gaussian direct-velocity error model is not stable enough across splits.
The next model should implement a validated lognormal-bias treatment such as
the CF4 Bias Gaussianization correction, or a hierarchical likelihood in
distance modulus, instead of further selecting `sigma_nl` on these validation
rows.

The CF4 large-scale reconstruction paper applies BGc before WF/CR and validates
it on mocks: <https://arxiv.org/abs/2311.01340>.
