# CF4 statistically valid constrained-realization gate

**Overall: PASS**

This gate validates a linear-Gaussian posterior ensemble. It does not yet select
the member that best reproduces the MW–M31–M33 system or the named clusters/voids;
that is the next, pre-registered posterior-predictive selection stage.

## Frozen model

- Test manifest: `manifest_final_test_n192.json`
- All-data ensemble: `manifest_parent_v1_all.json`
- Grid: N=192, L=384.0 Mpc/h
- Cosmology: Om=0.31, Ob=0.05, h=0.746, As(1e9)=1.63, ns=0.96
- CF4 constraints: 19265 grouped distances; WF15 Gaussian velocity estimator
- Error model: scale=0.9, sigma_NL=0.0 km/s
- Ensemble: 16 exact Matheron draws, seeds 1001–1016

## Gate checks

| Check | Result | Value | Limit |
|---|---:|---:|---|
| `operator_adjoint` | PASS | `2.3269731515953198e-06` | <1e-4 |
| `cg_accuracy` | PASS | `3.793250652961433e-05` | <1e-4 |
| `heldout_mean` | PASS | `0.007738348395376464` | |mean z|<0.1 |
| `heldout_scale` | PASS | `1.0492431306885057` | 0.9<=std(z)<=1.1 |
| `heldout_coverage_68` | PASS | `0.7050116852765516` | 0.65..0.75 |
| `heldout_coverage_95` | PASS | `0.9444300181770968` | 0.93..0.97 |
| `heldout_information` | PASS | `815.0406444460386` | delta log score > 0 |
| `white_field_variance` | PASS | `[0.9996601416473206,1.000294649306119]` | all sample std in 0.995..1.005 |
| `white_field_gaussianity` | PASS | `{"max_abs_skew":0.0020033384917752146,"max_abs_excess_kurtosis":0.0033403858217191384}` | both < 0.01 |
| `lcdm_shell_power` | PASS | `{"mean_ratio":[1.262071288519584,1.2078688789034682,1.1598882366981627,1.05288528907708...` | |<Pwhite>-1| <= max(0.05,3/sqrt(Nmode)) |
| `training_residual` | PASS | `0.9985125064849854` | 0.8..1.2 |

## Held-out posterior predictive test

- N=3851; standardized residual mean/std = +0.0077/1.0492.
- 68/95% coverage = 0.7050/0.9444.
- Relative to noise-only, delta log predictive density = +815.0.

## LCDM power and phase

The field stored as `s_out` is the whitened primordial field. Its shell power
therefore should be unity; applying the frozen transfer function yields the target
LCDM P(k) by construction. The ensemble-mean shell ratios are:

| k [h/Mpc] | N(rFFT) | <Pwhite> | residual power | WF-mean power | coherence |
|---:|---:|---:|---:|---:|---:|
| 0.0205 | 13 | 1.2621 | 0.5131 | 0.7823 | 0.7770 |
| 0.0308 | 9 | 1.2079 | 0.5996 | 0.5813 | 0.7153 |
| 0.0429 | 53 | 1.1599 | 0.8861 | 0.2608 | 0.4861 |
| 0.0637 | 153 | 1.0529 | 0.9239 | 0.1214 | 0.3500 |
| 0.0937 | 458 | 1.0330 | 0.9807 | 0.0538 | 0.2250 |
| 0.1374 | 1411 | 1.0024 | 0.9751 | 0.0279 | 0.1648 |
| 0.2016 | 4377 | 1.0023 | 0.9903 | 0.0122 | 0.1096 |
| 0.2951 | 13458 | 1.0030 | 0.9971 | 0.0057 | 0.0768 |
| 0.4320 | 41902 | 0.9999 | 0.9979 | 0.0020 | 0.0448 |
| 0.6322 | 130442 | 1.0008 | 1.0002 | 0.0005 | 0.0235 |
| 0.9249 | 406529 | 1.0000 | 0.9999 | 0.0001 | 0.0105 |
| 1.3529 | 1268194 | 1.0003 | 1.0003 | 0.0000 | 0.0045 |

## Status

- The 16-member ensemble is accepted as the statistically valid parent-CR ensemble.
- Seed 1001 is the deterministic reference member only; it was not chosen using
  held-out data or Local-Group morphology.
- No member is yet the final physical parent. Named-structure and LG acceptance
  criteria must be frozen before forwarding and choosing among these 16 draws.
