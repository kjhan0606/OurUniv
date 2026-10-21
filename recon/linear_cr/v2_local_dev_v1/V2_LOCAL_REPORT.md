# V2 local-likelihood N64 development result

- Verdict: **DO_NOT_PROMOTE**
- Common WF15 holdout rows: 1318
- Common-WF log predictive change (hybrid-control): `+7.66`
- Hybrid local-direct holdout: `n=51`, `z=-0.422 +/- 0.570`, `cov68=0.843`, `cov95=1.000`

| model | z mean | z std | cov68 | cov95 | common-WF logp |
|---|---:|---:|---:|---:|---:|
| control | +0.032 | 1.016 | 0.725 | 0.950 | -11591.5 |
| hybrid | +0.024 | 1.011 | 0.722 | 0.951 | -11583.8 |

Matched-seed observer linear density, Gaussian R=5 Mpc/h:

- control: `[-0.8447650671005249, -0.8545865416526794, -1.194061279296875, -0.8715434670448303]`
- hybrid: `[0.11391127109527588, -0.020948410034179688, 0.2041919231414795, -0.06031873822212219]`
- hybrid-control: `[0.9586763381958008, 0.8336381316184998, 1.3982532024383545, 0.8112247288227081]`

The aggregate delta-log scores in the sampler manifests are not
compared because the two models contain different catalog rows.
The direct local component is over-dispersed and biased on its own
held-out rows, so this N64 model is not promoted to N192.
