# R2/5 — public SDSS FP randoms and the CF4 radial-selection boundary

The actual CF4+galaxy-conditioned z=0 posterior remains **NO-GO**. This
source-only bundle checks the meaning of an available radial-selection input;
it does not fit a field, infer a true-distance group-inclusion law or read FP
distance marks for prediction. No email was sent.

The [SDSS PV source paper](https://academic.oup.com/mnras/article/515/1/953/6611706)
states that its random redshifts were sampled from a smoothed spline of the
**observed** SDSS PV redshift distribution, with the published angular mask.
The same paper's individual FP log-distance-ratio PDF uses a flat eta prior
and a distance-dependent `f_n` normalizer for the FP measurement selection.
These are different objects: the random `n(z_obs)` is not the true-distance
selection probability for a CF4/Tempel group, nor a second correction to add
to the source FP PDF.

Typed-H100 Slurm **406602 COMPLETED/exit0** in 5m08s, streaming the official
[v1.1 random catalogue](https://zenodo.org/records/6824749) without retaining
its 284 MB raw file. All 284,000,071 bytes pass the published MD5
`8627b4063e8a71572e333e0ac65d6657`. There are 4,000,000 random rows;
3,704 lie just above z=0.1 (maximum 0.1000692) and are explicitly outside
the fixed [0,0.1] profile. The public v1.1 FP data hash also passes; its
final catalogue has 34,059 rows and **33,121** `in_mask` rows. The paper's
33,618 mask count refers to an earlier stage before the final outlier cut.
The preserved [small result JSON](/gpfs/kjhan/CF4/z0_density/r2_sdss_random_radial_v1/result.json)
contains all 20 bins and source hashes. Batch MaxRSS was only 67,096 KiB,
not an estimate of total network/file-system cache use.

Dividing random counts by the comoving shell volume gives a *relative
observed-redshift-space* selected number-density shape. Normalized to the
z=[0.030,0.035) shell, it is 0.614 in [0.050,0.055) and 0.239 in
[0.095,0.100). The CF4-eligible FP **sky-training** rows inside the source
mask number 8,708. Their fraction of the full in-mask SDSS PV catalogue is
0.892 in [0.030,0.035), 0.0460 in [0.060,0.065), and 0.000374 in
[0.065,0.070). This sharp edge is not evidence of a physical group-selection
function: the frozen source bridge explicitly requires the **observed group
redshift distance** to be below 180 cMpc/h, about z=0.061. The sky-training
role also differs from the full-footprint random sample.

For a mark likelihood conditioned on a group's measured redshift z, a
selection factor depending **only** on that fixed z cancels between its
distance-integral numerator and denominator. It must not be evaluated at a
candidate true distance and multiplied into `d²ρ` as if it were a recovered
`S_group(d_true)`. A genuine selection dependence on latent distance, FP
observables or group/CF4 association need not cancel. The source `f_n`, the
known observed-redshift eligibility cut, and any such residual group-inclusion
factor must be owned consistently in one model. The existing linked 2M++
count/FP redshift dependence also remains unresolved. The published randoms
were themselves built using the full FP sample; neither their profile nor the
full-source redshift histogram is a prospective independent FP sky-holdout
test. The source file's full lines were streamed, but no heldout FP eta/distance
column was parsed or used in a profile or field score. The result flag
`heldout_FP_marks_read=false` has this *statistical-use* meaning, not a claim
that the raw text bytes were absent from the stream.

The first run406599 stopped after four seconds on 3,704 near-boundary random
redshifts; the first corrected stream406600 completed the remote read but
failed on the driver's mistaken pre-outlier mask count. Neither produced a
science result. The final source-matched correction406602 above is the only
accepted artifact. The abandoned idea of multiplying the random `n(z)` into
the true-distance prior was **not** run or adopted.

Driver decision: retain randoms as an observed-sample intensity/geometry
reference, not a calibrated CF4 selected-group radial prior. The next R2
model must distinguish (i) selected FP measurement likelihood already carrying
`f_n`, (ii) group/CF4 occurrence and crossmatch selection, and (iii) one
joint ownership of linked 2M++ count-point and FP redshift/mark data. The
official selected mocks can test individual true-distance/velocity behavior,
but without pre-selection parent and recovered Tempel memberships they cannot
alone certify the full group law. Any candidate is tested against the frozen
v5 sky roles without promoting the old nonstationary sampler or N256.

Q-GOAL: prevents an apparently convenient but incorrect radial-prior repair
on the path to the actual present-field posterior. Q-LEAN: one streamed public
catalogue and small summary, no full mock archive, gravity run, fit, new gate
framework or external audit. MW/M31/M33 still require latent identification
from each **new** evolved field; MW/M31 ambiguity and unresolved M33 remain,
and their observables must constrain that same state. No native/mock truth
identity selects generated candidates.
