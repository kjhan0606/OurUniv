# R2/5 — live count + FP + disjoint TF current-field connection

The TF source bridge 406479 supplies8,502 corrected, disjoint TF-only CF4
groups and one numerical field factor. This bundle checks the next real
dependency: all galaxy counts, FP marks and TF marks must read ONE evolved
LCDM IC/current density and velocity state before any joint inference.

Use the predeclared `initial_chain0` N128/384 white IC and existing FP
hyper/group coordinates, not a TF-selected state. Evolve it with the pinned
PMWD cosmology, score57,238 eligible 2M++ counts once, existing conditional
FP source-group marks once, and the disjoint TF-only group summaries once.
The TF term uses the *existing* shared relative TF zero point also used by
the12 TF anchors inside FP source groups. Each TF group redshift enters only
inside its numerator and denominator, not as an independent velocity factor.
Counts retain integrated selection and uncertain Gamma rates. This is a
partial target with transferred count bias/FoG reference, provisional TF
`b=1`, group width150 km/s and missing group-inclusion covariance; none is
estimated by this one-state calculation.

Compute the four factors, positive count support, full IC reverse gradient
and an independent TF IC directional finite difference. A TF-source/anchor
CF4-group overlap fails closed. No posterior chain, parameter fit, new
simulation, map promotion, or N256 is authorized by this numerical control.
One typed-H100 Slurm allocation,4CPU/22GiB/30min. The old joint control
peaked~5GiB and the standalone TF factor~1.3GiB; reverse-mode and both
factors together may grow toward18GiB, so22GiB provides >20% headroom.

Q-GOAL: this is the first same-field connection of both major CF4 distance
methods with the density survey. It directly advances the actual R2
observation target but does not close its survey/covariance or sampler gaps.
Q-LEAN: one saved IC, existing source products, one analytic-vs-finite
difference check, no frozen-state parameter sweep or HMC extension.

MW/M31/M33 remain R3 latent roles drawn from each NEW evolved state; this
N128 field does not resolve them. The future normalized role law must keep
MW/M31 ambiguity and missing/shared M33, and their observed quantities must
constrain that same state. Neither a CF4 group ID nor a native truth identity
can seed/select a generated-field candidate.

## Execution and decision

Typed-H100 Slurm **406482 COMPLETED/exit0 in3m47s**. The two focused TF
tests passed. Its pinned result is
`/gpfs/kjhan/CF4/z0_density/r2_live_tf_joint_control_v1/result.json`.
At the predeclared IC, the four log factors (arbitrary fixed mark references)
are count -274191.36, FP +1371.25, TF +29293.98 and white prior
-1051525.00. The 57,238 observed galaxy counts have positive unit support
(occupied minimum 2.869e-5). The 6,745 TF training groups are evaluated;
1,757 heldout groups contribute zero to this training score. The TF-specific
IC directional derivative agrees with a central finite difference to
relative 6.41e-8. The process peak was5.51GiB (Slurm batch MaxRSS~5.42GiB)
under the22GiB request.

This passes the **one-current-state wiring and reverse derivative** only.
Absolute score levels are not model evidence or goodness of fit. TF group
inclusion, method/source covariance, velocity dispersion and the count
selection/bias response remain uncalibrated; neither an equilibrated sampler
nor a heldout posterior predictive check ran. **NO-GO for R2 posterior and
N256 promotion.** The next bundle must calibrate or bound those observation
dependencies and establish an equilibrating joint sampler, not simply extend
the old short chains.
