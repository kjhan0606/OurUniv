# R2 retained HMC block-jump diagnostic — bounded plan (2026-11-02)

## Question and disposition

Record whether two unchanged exact-target HMC transitions from the saved,
discarded state60 actually move the fixed Fourier blocks that were absent
from the historical transition trace. This is the smallest follow-up that
closes the specific evidence gap identified in
`CF4_R2_MALA_HMC_AUGMENTATION_RECONCILIATION_20261102.md`; it does not choose
or augment a sampler.

Use the existing exact conditional-v6 target, prior-split HMC kernel, saved
state60 coordinates/gradient/RNG, fixed metric, saved step size, and
state-independent trajectory-length mixture `{8,12}` without retuning. Run
at most transitions60 and61, then stop. Retained displacement is the
post-Metropolis `q_new-q_old`, so an HMC rejection must record exact zero in
every block. Record retained RMS per real coordinate in `k^2=1` (6 modes),
the measured drift shell `16<k^2<=64` (1,852 modes at N256), the remaining IC
coordinates, and all24 canonical nuisance coordinates. For each transition,
verify that actual exact-target record growth equals the kernel's
`force_evaluations` count.

The predeclared hard ceiling is25 exact value+gradient target evaluations:
one checkpoint replay (cached by the common runner) plus at most24 HMC force
evaluations from two length-12 transitions. The application cap is4 hours;
Slurm wall time is4h30m. Request24GiB host RAM, rounded above the measured
18.7825GiB process peak plus20% headroom (22.54GiB). Require allocated GPU
memory of at least39,826,427,463 bytes, 20% above the prior measured exact
GL2 peak. No retry or automatic extension if a budget guard stops before two
transitions. For allocation selection, an H200 typed device counts as idle
only when its node is `IDLE`; a spare GPU on a `MIXED` H200 node is recorded as
available but occupied, and selection falls through to a free H100 then A100.

## Interpretation limits

Both the starting state and any resulting trace remain discarded transport
states, not posterior draws. This records movement and call cost only; it
does not estimate stationarity, ESS, mixing efficiency, uncertainty, heldout
prediction, calibration, or a density/velocity field. No heldout values are
loaded and no separate gravity simulation is run. R2 remains open/NO-GO and
R3 does not start.

**Q-GOAL:** per-transition block movement is one necessary sampler diagnostic
on the path to the conditional-v6 z=0 density/velocity posterior, not the
posterior itself. MW/M31 remain role-ambiguous and M33 unresolved; their
observables still must constrain those same roles on the same NEW evolved LG
field at `<=0.3 cMpc/h`. Native truth identities remain calibration/evaluation
only, never generated-field candidate seeds or selectors.

**Q-LEAN:** one instrumentation helper, focused unit tests, and at most two
already-defined exact HMC transitions answer the missing retained-shell
movement question. No pCN, MALA augmentation, scale/metric ladder, extra
diagnostic chain, heldout read, field promotion, or new simulation.

Implementation: `scripts/cf4_r2_hmc_transport.py` and
`tests/test_cf4_r2_hmc_transport.py`. Slurm wrapper:
`scripts/run_cf4_r2_hmc_retained_block_jumps.sbatch`.
