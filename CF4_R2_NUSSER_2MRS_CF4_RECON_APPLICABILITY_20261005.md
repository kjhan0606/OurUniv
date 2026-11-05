# R2 literature applicability: 2MRS Bayesian reconstruction tested with CF4 — 2026-10-05

## Decision

Adi Nusser's 2026 reconstruction is a close methodological comparator, not a
replacement or completed solution for OurUniv. It reconstructs a local field
from 2MRS and tests predicted velocities against CF4; CF4 distances are not
part of that posterior. The author explicitly leaves a joint 2MRS+CF4
likelihood for future work because CF4 distance-indicator errors and
inhomogeneous Malmquist bias require careful treatment. Therefore the paper
does not close the active R2 conditional-mark, selection, or shared-group
uncertainty gaps.

## What transfers as a comparison

- **A useful point-process construction:** the observed 2MRS positions enter
  an inhomogeneous Poisson likelihood with its expected-count integral. The
  radial selection is evaluated at each model particle's real-space radius
  before redshift-space deposition, explicitly to avoid coupling selection
  gradients to peculiar-velocity shifts (the Kaiser-rocket effect).
- **A separate dynamical approximation:** the field is evolved with the
  Zel'dovich approximation, and the effective nonlinear galaxy bias is
  defined on a coarser grid. This is not the active OurUniv PM/RAMSES
  likelihood and its bias prescription is not a calibrated substitute for
  the six 2M++ populations.
- **A useful validation pattern after a valid posterior exists:** compare
  posterior-predicted velocities to untouched CF4 data at observed
  redshift-space positions, with conditional-mean, density–velocity, and
  shell-dipole summaries. In OurUniv, only graph-closed heldout CF4 outcomes
  can serve that role; results from CF4 marks used in the fit are not an
  independent validation.
- **A high-k distinction:** the paper samples a 128^3 field in a
  300 h^-1 Mpc box (2.34 h^-1 Mpc cells), then refines one posterior draw to
  256^3 by adding Gaussian small-scale modes conditional on preserving each
  coarse-cell average. Those added modes are prior draws, not information
  inferred from 2MRS. The reported 1.15 h^-1 Mpc particle spacing is not the
  resolution of the inferred field.

## Limits relevant to OurUniv

1. **Different conditioning direction:** 2MRS constrains their posterior;
   CF4 is an external flow test. Our first science delivery remains a
   CF4-conditioned z=0 density/velocity posterior from one latent LCDM IC.
2. **Insufficient local resolution:** the inferred field is 2.34 cMpc/h per
   cell, far coarser than the 1.5 cMpc/h active target and the requested
   `<=0.3 cMpc/h` LG region. The 256^3 refinement adds unconstrained prior
   modes rather than observed LG information.
3. **Local-galaxy warning:** the paper reports that its realizations can have
   a mean density within a few h^-1 Mpc of the observer about 2–3 times
   nearby-galaxy estimates. It identifies equal weighting of all 2MRS
   galaxies, despite the nearby environment being dominated by low-mass
   galaxies, as one possible cause. This is a caution against treating a
   flux-limited galaxy count as matter without a calibrated population/bias
   model; it does not provide a correction for 2M++.
4. **LG/structure coverage:** Virgo and Coma are shown qualitatively, but a
   detailed Coma/Virgo/Local-Group comparison is explicitly outside the
   paper's scope. It does not identify MW, M31, or M33 components.
5. **No independent-data shortcut:** Our 2M++ likelihood and this 2MRS-based
   product are not independent datasets by construction. Its reported CF4
   predictive agreement cannot be treated as a heldout test for a model that
   conditions on the same CF4 outcomes.

## Driver assessment and next use

**Q-GOAL:** partially aligned. It demonstrates that a coarse, explicit
redshift-space galaxy point-process model can reconstruct large-scale flow
and predicts external peculiar velocities, but it does not make our desired
CF4-conditioned map, resolve the Local Group at `<=0.3 cMpc/h`, or establish
MW/M31/M33 identities. Those remain latent roles on the same NEW evolved
field, with MW/M31 ambiguity and unresolved M33; native truth identities may
be used only for calibration/evaluation.

**Q-LEAN:** literature review only. Reuse the paper's real-space selection
evaluation and untouched-velocity validation as comparison ideas; do not
rewrite the active likelihood, import its MAP/IC as a candidate, add a second
2MRS factor, or run another gravity/sampler job from this result. R2 remains
NO-GO pending an identified conditional observation law, a stationary
posterior with uncertainty, untouched heldout prediction, and the actual
z=0 map.

### Source

- A. Nusser, *Bayesian Reconstruction of the Local Universe from 2MRS:
  Testing the Gravitational Flow with Cosmicflows-4*, arXiv:2606.08593v2
  (2026), accepted for ApJ: <https://arxiv.org/abs/2606.08593>.
  Relevant sections: §IV (likelihood and forward model), §§V–VI (MDPL2 test,
  128^3 posterior and conditional 256^3 refinement), and §§VII–VIII (CF4
  external comparison and limitations).
