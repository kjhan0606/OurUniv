# Driver disposition — Fable5 position-link advice

2026-09-12. Actual verdict PROCEED, full response preserved in
config/cf4_position_link_fable5_v1.response.json. Q-GOAL/Q-LEAN accepted as
one observation-space bridge, not a posterior or another location repair.

Adopt log-domain integration, explicit units/selection approximations,
separate MW/host/incremental-M33 contributions, all seven parent-moment
conservation and two finite-difference step sizes. The native Qdiag is the
EXTENSIVE sum(m*v_axis^2), not a positional second moment; its permutation
conservation follows directly and will also be checked numerically.

Improve the proposed quadrature instead of adding a Monte Carlo layer:
split each sky ray at ALL crossed cell faces. Within the resulting distance-
modulus rectangles q is constant. With k=ln(10)/5 and a=3k(1,1), exponential
tilting makes the Jacobian-weighted Gaussian exactly another Gaussian:
E[exp(a.u) 1_rectangle] = exp(a.d + a.C.a/2)
 * Pr[N(d+C.a,C) in rectangle].
Compute each rectangle probability by deterministic one-dimensional adaptive
integration of a standard-normal density times a conditional-normal CDF
difference. Clip only at12 marginal standard deviations, report the explicit
absolute tail bound <=4 Phi(-12) before the Jacobian/q scaling, and refuse
numerical certainty if that bound can materially affect the likelihood.
Use a tighter integration tolerance as the numerical comparison, not a
science-selected order. No GH aliasing at voxel faces, random quadrature,
new data, trained coefficient, or inferred measurement broadening.

Gradient finite differences use eps=.05 INTERIOR along each conservative
convex-mixture direction, with two small centered steps. At eps=0 cold/empty
cells can be nondifferentiable; do not falsely certify a boundary derivative
from one forward step. Native multiplicative-moment sensitivity uses the
explicit cold/empty zero-derivative convention and is labelled as sensitivity.

Clarify two adviser claims rather than changing the accepted model:
- q is strictly positive on all in-cube cells because the accepted density
  reference is 1+M/mean, a spatial probability reference NOT added physical
  mass. Empty cells therefore do not imply zero q. Outside-cube observation
  support is zero and must not be renormalized away; report it explicitly.
- Reusing a data-conditioned mark posterior as a likelihood double-counts
  observed LG data, not merely reuse of the same calibration archive.

Report exact native triple counts before scoring. If the bounded computation
cannot complete, preserve it as incomplete; do NOT silently subsample after
seeing costs/scores. Separate W Cartesian density, A marginal observation
log-likelihood, and T conditional incremental log-likelihood (a correlated
integral does not factor into three independent average log-scores).
Distance modulus is interpreted as the development z=0 physical-distance
proxy, converted kpc*h/1000 to cMpc/h; no cosmological lightcone correction or
stellar-to-halo offset calibration is claimed. Eight real-data field scores
are interface diagnostics, NOT candidate selection or posterior weights.

Driver proceeds under autonomous approval. No additional review stage. The
missing physically calibrated joint field/mass/COM law remains the next main
science bottleneck; this bridge is not permission to paint a density map.
