# Astra mid-course review — rejected-shell direction result (2026-11-02)

Verdict: accept the diagnostic and its narrow interpretation. Keep the
`epsilon=0.05` shell candidate closed. This is mid-course advice, not an R2
exit audit or R3 approval.

The plan, implementation, updated master plan, result note and authoritative
JSON agree:

- Five target evaluations, within the declared budget.
- Reconstructed endpoint energies match job447590 exactly; the sole component
  difference is count-log-likelihood roundoff of `2.91e-11`.
- The shell Hastings terms reproduce:
  `-120.72849 + 9.74809 = -110.98040`.
- AD/FD discrepancy `9.11e-7` passes the predeclared1% criterion.
- The finite displacement changes `U` by `+120.72849` despite its local
  linear prediction `-289.97660`.

This supports local directional consistency and substantial departure from
the initial linear approximation along this particular move. It neither
validates the entire gradient nor identifies smooth curvature versus
accumulated support/CIC effects. The reported `0.2205` endpoint-gradient
change is for the **full gradient**, not specifically its1,852-coordinate
shell projection; label it accordingly. No substantive result inconsistency
was found.

The smallest defensible next action is a design-only reconciliation with the
existing exact-GL2 prior-split HMC route—not another target evaluation. Specify
one possible HMC augmentation using the preserved fixed fundamental-MALA
component, while retaining full-target complementary and nuisance transport.
Use existing traces to define comparable retained movement in the correct
`16<k^2<=64` shell and complete evaluation costs. Do not infer benefit from
fundamental acceptance alone or compare it with unmatched whole-IC HMC
displacement. If existing evidence cannot justify that augmentation, report
the missing evidence rather than launching it. This diagnostic does not
justify a new learned metric, Hessian sketch, scale ladder, or automatic
long-chain extension.

Q-GOAL: yes, this resolves one sampler-engineering ambiguity on the route to
the first conditional-v6 z=0 density/velocity posterior. It supplies no
posterior uncertainty, equilibration evidence or heldout prediction.
Observation-model selection/group-calibration limitations remain separate
from sampler correctness.

MW/M31 remain role-ambiguous and M33 unresolved. Their eventual observables
must constrain those same roles identified from the **same NEW evolved LG
field at `<=0.3 cMpc/h`**; native truth identities may serve
calibration/evaluation only. This N256/384 field samples1.5 cMpc/h and does
not establish that LG capability.

Q-LEAN: the completed five-call diagnostic was proportionate and is now
finished. Reuse the existing code/results for the next bounded design; no
repeated FD, extra directions, heldout reads, new simulation, additional gate
framework or R3 work. R2 remains open/NO-GO.
