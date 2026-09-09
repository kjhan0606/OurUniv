# Fable5 plan audit and driver disposition

2026-09-09. User authorizes bundle entry and requests Fable5 planning audits,
explicit Q-GOAL/Q-LEAN, Astra backup on inability to obtain an audit.

## Invocation and result

`claude --safe-mode --model claude-fable-5 --effort high --print --tools ''
--strict-mcp-config --no-session-persistence --output-format json`, with the
project-scoped request and full redesign on stdin;600s timeout. No tools,
external filesystem inspection or numerical calculation by the auditor.

Completed normally, exit0/is_error=false; reported duration126215ms. Requested
model confirmed by response modelUsage `claude-fable-5` (a Haiku auxiliary
entry also appears; substantive audit usage is Fable5). No fallback invoked.
Original request/result:
`config/cf4_structure_lg_fable5_plan_audit_v1.txt`
`config/cf4_structure_lg_fable5_plan_audit_v1.response.json`.

Verdict: CONDITIONAL GO.

- Q-GOAL: field-objective work can help but must not remain another TNG-looking
  image exercise; native readout alone leaves the observation connection absent.
- Q-LEAN: existing checks, two draws, three checkpoints and4h envelope broadly
  proportionate. Minimize fixed patch features, make full rollout scoring less
  frequent, and defer decoder/model-family/q_S expansion.
- Accept matched1000-update experiment as bounded evidence; require baseline,
  fixed gradient-noise stop and explicit outcome branches instead of an endless
  further-training response to ambiguous results.
- Proposed extra: Gaussian proxy conditional based on a field-only readout
  g(F) and native residual covariance, to enable a later LG-conditioned trial.
  Auditor explicitly identified this as an additional approval item.

## Driver accepts

1. Keep Q-GOAL and Q-LEAN as two questions in ONE plan audit, not new code gates.
2. Preserve normalized likelihood, mixed atoms and conservative decoding. At a
   fixed trace, current support depends on native/generated parent moments and
   mandatory mask ancestry, not a learned parameter-dependent truncation:
   inspect `condition`, `ConditionalSplitFlow.masks`, and `Coupling` in
   `src/cf4_conditional_split_flow.py`. Softmax changes probabilities, not
   structural support; bounded affine scales do not truncate Gaussian support.
   Existing fixed coordinate normalization is a checkpoint buffer. This supports
   score-function use under regularity/integrability conditions, not its SNR.
3. Add only a past-sample moving baseline, held fixed for the current pair,
   if the approved score-function branch is implemented. A baseline computed
   from the same pair must not be silently substituted as an unbiased one.
4. Predeclare a finite noise/time stop in the existing in-job feasibility
   segment; no repeated new batches until an attractive estimate appears.
   CPU controls still run through Slurm: 'offline CPU' is not permission to
   perform numerical tests on syntax. Use the already allocated CPUs.
5. One fixed minimal spatial transform; full rollout at most every8 updates.
   Preserve the native NLL objective on all existing scales.
6. Outcomes: repair wins AND meets morphology criteria -> development-only;
   control wins -> no evidence for the structural addition; both fail/no clear
   improvement -> hold fine-model adoption and move to observation-connection
   design, NOT another implicit extension. No automatic extra1000/6000 steps.

## Driver does NOT accept as written

The proposed `qhat_S = N(g(F), Sigma_cal)` is not yet executable. The audit's
definition of g requires S_MW/S_M31/S_M33 component moments, whereas a newly
generated F supplies only their total. The existing native component catalogue
cannot be copied onto it. Covariance assembly does not solve this circularity.
Even with a field-only g, a Gaussian distribution on mass/component moments
does not enforce positivity, disjointness, binding or exact total conservation.
A transformed Gaussian on OBSERVATIONAL PROXIES could be a labelled approximate
likelihood ingredient; it is not the missing physical component law q_S.

Likewise, dropping ancestor score terms to 'train only the finest parameters'
is not automatically valid: the present six-scale model SHARES parameters.
That would not be the gradient of the stated rollout objective. A separately
defined native-parent finest-only objective is possible, but is a changed
experiment, not a silent in-job fallback. Do not implement the suggested
fallback under the original unbiased-gradient claim.

Finally, no observed-LG posterior becomes acceptable regardless of q_F quality.
A labelled mock/diagnostic observation-link trial can be useful; a failed field
prior remains failed. The existing12-grid posterior must not be reused as an
independent prior with repeated CF4/count likelihoods.

## Execution disposition

No Fable tool/model failure occurred; invoking Astra to overturn the conditional
verdict would not be the requested backup policy. Retain the substantive
feedback and the driver corrections above.

Original bundle entry is approved, but its Q-GOAL condition exposes a missing
scientific operator. Do not launch both expensive branches while pretending
that operator is solved. Ask the user to prioritize a FIELD-ONLY LG readout/
proxy-likelihood design and native calibration before the paired training.
This would change the approved bundle's order/scope, so it is not assumed.
No new numerical job or fitted Gaussian proxy was submitted. No additional
generic gate framework, audit ladder or GPFS work is proposed.
