# R2 fixed physical-mode ESS A/B screen — plan (2026-11-02)

## Purpose and decision

Run one capped, same-anchor comparison of (A) a broad all-IC conditional
elliptical-slice control and (B) fixed physical Fourier-mode moves. This is a
transport-feasibility screen prompted by the unstable learned rank-4 sketch;
it is not a retry of that sketch or of the closed million-dimensional
`|k|<=64` low-block pilot. It is not an HMC-backbone comparison or a claim
that ESS should replace HMC. The selected state is the retained, explicitly
discarded HMC transport anchor, never a posterior draw. No state or field is
promoted.

The result may justify designing a later multi-state assessment that augments
a full-coordinate, full-target-corrected transition only if the fixed-mode
arm completes within cost and produces predeclared movement. A failed
bracket, resource guard, or insufficient movement closes this one-sweep,
two-proposals-per-block candidate; it does not reject all physical-mode moves
or R2 feasibility. There is no extra sweep, mode-band ladder, or angle tuning.

## Frozen target and transitions

- Keep the conditional-v6 N256 target in the384 cMpc/h box:47,121 training
  count occurrences/37,951 keys,1,414 linked raw-FP marks, and all24
  standard-normal nuisance coordinates. Every scalar proposal scores the
  complete unchanged count+FP likelihood; the standard-normal prior remains
  the exact Gaussian reference. Rebuild/check support on every candidate.
- Use one fixed `direct_value_only` scalar-score backend for both arms. First
  replay the saved anchor through the matched value-and-gradient contract,
  then evaluate the anchor with the selected scalar backend; preserve the
  known one-ULP FP reduction difference as floating-point implementation
  detail, not as a changed likelihood.
- Arm A uses two fixed orthogonal blocks: all16,777,216 IC coordinates, then
  all24 nuisance coordinates.
- Arm B uses four fixed orthogonal blocks: fundamental `k^2=1` (6 real
  coordinates), `k^2=4..8` (66), their full IC complement (16,777,144), then
  all24 nuisance coordinates. These are Fourier modes of the periodic384-box,
  not a learned/data-selected basis.
- Each completed block update is standard-normal elliptical slice sampling.
  Every proposal uses the same full target score. A fixed composition of
  invariant conditional kernels preserves the target; no partial likelihood
  decides a move. Both arms start independently from the same anchor with
  fixed, separately recorded RNG seeds. A bracket that exhausts its cap
  aborts that arm as an incomplete diagnostic; it is not converted into a
  self-transition, and that arm is never continued from a partially accepted
  prefix or treated as posterior draws.

## Fixed caps and decision statistics

- Maximum proposal evaluations:8 per arm. Arm A budgets4 for IC and4 for
  nuisance; arm B budgets2 per block. Plus one matched anchor replay and one
  scalar-anchor score, maximum18 total target evaluations.
- One sweep per arm;3-hour application /3.5-hour Slurm cap. Request24GiB host
  memory and stop at20GiB measured process RSS. The20GiB planning maximum is
  the14.06GiB peak measured in the earlier same-target ESS pilot, plus up to
  1GiB for simultaneously live IC/auxiliary/proposal arrays and real-Fourier
  work buffers, plus a4.94GiB allocator/transient reserve. This is a
  conservative estimate, not a measured peak for this implementation; the
  24GiB request leaves20% above it. Before target initialization, query the
  allocated JAX device memory limit and require it to exceed1.2 times the
  previously measured43,799,579,648-byte device peak. The subsequent runner also applies the
  existing device-peak guard before proposals.
- Report every attempted exact evaluation, bracket contraction, block time,
  and accepted per-coordinate RMS displacement. For each arm, measure the
  final displacement from its anchor in the fixed fundamental and `k^2=4..8`
  subspaces, divided by the square root of each subspace dimension and by
  both total evaluations and that arm's elapsed seconds; report the shared
  projector-setup and full-job times separately. A bracket that exhausts its
  limit is an incomplete move with zero accepted displacement.
- The physical-mode arm is a viable candidate only if it completes all four
  blocks and each slow block's accepted RMS displacement is at least0.10
  prior-normalized units. The0.10 threshold is an operational spending rule,
  not a scientific or posterior-scale boundary; a posterior-constrained mode
  can move less. Compare its slow-band displacement/evaluation and
  displacement/second to arm A; report ratios without treating one anchor as
  a mixing or stationarity estimate. This is resource-matched, not a decisive
  algorithm comparison: A and B allocate their eight attempts differently,
  use separate random streams and share only one anchor. If either arm aborts,
  label the comparison incomplete and retain per-block diagnostics; an
  incomplete complement/nuisance move does not by itself show the two slow
  blocks failed. If B does not meet the absolute movement criterion, close
  this one-sweep candidate even if its ratio to A is large.

## Scope and limits

No held-out outcomes/scores, native truth identities, new simulation,
field/uncertainty output, HOP, or RAMSES work. No sampler posterior is
claimed. The observation model remains conditional on the frozen linked-mark
presence and is not full-survey selection calibrated.

## Astra audit and driver disposition

Astra's read-only verdict is **CONDITIONAL GO for this bounded engineering
screen**, not GO for an inference run. The driver adopts these clarifications:
an exhausted bracket aborts its arm; it is never silently treated as a
self-transition. A/B is a resource-matched one-anchor diagnostic, not a
decisive algorithm comparison and not a test of ESS replacing HMC. The0.10
movement rule is only a compute-spending threshold; lower movement does not
mean a physical mode is scientifically impossible. If either arm is
incomplete, the A/B comparison is incomplete, with per-block outcomes
retained. Failure to finish complement/nuisance transitions does not by
itself diagnose the two targeted slow modes.

The20GiB host planning maximum is based on the previous same-target full
evaluation, explicit simultaneously-live vector/Fourier work buffers, and a
separately stated allocator/transient reserve; it is an estimate, not a
measured bound. The24GiB request and20GiB guard are retained. A startup GPU
memory-capacity check is required before target evaluation, in
addition to the runner's reused measured-device-peak guard. A pass can support
only a later design inserting fixed physical moves into a full-coordinate,
full-target-corrected transition; it is not posterior evidence. No further
rank-4 geometry probes or `F_p` run are needed.

**Q-GOAL:** the screen tests whether fixed, physically interpretable slow
modes can move under the complete target on the route to the first conditional
z=0 posterior. It does not deliver R2. MW/M31 roles remain ambiguous and M33
unresolved; their observables must constrain those roles on the same NEW
evolved LG field at `<=0.3 cMpc/h`; truth identities are calibration/evaluation
labels only.

**Q-LEAN:** reuse the existing target and exact ESS primitive; one A/B pair,
fixed mode partition,8 proposal evaluations per arm and hard resource limits.
No learned sketch, `F_p` follow-up, rank/seed/threshold ladder, held-out read,
extra gravity simulation or field promotion. A pass authorizes only a
separately designed multi-state HMC-augmentation assessment, not posterior
inference or replacement of HMC.
