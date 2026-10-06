# OurUniv / CF4

Read `/home/kjhan/.codex/SHARED_CONTEXT.md` fully at run start, then read
`CF4_MASTER_PLAN.md` fully. The latter is the active project-level science,
priority and execution plan, approved on 2026-09-07. It supersedes historical
route/plan files; direct newer user instructions still take precedence.
User2026-09-13 explicitly ratifies the latent-IC joint present-state/history
route in CF4_END_TO_END_REPLAN_20260913.md and R1 execution. Read that plan
for the new R1–R5 order; old independent-z0-first implementation restrictions
are historical. The actual z=0 posterior remains the first science delivery.

Confirm cwd/repository before actions. Syntax is a Slurm login server.
GPFS is ordinary shared storage, not a subject for filesystem diagnostics.
Do not run pgrep/process-scan monitoring loops. Preserve unrelated work.

Current audit policy (user update2026-09-13): external review ONLY for an
important discovery, a goal revision, or a large calculation. Routine plans,
code checks, diagnostics and evaluations are handled by the driver. This
supersedes the prior every-bundle external planning-review requirement.
When external review is warranted, Fable5 primary and Astra backup if Fable
cannot complete a usable audit. Ask Q-GOAL (alignment
with the CF4/LG reconstruction goal) and Q-LEAN (proportionate implementation,
instrumentation and gates) explicitly.
Do not bypass an adverse substantive verdict by treating it as tool failure.
Every bundle plan and audit request must explicitly address MW/M31/M33
identification from the available NEW field/state, role ambiguity and unresolved
members (especially M33), and how their observables constrain that same field.
Native truth identities may label calibration/evaluation, never seed or select
generated-field candidates. Do not assume known components inside g(F).

User clarification2026-09-12: plan audits are advice, not automatic authority.
The driver independently checks their evidence (including existing verified
code/results) and decides which recommendations to adopt, amend or reject,
recording concrete reasons. A substantive disagreement is not an invocation
failure.

User update2026-10-06: during R2, Astra performs every warranted external
audit, including mid-course reviews and the exit review. This supersedes the
Fable-primary route for R2 only. The trigger is unchanged: an important
discovery, a goal revision, or a large calculation. Routine plans, code
checks, diagnostics and evaluations stay with the driver. Ask Q-GOAL and
Q-LEAN, including MW/M31 role ambiguity, unresolved M33, and the same NEW
evolved field. A mid-course audit is advice: check its evidence, adopt the
supported part, record any amendment, and continue R2. It does not close R2
or send the exit email. Closing R2 still requires a separate Astra exit audit.
Send the R2-exit email to kjhan0606@gmail.com only after Astra explicitly
approves the exit and the driver has checked that the cited evidence matches
the repository. Put Astra's verdict, findings, and the driver disposition in
the email body. Do not start R3 in that turn, and do not start it before the
user replies. If Astra withholds exit approval, adopt that advice and continue
R2; do not send the exit email and do not enter R3. An unusable Astra response
is neither approval nor a substantive rejection: retry Astra, and do not
substitute another model. Detail: `CF4_R2_EXIT_AUDIT_GATE_20261006.md`.

User update2026-09-12 after population-location comparison347085: autonomous
continuation is approved within the CF4/LG goal. Do not stop at each new bundle
to request repetitive approval. Driver may plan, implement, submit via Slurm,
evaluate and commit/push coherent work; retain bounded experiments, advisory
plan review, substantive result reports, preservation and scientific limits.
This supersedes prior per-bundle approval waits, not the science route or
resource/safety constraints. Seek input only for genuinely new authority,
material goal changes or an impasse that cannot be resolved in scope.

User confirmation2026-09-13: no approval procedure is needed when there is
no major issue; continue work rather than stopping at routine boundaries.
External-review triggers above and scientific/resource limits still apply.

User update2026-09-26: for the time being, submit GPU calculations using the
H200/H100/A100 Slurm modes. The node GRES types are `gpu:H200:1` on `h200`,
`gpu:H100:1` on `h100`, and `gpu:A100:1` on `a100`; H200 must include the
literal `H200` type. A single typed GRES request cannot represent all three
types, so check the three modes and submit one compatible job, recording the
chosen mode. Do not use manual node execution.

User update2026-10-05: when a GPU job is ready to submit, check H200 first;
if the H200 resource is idle, prefer the `h200` partition with its typed GRES
(for one GPU: `--gres=gpu:H200:1`). If H200 is occupied, check the permitted
H100/A100 modes and submit one compatible job. Record the selected partition
and typed GRES; never run manually on a node.
