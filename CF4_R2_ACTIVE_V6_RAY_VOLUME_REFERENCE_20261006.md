# Active-v6 local ray-volume reference — 2026-10-06

Status: **H200 calculation complete; local sensitivity found, but no global
operator promotion. R2 remains NO-GO.**

Slurm job 414452 ran on `h200` with `--gres=gpu:H200:1` (syn104), 4 CPUs,
32 GiB requested memory and a 45-minute limit. It completed in 12:56 with
MaxRSS 3,343,148 KiB. The flat-to-grid exposure-mask contract regression was
fixed before this run. Tests passed: 7 angular-geometry, 5 exposure-mask and
15 shell-CDF tests. The failed predecessor 414437 used the same requested H200
GRES but stopped at that shape mismatch; its output remains preserved.

No external audit verdict was obtained: the Fable CLI returned its usage-limit
message, and the Astra Codex CLI rejected that model as unsupported for the
active ChatGPT account. The disposition below is the driver's evidence-based
review, not an external PASS.

## Scope

The calculation compares three expected-intensity operators on the same
conditional field and twelve score-blind, geometry-selected N256 source cells:
active GL2 with cell-constant completeness; ray-volume refinement with the
parent completeness map held constant; and the same ray refinement with native
NSIDE512 angular maps. Ray weights are unrenormalized
`dOmega r^2 dr / Vcell`. One NSIDE512-to-1024 check and one radial order
2-to-4 check were made on the predeclared `map_boundary_55_95` control.
No observed counts, held-out values, FP scores, optimizer steps or PMWD
evolutions were used. The diagnostic does not produce a posterior or map.

## Results

For the twelve selected patches, refined-parent versus active-GL2 expected
subtotals differ by `+0.012%, -0.010%, +0.017%, -0.050%, -0.209%, +0.034%`
for populations 0–5. Replacing the parent angular map with native NSIDE512
changes those selected-patch subtotals by `+11.42%, +5.06%, +1.96%, +3.44%,
+2.12%, -0.77%`. These are geometry-selected subtotals, not full-field rates
or likelihood changes.

The support-aware key check found one positive training-key intensity in the
predeclared reference source cell: population 1, key 3,088,467 at voxel
`[60,64,83]`. Its active-GL2, refined-parent and native-map intensities are
`3.20268e-7`, `3.18103e-7` and `4.96632e-7`; the native map is 56.1% above the
refined-parent value. The other five populations had no positive supported
training key in that source cell. No observed count was used to choose the key.

At the reference control, radial order 2-to-4 changed nonzero expected
outputs by at most `1.7e-9` relative, so the radial integral is stable there.
Angular subdivision from NSIDE512 to 1024 changed the two nonzero parent-map
outputs by 0.22% and native-map outputs by 0.44%. Its unrenormalized volume
closure error improved from `-0.259%` to `-0.0405%`. Across all twelve
NSIDE512 controls, however, the volume closure errors span `-0.960%` to
`+1.334%`; only the reference cell received the higher-NSIDE check.

## Driver disposition and next work

The tested reference cell shows a substantial angular-map sensitivity relative
to its single subdivision check, especially for the one supported population-1
key. This is a real local operator warning, not evidence of a global CF4
field effect. The ray-volume comparison is **not certified across all twelve
controls**: finite angular volume closure varies by about one percent and was
not refined on every control, while five of six populations have no supported
reference key. Therefore do not change the production count operator, rerun the
conditional field fit, score held-out data, or call the map-aware arm a
validated global reference from this result.

Do not launch a full-grid or higher-NSIDE ladder. The next R2 work should
return to the unresolved calibrated observation/count law and conditional
field evidence; any new geometric estimator must first have a bounded,
predeclared closure criterion and preserve the unrenormalized physical volume
measure. MW/M31 roles remain ambiguous and M33 unresolved. Their observables
must eventually constrain those same roles on the same NEW evolved LG field
at `<=0.3 cMpc/h`; native truth identities remain calibration/evaluation-only.

Q-GOAL: the diagnostic is relevant only to one expected-count operator in the
same z=0 inference and neither creates nor validates the N256/1.5 cMpc/h map.
Q-LEAN: one twelve-cell comparison plus one subdivision and one radial-order
check; no full-grid exposure, heldout score, fit, gravity replay or IC output.

Artifacts: `/gpfs/kjhan/CF4/z0_density/r2_v6_angular_ray_reference_20261006_h200_v7/`
and Slurm logs `/gpfs/kjhan/CF4/logs/cf4_R2_angray_414452.{out,err}`.
