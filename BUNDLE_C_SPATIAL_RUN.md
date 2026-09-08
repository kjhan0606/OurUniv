# C-spatial execution record

User authorized entry2026-09-08. This remains within master C; no D/IC/zoom
entry. Design: BUNDLE_C_SPATIAL_DESIGN.md.

Implemented first native spatial calibration job:16 training and16 heldout
MW-role selections from the unchanged satellite population, one companion
pair per observer, unique source members streamed once. All member gas/DM/
stars+wind/BH dynamical masses participate. Native profiles, total velocity
covariance and sparse fine moments link to exact total-matter remainders.
Native mass/COM, residual realizability, reconstruction and coarse restriction
are checked in the calculation using existing moment code, not a new generic
audit framework. No sphericalization, frame rotation or actual field inference.

Output `bundle_c_v1/spatial_calibration_v1`; resource request2 CPUs/4800 MiB/
30min, estimated4000 MiB peak+20%, <=50m source rows and<8 GiB outputs.
This job prepares the missing spatial calibration INPUT. It does not fit the
joint spatial probability law or deliver the CF4/LG-on/off posterior map.
