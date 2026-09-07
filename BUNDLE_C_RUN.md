# Bundle C execution

User approved C entry after B closure. Design: BUNDLE_C_DESIGN.md.
Native-observation builder and conservative multiresolution parameterization
implemented. Initial CPU job:2 cores,3600 MiB (3000+20%),20min cap.
No GPU, new IC/PM/RAMSES run or fine posterior fit submitted at this point.

CPU334407 stopped before output generation: NumPy2 does not multiply an int8
population array by32768 without explicit promotion. Corrected to int64,
matching the existing fine-key construction. Preserve the first directory;
retry uses native_data_v2. No scientific selection or input rows changed.

The current task is C1–2; C3–5 remain. Do not call input preparation or
zero-detail cell refinement a reconstruction. Do not claim0.1875 information
resolution from the configured cell size. Continue within C after the native
inputs and geometry tests; D still needs separate user approval.
