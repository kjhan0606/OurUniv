"""Pure local functions for the Q4 seed and procedural-geometry preflight."""

from __future__ import annotations

import hashlib
import json
import struct

M = 192**3
N_PARENT = 192
BOX_SIZE_CMPCH = 384.0
CELL_SPACING_CMPCH = 2.0
CHUNK = 1024
HOST_CEILING = 64 * 1024**3
DEVICE_CEILING = 16 * 1024**3
SIGMA_TOTAL_BOUND_KM_S = 310.4834939252005
DISPLACEMENT_BOUND_CMPCH = 2.316206864681996
DEVELOPMENT_NAMESPACE = "cf4-q4-development-20260904"
HELDOUT_NAMESPACE = "cf4-q4-heldout-20260904"


class MemoryCeilingError(MemoryError):
    """Fail-closed memory breach carrying both limb labels for auditing."""

    def __init__(self, message: str, details: dict[str, object]) -> None:
        super().__init__(message)
        self.details = details


def seed_record(namespace: str, index: int) -> dict[str, object]:
    digest = hashlib.sha256(f"{namespace}:{index}".encode("utf-8")).hexdigest()
    return {"namespace": namespace, "index": index, "digest_sha256": digest, "seed_uint63": int(digest[:16], 16) & ((1 << 63) - 1)}


def canonical_sha256(payload: object) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def enforce_memory_ceiling(
    projected_bytes: int,
    *,
    host_ceiling_bytes: int = HOST_CEILING,
    device_ceiling_bytes: int = DEVICE_CEILING,
) -> dict[str, object]:
    """Fail closed before allocation if either persistent ceiling is exceeded."""

    if not isinstance(projected_bytes, int) or projected_bytes < 0:
        raise ValueError("projected_bytes must be a non-negative integer")
    host_status = "PASS_PROJECTED_BELOW_CEILING" if projected_bytes <= host_ceiling_bytes else "FAIL_PROJECTED_ABOVE_HOST_CEILING"
    device_status = "PASS_PROJECTED_BELOW_CEILING" if projected_bytes <= device_ceiling_bytes else "FAIL_PROJECTED_ABOVE_DEVICE_CEILING"
    details = {
        "projected_bytes": projected_bytes,
        "host_ceiling_bytes": host_ceiling_bytes,
        "device_ceiling_bytes": device_ceiling_bytes,
        "host_status": host_status,
        "device_status": device_status,
        "status": "PASS_PROJECTED_BELOW_CEILING",
    }
    if projected_bytes > host_ceiling_bytes or projected_bytes > device_ceiling_bytes:
        details["status"] = "FAIL_PROJECTED_ABOVE_MEMORY_CEILING"
        raise MemoryCeilingError("projected persistent memory exceeds host/device ceiling", details)
    return details


def build_seed_manifest() -> dict[str, object]:
    development = [seed_record(DEVELOPMENT_NAMESPACE, i) for i in range(192)]
    heldout = [seed_record(HELDOUT_NAMESPACE, i) for i in range(256)]
    digests = [item["digest_sha256"] for item in development + heldout]
    seeds = [item["seed_uint63"] for item in development + heldout]
    if len(set(digests)) != len(digests) or len(set(seeds)) != len(seeds):
        raise RuntimeError("seed collision")
    if set(item["digest_sha256"] for item in development) & set(item["digest_sha256"] for item in heldout):
        raise RuntimeError("development/heldout digest intersection")
    if set(item["seed_uint63"] for item in development) & set(item["seed_uint63"] for item in heldout):
        raise RuntimeError("development/heldout seed intersection")
    manifest: dict[str, object] = {
        "schema": "ouruniv-cf4-q4-seed-manifest-v1",
        "derivation": "SHA256(namespace+':'+index), first 8 bytes masked to 63 bits",
        "development": development,
        "heldout": heldout,
        "development_count": len(development),
        "heldout_count": len(heldout),
        "cross_namespace_disjoint": True,
        "manifest_hash_scope": "SHA256 of canonical sorted JSON payload with manifest_sha256 field omitted; the emitted field is not included in its own digest.",
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def geometry_preflight() -> dict[str, object]:
    metadata = {
        "N_parent": N_PARENT,
        "M": M,
        "box_size_cMpc_h": BOX_SIZE_CMPCH,
        "cell_spacing_cMpc_h": CELL_SPACING_CMPCH,
        "observer": [192.0, 192.0, 192.0],
        "sigma_total_bound_km_s": SIGMA_TOTAL_BOUND_KM_S,
        "displacement_bound_cMpc_h": DISPLACEMENT_BOUND_CMPCH,
        "generation_order": "for z in range(N): for y in range(N): for x in range(N)",
        "position_formula": "((x+0.5)*2, (y+0.5)*2, (z+0.5)*2) cMpc/h",
        "los_formula": "[1.0,0.0,0.0] for every procedural source",
        "displacement_scale_formula": "1.0 cMpc/h for every procedural source",
        "dtype": "float64",
    }
    metadata_hash = canonical_sha256(metadata)
    chunk_count = (M + CHUNK - 1) // CHUNK
    boundary_ids = [{"chunk": chunk, "first_id": chunk * CHUNK, "last_id": min(M, (chunk + 1) * CHUNK) - 1} for chunk in range(chunk_count)]
    if boundary_ids[0]["first_id"] != 0 or boundary_ids[-1]["last_id"] != M - 1:
        raise RuntimeError("streaming lattice boundary check failed")
    # Stream the complete integer lattice mapping without allocating positions,
    # keys or a dictionary.  The recovered id must equal the original id for
    # every one of the 7,077,888 entries.
    streamed_checks = 0
    key_hasher = hashlib.sha256()
    for first in range(0, M, CHUNK):
        last = min(M, first + CHUNK)
        for linear_id in range(first, last):
            z, rem = divmod(linear_id, N_PARENT * N_PARENT)
            y, x = divmod(rem, N_PARENT)
            recovered = (z * N_PARENT + y) * N_PARENT + x
            if recovered != linear_id:
                raise RuntimeError("procedural lattice injectivity check failed")
            # Exercise the actual Q1 composite-key payload without storing a
            # dictionary: position, LOS and scale are packed as little-endian
            # float64 values in the same field order used by the candidate key.
            key_hasher.update(struct.pack("<3d", (x + 0.5) * 2.0, (y + 0.5) * 2.0, (z + 0.5) * 2.0))
            key_hasher.update(struct.pack("<3d", 1.0, 0.0, 0.0))
            key_hasher.update(struct.pack("<d", 1.0))
            streamed_checks += 1
    float64 = 8
    persistent = {
        "positions_bytes": M * 3 * float64,
        "los_bytes": M * 3 * float64,
        "displacement_scales_bytes": M * float64,
        "population_masses_bytes": M * 6 * float64,
        "source_to_group_bytes": M * 8,
        "output_N192_bytes": 6 * N_PARENT**3 * float64,
    }
    persistent_total = sum(persistent.values())
    projected_dict = M * 192
    projected_total = persistent_total + projected_dict
    memory_gate = enforce_memory_ceiling(projected_total)
    return {
        "schema": "ouruniv-cf4-q4-procedural-geometry-preflight-v1",
        "metadata": metadata,
        "metadata_sha256": metadata_hash,
        "uniqueness_method": "analytic injectivity plus chunk-boundary streaming over integer lattice IDs",
        "unique_positions_expected": M,
        "unique_positions_asserted_by_construction": M,
        "streamed_injective_entries": streamed_checks,
        "actual_composite_key_entries_hashed": streamed_checks,
        "actual_composite_key_payload_sha256": key_hasher.hexdigest(),
        "composite_key_injectivity": "ANALYTIC_ARGUMENT_BY_POSITION_REFINEMENT: the bitwise position+LOS+scale key cannot collide when bitwise positions are unique; LOS/scale do not weaken position uniqueness.",
        "group_count_implied_by_injective_lattice": M,
        "source_count": M,
        "compression_ratio_implied_M_over_G": 1.0,
        "chunk_size": CHUNK,
        "chunk_count": chunk_count,
        "boundary_id_checks": len(boundary_ids),
        "key_payload_bytes": 56,
        "projected_dictionary_entry_bytes": 192,
        "projected_dictionary_entry_bytes_assumption": "Conservative CPython bytes+dict+integer estimate; projected, not measured.",
        "projected_dictionary_bytes": projected_dict,
        "persistent_array_bytes": persistent,
        "projected_persistent_array_total_bytes": persistent_total,
        "projected_persistent_plus_dictionary_bytes": projected_total,
        "host_ceiling_bytes": HOST_CEILING,
        "device_ceiling_bytes": DEVICE_CEILING,
        "host_memory_gate": memory_gate["host_status"],
        "device_memory_gate": memory_gate["device_status"],
        "dictionary_allocation_performed": False,
        "full_pm_geometry_allocated": False,
        "terminal_exact_grouping_decision": "NO_GO_M_OVER_G_LE_1.01",
        "terminal_scope": "Undisplaced procedural N192 lattice only; a later displaced PM realization must be measured separately and cannot inherit this ratio.",
        "q4_3_gradient_timing": "SKIPPED_BY_TERMINAL_BRANCH",
        "gpfs_used": False,
        "slurm_used": False,
    }
