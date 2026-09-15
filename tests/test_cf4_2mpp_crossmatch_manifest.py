from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from cf4_2mpp_crossmatch_manifest import (
    CrossmatchManifestError,
    build_secure_crossmatch_manifest,
    validate_secure_crossmatch_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MAPPING = ROOT / "data/cf4_2mpp_crossmatch_v1.csv"
SUMMARY = ROOT / "config/cf4_2mpp_crossmatch_v1_result.json"


def test_secure_manifest_is_derived_from_canonical_mapping_and_group_bound():
    manifest = build_secure_crossmatch_manifest(MAPPING, SUMMARY)
    assert manifest["status"] == "VALIDATED_SECURE_OBJECT_MANIFEST"
    assert manifest["source"]["mapping_sha256"] == "64e4f8a1a8a612a19788ac759062930991a8ffe52bfa203635845fa1ad7a83bf"
    assert manifest["counts"] == {
        "mapping_rows": 55877,
        "secure_rows": 16584,
        "secure_cf4_groups": 11610,
    }
    first = manifest["entries"][0]
    assert first["secure_object_id"] == f"cf4:{first['cf4_recno']}"
    assert first["twompp_object_id"] == f"2mpp:{first['twompp_recno']}"
    assert first["cf4_group_id"].startswith("cf4_1PGC:")
    assert manifest["factor_ownership"]["independent_twompp_redshift_factor"] is False


def test_manifest_rejects_tampering_and_quarantine_promotion():
    manifest = build_secure_crossmatch_manifest(MAPPING, SUMMARY)
    tampered = deepcopy(manifest)
    tampered["entries"][0]["match_class"] = "extended_review_candidate"
    with pytest.raises(CrossmatchManifestError, match="quarantine"):
        validate_secure_crossmatch_manifest(tampered, mapping_path=MAPPING, summary_path=SUMMARY)
    tampered = deepcopy(manifest)
    tampered["entries"][1]["twompp_object_id"] = tampered["entries"][0]["twompp_object_id"]
    with pytest.raises(CrossmatchManifestError, match="identity does not bind"):
        validate_secure_crossmatch_manifest(tampered, mapping_path=MAPPING, summary_path=SUMMARY)
    tampered = deepcopy(manifest)
    tampered["entries"][0]["cf4_group_id"] = "cf4_1PGC:forged"
    with pytest.raises(CrossmatchManifestError, match="canonical secure rows"):
        validate_secure_crossmatch_manifest(tampered, mapping_path=MAPPING, summary_path=SUMMARY)
