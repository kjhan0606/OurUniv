import hashlib
from pathlib import Path

import numpy as np

from hong2021_residual_evaluate import CENTERED_SCHEMAS
from hong2021_v39_development_gate import classify
from hong2021_v39_patch_copula import (
    BLOCKS, CONTEXT_FEATURES, ENSEMBLE_SCHEMA, PROGRAM_SHA256,
    assignment, blocks_to_cube, cube_to_blocks, transport_blocks,
)


REPO = Path(__file__).resolve().parents[1]


def test_program_hash_and_firewall():
    path = REPO / "config/hong2021_v39_bijective_patch_copula_program.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PROGRAM_SHA256
    text = path.read_text(); assert '"additive_query_predictor": false' in text; assert '"donor_translation": false' in text; assert '"density_field_clipping": false' in text; assert '"posthoc_Ak": false' in text


def test_block_roundtrip_and_rank_multiset():
    cube = np.arange(64**3, dtype=np.float32).reshape(64, 64, 64)
    np.testing.assert_array_equal(blocks_to_cube(cube_to_blocks(cube)), cube)
    permutation = np.arange(BLOCKS)[::-1]
    transported = transport_blocks(cube, permutation)
    np.testing.assert_array_equal(np.sort(transported.reshape(-1)), cube.reshape(-1))


def test_hungarian_identity_is_bijective():
    rng = np.random.default_rng(39)
    descriptor = rng.normal(size=(BLOCKS, CONTEXT_FEATURES))
    permutation, diagnostics = assignment(descriptor, descriptor)
    np.testing.assert_array_equal(permutation, np.arange(BLOCKS))
    assert diagnostics["nonidentity_fraction"] == 0


def test_classification_and_evaluator_schema():
    assert classify(True, True, True, True)[0] == "bijective_local_patch_copula_sufficient"
    assert classify(False, False, False, False)[0] == "fixed_local_patch_copula_is_not_a_common_domain_repair"
    assert ENSEMBLE_SCHEMA in CENTERED_SCHEMAS
