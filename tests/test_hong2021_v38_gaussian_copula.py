import hashlib
from pathlib import Path

import numpy as np

from hong2021_residual_evaluate import CENTERED_SCHEMAS
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v38_development_gate import classify
from hong2021_v38_gaussian_copula import ENSEMBLE_SCHEMA, EPSILON, FEATURES, PROGRAM_SHA256, feature_transforms, source_balanced_normalization


REPO=Path(__file__).resolve().parents[1]


def test_program_hash_and_firewall():
    path=REPO/"config/hong2021_v38_gaussian_copula_innovation_program.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest()==PROGRAM_SHA256
    text=path.read_text();assert '"validation_power_used_for_scaling": false' in text;assert '"donor_translation": false' in text;assert '"density_field_clipping": false' in text;assert '"posthoc_Ak": false' in text


def test_grid_midrank_epsilon():
    assert EPSILON==1/(2*64**3)


def test_source_balance_and_feature_dc():
    rows={DOMAIN_ORDER[0]:(np.full(len(FEATURES),2.),np.full(len(FEATURES),4.),2),DOMAIN_ORDER[1]:(np.full(len(FEATURES),1000.),np.full(len(FEATURES),10000.),1000),DOMAIN_ORDER[2]:(np.full(len(FEATURES),6.),np.full(len(FEATURES),12.),2)}
    mean,std=source_balanced_normalization(rows);np.testing.assert_allclose(mean,5/3);assert np.all(std>0)
    value=np.arange(len(FEATURES)*4**3,dtype=float).reshape(len(FEATURES),4,4,4)
    # feature_transforms is fixed to 64^3, so exercise it on its native shape.
    native=np.zeros((len(FEATURES),64,64,64));native[:,0,0,0]=np.arange(1,len(FEATURES)+1)
    transformed=feature_transforms(native,np.zeros(len(FEATURES)),np.ones(len(FEATURES)));assert np.allclose(transformed[:,0,0,0],0)


def test_classification_and_evaluator_schema():
    assert classify(True,True,True,True)[0]=="linear_query_conditioned_gaussian_copula_sufficient"
    assert classify(False,False,False,False)[0]=="linear_gaussian_query_conditioning_is_not_a_common_domain_repair"
    assert ENSEMBLE_SCHEMA in CENTERED_SCHEMAS
