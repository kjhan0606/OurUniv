import json
from pathlib import Path

import numpy as np
import pytest
from astropy.coordinates import CartesianRepresentation, SkyCoord

from cf4_lg_generated_roles import enumerate_role_hypotheses, predict_role_hypothesis
from cf4_lg_observation_contract import basis, solar_reference


ROOT = Path(__file__).resolve().parents[1]


def fixture():
    contract = json.loads((ROOT / "config/cf4_lg_observation_contract_v1.json").read_text())
    solar_position, solar_velocity = solar_reference(contract)
    rotation = SkyCoord(CartesianRepresentation(np.eye(3)),
                        frame="supergalactic").icrs.cartesian.xyz.value
    dtype = [(name, "f8") for name in ("totm", "x", "y", "z", "vx", "vy", "vz")]
    children = np.zeros(4, dtype=dtype)
    children["totm"] = [1e12, 1.2e12, 8e11, 1e11]
    coordinates = np.tile(np.array([192., 192., 192.]), (4, 1))
    for index, name in ((1, "M31"), (3, "M33")):
        datum = contract["measurements"][name]
        distance = 10 ** ((datum["value"][0] - 10) / 5)
        position_icrs = solar_position + distance * basis(datum["ra_deg"], datum["dec_deg"])[0]
        coordinates[index] += rotation.T @ position_icrs * .746 / 1000
    coordinates[2] += [1., 1., 0.]
    for axis, name in enumerate(("x", "y", "z")):
        children[name] = coordinates[:, axis]
    return children, np.array([0, 1, 2, 1]), contract, solar_position, solar_velocity


def test_all_role_branches_and_observed_geometry():
    children, parent, contract, solar_position, solar_velocity = fixture()
    hypotheses = enumerate_role_hypotheses(children, parent, [0, 1, 2])
    assert len(hypotheses) == 36
    assert sum(row["M33_branch"] == "unresolved" for row in hypotheses) == 12
    assert any(row["MW_M31_share_FoF_host"] for row in hypotheses)
    target = next(row for row in hypotheses if
                  row["MW_component"] == 0 and row["M31_component"] == 1 and
                  row["M33_component"] == 3)
    assert target["M33_branch"] == "shared_FoF_host"
    prediction = predict_role_hypothesis(children, target, contract, h=.746,
        box_mpc_h=384, solar_position_kpc=solar_position,
        solar_velocity_km_s=solar_velocity)
    assert prediction["status"] == "GENERATED_ROLE_DIAGNOSTIC_ONLY"
    assert max(prediction["sky_offsets_deg"].values()) < 1e-5
    assert np.isfinite(prediction["observables"]).all()
    unresolved = next(row for row in hypotheses if row["M33_branch"] == "unresolved")
    missing = predict_role_hypothesis(children, unresolved, contract, h=.746,
        box_mpc_h=384, solar_position_kpc=solar_position,
        solar_velocity_km_s=solar_velocity)
    assert missing["status"] == "UNRESOLVED_M33"
    assert "observables" not in missing


def test_duplicate_host_support_rejected():
    children, parent, _, _, _ = fixture()
    with pytest.raises(ValueError, match="unique"):
        enumerate_role_hypotheses(children, parent, [0, 0, 1])
    with pytest.raises(ValueError, match="size cap"):
        enumerate_role_hypotheses(children, parent, [0, 1, 2], max_hypotheses=35)
