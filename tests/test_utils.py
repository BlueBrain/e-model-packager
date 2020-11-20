"""Contains tests for the utility functions."""

import os
import configparser

from e_model_packages.sscx2020.utils import get_gid_from_circuit, read_circuit


def test_get_gid_from_circuit():
    """Test the get_gid_from_circuit util function."""
    mtype = "L5_TPC:A"
    etype = "cADpyr"
    region = "S1ULp"
    gidx = 79598
    gid_gt = 4138379

    # get circuit path
    config_circuit = configparser.ConfigParser()
    config_circuit.read(os.path.join("tests", "luigi_test.cfg"))
    circuit_config_path = config_circuit.get("paths", "circuit")

    circuit_obj, _ = read_circuit(circuit_config_path)

    gid = get_gid_from_circuit(
        mtype=mtype, etype=etype, region=region, gidx=gidx, circuit=circuit_obj
    )

    assert gid == gid_gt
