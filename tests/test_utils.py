"""Contains tests for the utility functions."""

import os
from functools import partial
import configparser

from e_model_packages.sscx2020.utils import get_gid_from_circuit, read_circuit

test_config = configparser.ConfigParser()
test_config.read(os.path.join("tests", "luigi_test.cfg"))
get_param = partial(test_config.get, "params")


def test_get_gid_from_circuit():
    """Test the get_gid_from_circuit util function."""
    mtype = get_param("mtype")
    etype = get_param("etype")
    region = get_param("region")
    gidx = int(get_param("gidx"))
    gid_gt = int(get_param("gid"))

    # get circuit path
    config_circuit = configparser.ConfigParser()
    config_circuit.read(os.path.join("tests", "luigi_test.cfg"))
    circuit_config_path = config_circuit.get("paths", "circuit")

    circuit_obj, _ = read_circuit(circuit_config_path)

    gid = get_gid_from_circuit(
        mtype=mtype, etype=etype, region=region, gidx=gidx, circuit=circuit_obj
    )

    assert gid == gid_gt
