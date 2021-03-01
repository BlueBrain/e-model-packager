"""Contains tests for the bluepy_circuit."""

import os
from functools import partial
import configparser

from e_model_packages.circuit import BluepyCircuit

test_config = configparser.ConfigParser()
test_config.read(os.path.join("tests", "luigi_test.cfg"))
get_param = partial(test_config.get, "params")


def test_get_gid_from_circuit():
    """Test get_gid_from_circuit method."""
    mtype = get_param("mtype")
    etype = get_param("etype")
    region = get_param("region")
    gidx = int(get_param("gidx"))
    gid_gt = int(get_param("gid"))

    # get circuit path
    config_circuit = configparser.ConfigParser()
    config_circuit.read(os.path.join("tests", "luigi_test.cfg"))
    circuit_config_path = config_circuit.get("paths", "circuit")

    circuit = BluepyCircuit(circuit_config_path)
    gid = circuit.get_gid_from_circuit(
        mtype=mtype, etype=etype, region=region, gidx=gidx
    )

    assert gid == gid_gt
