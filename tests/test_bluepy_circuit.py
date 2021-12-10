"""Contains tests for the bluepy_circuit."""

import os
from functools import partial
import configparser

from e_model_packages.circuit import BluepyCircuit, BluepySimulation

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

    circuit_config_path = test_config.get("paths", "circuit")

    circuit = BluepyCircuit(circuit_config_path)
    gid = circuit.get_gid_from_circuit(
        mtype=mtype, etype=etype, region=region, gidx=gidx
    )

    assert gid == gid_gt


def test_morph_dir_from_bluepysimulation():
    """Test the values of morph_dir and morph_parent_dir in BluepySimulation."""
    morph_parent_dir = (
        "/gpfs/bbp.cscs.ch/project/proj83/entities/morph-release-2020-08-10"
    )
    morph_dir = (
        "/gpfs/bbp.cscs.ch/project/proj83/entities/morph-release-2020-08-10/ascii"
    )

    circuit_config_path = test_config.get("paths", "circuit")

    simulation = BluepySimulation(circuit_config_path)
    assert simulation.morph_parent_dir == morph_parent_dir
    assert simulation.morph_dir == morph_dir
