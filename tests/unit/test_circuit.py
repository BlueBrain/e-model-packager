"""Unit tests for the circuit module."""

import os
from functools import partial
import configparser

from e_model_packages.circuit import BluepyCircuit, BluepySimulation


class TestUsingSSCXSim:
    """Unit tests that use an SSCX simulation."""

    test_config = configparser.ConfigParser()
    test_config.read(os.path.join("tests", "luigi_test_sscx.cfg"))
    get_param = partial(test_config.get, "params")

    def test_get_gid_from_circuit(self):
        """Test get_gid_from_circuit method."""
        mtype = self.get_param("mtype")
        etype = self.get_param("etype")
        region = self.get_param("region")
        gidx = int(self.get_param("gidx"))
        gid_gt = int(self.get_param("gid"))

        circuit_config_path = self.test_config.get("paths", "circuit")

        circuit = BluepyCircuit(circuit_config_path)
        gid = circuit.get_gid_from_circuit(
            mtype=mtype, etype=etype, region=region, gidx=gidx
        )

        assert gid == gid_gt

    def test_morph_dir_from_bluepysimulation(self):
        """Test the values of morph_dir and morph_parent_dir in BluepySimulation."""
        morph_parent_dir = (
            "/gpfs/bbp.cscs.ch/project/proj83/entities/morph-release-2020-08-10"
        )
        morph_dir = (
            "/gpfs/bbp.cscs.ch/project/proj83/entities/morph-release-2020-08-10/ascii"
        )

        circuit_config_path = self.test_config.get("paths", "circuit")

        simulation = BluepySimulation(circuit_config_path)
        assert simulation.morph_parent_dir == morph_parent_dir
        assert simulation.morph_dir == morph_dir


class TestUsingThalamusSim:
    """Unit tests that use a Thalamus simulation."""

    test_config = configparser.ConfigParser()
    test_config.read(os.path.join("tests", "luigi_test_thalamus.cfg"))

    def test_extract_circuit_metype_gids(self):
        """Tests the extract_circuit_metype_gids method on a thalamus sim."""
        circuit_config_path = self.test_config.get("paths", "circuit")
        circuit = BluepyCircuit(circuit_config_path)
        metype_gids = circuit.extract_circuit_metype_gids(n_gids=5)
        assert set(metype_gids.keys()) == {
            ("VPL_IN", "bAC_IN"),
            ("VPL_TC", "dAD_ltb"),
            ("Rt_RC", "cAD_noscltb"),
            ("Rt_RC", "cNAD_noscltb"),
            ("VPL_TC", "dNAD_ltb"),
        }
