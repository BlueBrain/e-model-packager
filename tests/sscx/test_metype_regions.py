"""Tests the values of metype_regions.csv input used in building the sscx packages."""

import configparser
from pathlib import Path
import pandas as pd
from e_model_packages.circuit import BluepyCircuit

test_config = configparser.ConfigParser()
test_config.read(Path.cwd() / "tests" / "luigi_test_sscx.cfg")


def test_intended_regions():
    """Check if all regions are intended."""
    metype_gids_path = test_config.get("paths", "metype_gids")
    df = pd.read_csv(metype_gids_path)
    assert all(df.region == df.intended_region)


def test_cell_values():
    """Check if the cell attributes are the same as the ones retrieved from circuit."""
    metype_gids_path = test_config.get("paths", "metype_gids")
    df = pd.read_csv(metype_gids_path)
    gids = df.gid.values

    circuit_config_path = test_config.get("paths", "circuit")
    circuit = BluepyCircuit(circuit_config_path)
    bluepy_gids_df = circuit.circuit.cells.get(gids)

    assert all(df.mtype.values == bluepy_gids_df.mtype.values)
    assert all(df.etype.values == bluepy_gids_df.etype.values)
    assert all(df.region.values == bluepy_gids_df.region.values)
    assert all(df.morphology.values == bluepy_gids_df.morphology.values)
