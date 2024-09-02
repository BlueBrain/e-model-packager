"""Tests the values of metype_regions.csv input used in building the sscx packages."""

"""
Copyright 2024 Blue Brain Project / EPFL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import configparser
from pathlib import Path
import pandas as pd
from e_model_packager.circuit import BluepyCircuit

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
