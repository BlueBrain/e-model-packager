"""Contains circuit reading operations."""

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

from functools import lru_cache
from tqdm import tqdm
import bluepy
from bluepy import Cell as bpcell


@lru_cache(maxsize=1)
def read_circuit(circuit_config_path):
    """Read the circuit into Bluepy object.

    Args:
        circuit_config_path (str): path to the CircuitConfig.

    Returns:
        Object: Bluepy's Circuit object.
    """
    circuit_config = bluepy.Circuit(circuit_config_path)
    return circuit_config


class BluepyCircuit:
    """Wrapper over Bluepy's Circuit class."""

    def __init__(self, circuit_config_path):
        """Use cached method to read the circuit."""
        self.circuit = read_circuit(circuit_config_path)

    def get_gid_from_circuit(self, mtype, etype, region, gidx):
        """Returns the circuit gid given the index of cell properties dataframe.

        Args:
            mtype (str): morphological type
            etype (str): electrophysiological type
            region (str): circuit region
            gidx (int): index of the bluepy circuit cell ids dataframe
        Returns:
            int: The gid from the circuit.
        """
        gids = list(
            self.circuit.cells.ids(
                {
                    bpcell.MTYPE: mtype,
                    bpcell.ETYPE: etype,
                    bpcell.REGION: region,
                }
            )
        )
        gid = gids[gidx - 1]
        return gid

    def extract_circuit_metype_gids(self, n_gids):
        """Extracts the metype gids.

        Args:
            n_gids (int): Number of gids to be retrieved per metype.

        Returns:
            Dictionary containing gids per mtype and etype.
        """
        metype_gids = {}

        cell_props_df = self.circuit.cells.get(
            properties=[bpcell.MTYPE, bpcell.ETYPE]
        ).drop_duplicates()
        cell_props = list(zip(cell_props_df.mtype, cell_props_df.etype))
        print("Extracting gids from circuit.")
        for mtype, etype in tqdm(cell_props):
            metype_gids[(mtype, etype)] = list(
                self.circuit.cells.ids(
                    {
                        bpcell.MTYPE: mtype,
                        bpcell.ETYPE: etype,
                    },
                    limit=n_gids,
                )
            )
        return metype_gids

    def get_cell_attributes(self, gid):
        """Retrieve the cell attributes from circuit.

        Args:
            gid (int): cell identifier.
        """
        cell = self.circuit.cells.get(gid)
        return CellAttributes(cell)

    def get_emodel_attributes(self, gid):
        """Retrieve the emodel attributes of a gid.

        Args:
            gid (int): cell identifier.
        """
        # pylint: disable=no-member
        emodel = self.circuit.emodels.get_mecombo_info(gid)
        return EmodelAttributes(emodel)


class CellAttributes:
    """Cell attributes access class."""

    def __init__(self, cell):
        """Store only attributes of interest."""
        self.me_combo = cell.me_combo
        self.morphology = cell.morphology
        self.morphology_fname = f"{self.morphology}.asc"
        self.layer = f"L{cell.layer}"


class EmodelAttributes:
    """Emodel attributes of a cell."""

    def __init__(self, emodel):
        """Attributes of a gid's emodel."""
        self.name = emodel["emodel"]
        self.holding_current = emodel["holding_current"]
        self.threshold_current = emodel["threshold_current"]
