"""Contains circuit reading operations."""

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

    def extract_circuit_metype_region_gids(self, n_gids, regions):
        """Extracts the metype region and gids from the circuit.

        Args:
            n_gids (int): number of gids to be extracted for each combo
            regions (iterable of str): the regions of interest to be extracted
        Returns:
            Dictionary contaning the metype, region and gids.
        """
        metype_region_gids = {}

        cell_props_df = self.circuit.cells.get(
            properties=[bpcell.MTYPE, bpcell.ETYPE, bpcell.REGION]
        ).drop_duplicates()
        cell_props_df = cell_props_df.loc[cell_props_df["region"].isin(regions)]
        cell_props = list(
            zip(cell_props_df.mtype, cell_props_df.etype, cell_props_df.region)
        )

        print("Extracting mtype, etype, region and gids from circuit.", flush=True)
        for mtype, etype, region in tqdm(cell_props):
            metype_region_gids[(mtype, etype, region)] = list(
                self.circuit.cells.ids(
                    {
                        bpcell.MTYPE: mtype,
                        bpcell.ETYPE: etype,
                        bpcell.REGION: region,
                    },
                    limit=n_gids,
                )
            )
        return metype_region_gids

    def get_cell_attributes(self, gid):
        """Retrieve the cell attributes from circuit.

        Args:
            gid (int): cell identifier.
        """
        cell = self.circuit.cells.get(gid)
        return CellAttributes(cell)


class CellAttributes:
    """Cell attributes access class."""

    def __init__(self, cell):
        """Retrieve the cell & set store the attributes."""
        self.me_combo = cell.me_combo
        self.morphology = cell.morphology
        self.morphology_fname = f"{self.morphology}.asc"
        self.layer = f"L{cell.layer}"
