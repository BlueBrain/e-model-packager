"""Access to bluepy Simulation."""

import os.path
from functools import lru_cache
import pandas as pd
from bluepy_configfile.configfile import BlueConfig


@lru_cache(maxsize=1)
def read_simulation(blueconfig_path):
    """Read the simulation into Bluepy object.

    Args:
        blueconfig_path (str): path to the BlueConfig.

    Returns:
        Object: Bluepy's Simulation object.
    """
    blue_config = BlueConfig(open(blueconfig_path))
    return blue_config


class BluepySimulation:
    """Wrapper over Bluepy's Simulation class."""

    def __init__(self, blueconfig_path):
        """Use cached method to read the sim."""
        self.blueconfig = read_simulation(blueconfig_path)

    @property
    def morph_dir(self):
        """Directory containing the morphology."""
        return os.path.join(self.blueconfig.Run["MorphologyPath"], "ascii")

    def get_mecombo_emodel(self, mecombo):
        """Returns the emodel name as well as its threshold and holding currents.

        Args:
            mecombo(str): Name of mecombo.
        """
        mecombo_filename = self.blueconfig.Run["MEComboInfoFile"]

        df = pd.read_csv(mecombo_filename, sep="\t")
        mecombo_row = df[df["combo_name"] == mecombo]

        emodel = mecombo_row["emodel"].values[0]
        threshold_curr = mecombo_row["threshold_current"].values[0]
        holding_curr = mecombo_row["holding_current"].values[0]

        return emodel, threshold_curr, holding_curr
