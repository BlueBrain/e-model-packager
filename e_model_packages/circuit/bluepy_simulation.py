"""Access to bluepy Simulation."""

import os.path
from functools import lru_cache
from bluepy_configfile.configfile import BlueConfig


@lru_cache(maxsize=1)
def read_simulation(blueconfig_path):
    """Read the simulation into Bluepy object.

    Args:
        blueconfig_path (str): path to the BlueConfig.

    Returns:
        Object: Bluepy's Simulation object.
    """
    with open(blueconfig_path, "r", encoding="utf-8") as blueconfig_handle:
        blue_config = BlueConfig(blueconfig_handle)
    return blue_config


class BluepySimulation:
    """Wrapper over Bluepy's Simulation class."""

    def __init__(self, blueconfig_path):
        """Use cached method to read the sim."""
        self.blueconfig = read_simulation(blueconfig_path)

    @property
    def morph_dir(self) -> str:
        """Directory containing the morphology."""
        path = str(self.blueconfig.Run["MorphologyPath"])

        if path.endswith("ascii"):
            return path

        return os.path.join(path, "ascii")

    @property
    def morph_parent_dir(self) -> str:
        """Directory containing the morphology."""
        path = str(self.blueconfig.Run["MorphologyPath"])

        if "ascii" in path:  # if ascii, remove it
            return os.path.dirname(path)

        return path
