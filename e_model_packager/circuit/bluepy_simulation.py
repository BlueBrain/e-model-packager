"""Access to bluepy Simulation."""
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
