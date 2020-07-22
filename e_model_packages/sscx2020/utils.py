"""Contains the utility functions for needed the workflow."""

import json
import numpy as np
import bluepy
from bluepy_configfile.configfile import BlueConfig


def read_circuit(config_path):
    """Read circuit info"""

    circuit_config = bluepy.Circuit(config_path).v2
    blue_config = BlueConfig(open(config_path))

    return circuit_config, blue_config

def combine_names(mtype, etype, gid):
    """Returns the combined metype and gid name."""
    return "_".join([mtype, etype, str(gid)])

class NpEncoder(json.JSONEncoder):
    """Class to encode np.integer as python int"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        else:
            return super(NpEncoder, self).default(obj)
