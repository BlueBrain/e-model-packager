"""Contains the utility functions for needed the workflow."""

import json
import numpy as np
import bluepy
from bluepy_configfile.configfile import BlueConfig


def read_circuit(config_path):
    """Read circuit info."""
    circuit_config = bluepy.Circuit(config_path).v2
    blue_config = BlueConfig(open(config_path))

    return circuit_config, blue_config


class NpEncoder(json.JSONEncoder):
    """Class to encode np.integer as python int."""

    def default(self, obj):
        """Convert numpy integer to int."""
        if isinstance(obj, np.integer):
            return int(obj)
        else:
            return super(NpEncoder, self).default(obj)


def get_mecombo_emodels(blueconfig):
    """Create a dict matching me_combo names to template_names."""
    mecombo_filename = blueconfig.Run["MEComboInfoFile"]

    with open(mecombo_filename) as mecombo_file:
        mecombo_content = mecombo_file.read()

    mecombo_emodels = {}
    mecombo_thresholds = {}
    mecombo_hypamps = {}

    for line in mecombo_content.split("\n")[1:-1]:
        mecombo_info = line.split("\t")
        emodel = mecombo_info[4]
        me_combo = mecombo_info[5]
        threshold = float(mecombo_info[6])
        hypamp = float(mecombo_info[7])
        mecombo_emodels[me_combo] = emodel
        mecombo_thresholds[me_combo] = threshold
        mecombo_hypamps[me_combo] = hypamp

    return mecombo_emodels, mecombo_thresholds, mecombo_hypamps


def combine_names(mtype, etype, gidx):
    """Returns the combined metype and cell index."""
    return "_".join([mtype, etype, str(gidx)])
