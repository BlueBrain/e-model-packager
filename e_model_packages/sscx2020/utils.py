"""Contains the utility functions for needed the workflow."""

import json
import os
import configparser
from contextlib import contextmanager
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

    def default(self, o):
        """Convert numpy integer to int."""
        if isinstance(o, np.integer):
            return int(o)
        else:
            return super(NpEncoder, self).default(o)


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


def get_morph_emodel_names(path, gid):
    """Get morphology and emodel filenames."""
    config = configparser.ConfigParser()
    config.read(os.path.join(path, "luigi.cfg"))
    circuit, blueconfig = read_circuit(config["paths"]["circuit"])

    mecombo_emodels, _, _ = get_mecombo_emodels(blueconfig)
    cell = circuit.cells.get(gid)

    morph_fname = "%s.asc" % cell.morphology
    emodel = mecombo_emodels[cell.me_combo]
    emodel_fname = "%s.hoc" % emodel

    return morph_fname, emodel_fname


def get_output_path(mtype, etype, gidx, workflow_output_dir):
    """Returns the path to the outputs directory of one cell model."""
    inner_folder_name = combine_names(mtype, etype, gidx)
    recording_path = os.path.join(mtype, etype, inner_folder_name)

    return os.path.join(workflow_output_dir, "memodel_dirs", recording_path)


@contextmanager
def cwd(path):
    """Cwd function that can be used in a context manager."""
    old_dir = os.getcwd()
    os.chdir(path)
    yield
    os.chdir(old_dir)
