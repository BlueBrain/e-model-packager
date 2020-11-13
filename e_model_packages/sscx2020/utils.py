"""Contains the utility functions for needed the workflow."""

import json
import os
from contextlib import contextmanager
import numpy as np
import pandas as pd
import bluepy
from bluepy_configfile.configfile import BlueConfig

# pylint: disable=super-with-arguments


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


def get_mecombo_emodel(blueconfig, mecombo):
    """Returns the emodel name as well as it's threshold and holding currents.

    Args:
        blueconfig(object): Blueconfig object.
        mecombo(str): Name of mecombo.
    """
    mecombo_filename = blueconfig.Run["MEComboInfoFile"]

    df = pd.read_csv(mecombo_filename, sep="\t")
    mecombo_row = df[df["combo_name"] == mecombo]

    emodel = mecombo_row["emodel"].values[0]
    threshold_curr = mecombo_row["threshold_current"].values[0]
    holding_curr = mecombo_row["holding_current"].values[0]

    return emodel, threshold_curr, holding_curr


def combine_names(mtype, etype, gidx):
    """Returns the combined metype and cell index."""
    return "_".join([mtype, etype, str(gidx)])


def get_morph_emodel_names(gid, config):
    """Get morphology and emodel filenames."""
    circuit, blueconfig = read_circuit(config["paths"]["circuit"])

    cell = circuit.cells.get(gid)
    morph_fname = "%s.asc" % cell.morphology

    emodel, _, _ = get_mecombo_emodel(blueconfig, cell.me_combo)
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


def create_single_step_config(original_config, new_config, config_dir):
    """Create a config file which launches one single step protocol.

    Takes another config file as basis.
    Two lines have to be added: one disabling running all steps,
        and one putting the step number to be run to 1.
    If any of the two lines are encountered, they are replaced.
    If any of the two lines are not encountered,
        they are added at the end of the Protocol field, if any.
    If there is no Protocol field, it is added with the two lines at the end of the file.
    """
    original_config_path = os.path.join(config_dir, original_config)
    new_config_path = os.path.join(config_dir, new_config)

    in_protocol = False
    all_steps_written = False
    one_step_written = False

    with open(new_config_path, "w") as out_file:
        with open(original_config_path, "r") as in_file:
            for line in in_file:
                # At the end of protocol field, add any parameter not yet replaced.
                if line[0] == "[" and in_protocol:
                    if not all_steps_written:
                        out_file.write("run_all_steps=False\n")
                        all_steps_written = True
                    if not one_step_written:
                        out_file.write("run_step_number=1\n")
                        one_step_written = True
                    in_protocol = False
                if line.rstrip() == "[Protocol]":
                    in_protocol = True
                # If encountered, replace these parameters.
                if "run_all_steps" in line.split("="):
                    line = "run_all_steps=False\n"
                    all_steps_written = True
                if "run_step_number" in line.split("="):
                    line = "run_step_number=1\n"
                    one_step_written = True
                # Copy the file.
                out_file.write(line)

            # If no Protocol field was encountered, add it with the two parameters.
            if not (all_steps_written and one_step_written):
                last_lines = "[Protocol]\n"
                if not all_steps_written:
                    last_lines += "run_all_steps=False\n"
                if not one_step_written:
                    last_lines += "run_step_number=1\n"
                out_file.write(last_lines)
