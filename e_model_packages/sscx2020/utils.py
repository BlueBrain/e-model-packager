"""Contains the utility functions for needed the workflow."""

import json
import os
from contextlib import contextmanager
from functools import lru_cache
import numpy as np
import pandas as pd
from tqdm import tqdm

import bluepy
from bluepy.v2 import Cell as bpcell
from bluepy_configfile.configfile import BlueConfig

# pylint: disable=super-with-arguments


@lru_cache(maxsize=1)
def extract_circuit_metype_region_gids(circuit_config_path, gids_per_metype, regions):
    """Extracts the metype region and gids from the circuit.

    Args:
        circuit_config_path (str): the path to circuit config
        gids_per_metype (int): number of gids to be extracted for each combo
        regions (tuple of str): the regions of interest to be extracted
    Returns:
        Dictionary contaning the metype, region and gids.
    """
    circuit, _ = read_circuit(circuit_config_path)
    metype_gids = {}

    cell_props_df = circuit.cells.get(
        properties=[bpcell.MTYPE, bpcell.ETYPE, bpcell.REGION]
    ).drop_duplicates()
    cell_props_df = cell_props_df.loc[cell_props_df["region"].isin(regions)]
    cell_props = list(
        zip(cell_props_df.mtype, cell_props_df.etype, cell_props_df.region)
    )

    print("Extracting mtype, etype, region and gids from circuit.")
    for mtype, etype, region in tqdm(cell_props):
        metype_gids[(mtype, etype, region)] = list(
            circuit.cells.ids(
                {
                    bpcell.MTYPE: mtype,
                    bpcell.ETYPE: etype,
                    bpcell.REGION: region,
                },
                limit=gids_per_metype,
            )
        )
    return metype_gids


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


def get_output_path(mtype, etype, region, gidx, workflow_output_dir):
    """Returns the path to the outputs directory of one cell model."""
    inner_folder_name = combine_names(mtype, etype, gidx)
    recording_path = os.path.join(mtype, etype, region, inner_folder_name)

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

    Takes another config file as basis (or None).
    Two lines have to be added: one disabling running all steps,
        and one putting the step number to be run to 1.
    If any of the two lines are encountered, they are replaced.
    If any of the two lines are not encountered,
        they are added at the end of the Protocol field, if any.
    If there is no Protocol field, it is added with the two lines at the end of the file.
    """
    new_config_path = os.path.join(config_dir, new_config)

    if original_config is None:
        with open(new_config_path, "w") as out_file:
            out_file.write(
                "[Protocol]\n" + "run_all_steps=False\n" + "run_step_number=1\n"
            )
    else:
        original_config_path = os.path.join(config_dir, original_config)

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


def get_gid_from_circuit(mtype, etype, region, gidx, circuit):
    """Returns the circuit gid given the index of  cell properties dataframe.

    Args:
        mtype (str): morphological type
        etype (str): electrophysiological type
        region (str): circuit region
        gidx (int): index of the bluepy circuit cell ids dataframe
        circuit (bluepy.v2.circuit.Circuit): the circuit object
    Returns:
        int: The gid from the circuit.
    """
    gids = list(
        circuit.cells.ids(
            {
                bpcell.MTYPE: mtype,
                bpcell.ETYPE: etype,
                bpcell.REGION: region,
            }
        )
    )
    gid = gids[gidx - 1]
    return gid
