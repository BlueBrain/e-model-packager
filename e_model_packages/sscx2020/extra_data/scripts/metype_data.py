"""Produce a me-type data json file."""
import argparse
import collections
import json
import os
import numpy as np

import efel
import neurom as nm

from load import find_param_file, load_config


class NpEncoder(json.JSONEncoder):
    """Class to encode np.integer as python int."""

    def default(self, o):
        """Convert numpy integer to int."""
        if isinstance(o, np.integer):
            return int(o)
        else:
            return super(NpEncoder, self).default(o)


def base_dict(unit, name, value):
    """Basic dictionnary for building the me-type json file."""
    return {"unit": unit, "name": name, "value": value, "tooltip": ""}


def get_morph_path(config):
    """Return path to morphology file."""
    # get morphology path from constants
    constants_path = os.path.join(
        config.get("Paths", "constants_dir"), config.get("Paths", "constants_file")
    )
    with open(constants_path, "r") as f:
        data = json.load(f)
    morph_dir = data["morph_dir"]
    morph_fname = data["morph_fname"]

    # change it if it is specified in config file
    if config.has_option("Paths", "morph_dir"):
        morph_dir = config.get("Paths", "morph_dir")
    else:
        morph_dir = os.path.join(config.get("Paths", "memodel_dir"), morph_dir)
    if config.has_option("Paths", "morph_file"):
        morph_fname = config.get("Paths", "morph_file")

    return os.path.join(morph_dir, morph_fname)


def get_morph_data(config):
    """Return the morphological data in a dictionnary."""
    # get morph path
    morph_path = get_morph_path(config)

    # extract data
    values = []
    nrn = nm.load_neuron(morph_path)
    neurite_names = ["axon", "apical", "basal"]
    neurite_types = [nm.AXON, nm.APICAL_DENDRITE, nm.BASAL_DENDRITE]
    for n_name, n_type in zip(neurite_names, neurite_types):
        leng = nm.get("total_length", nrn, neurite_type=n_type)
        # to avoid error when there is no neurite
        if len(leng) == 0:
            leng = [0]
        length = base_dict("\u00b5m", "total {} length".format(n_name), leng[0])
        values.append(length)

        vol = nm.get("neurite_volumes", nrn, neurite_type=n_type)
        if len(vol) == 0:
            vol = [0]
        volume = base_dict("\u00b5m\u00b3", "total {} volume".format(n_name), vol[0])
        values.append(volume)

        branch_order = nm.get("section_branch_orders", nrn, neurite_type=n_type)
        if len(branch_order) == 0:
            branch_order = [0]
        max_branch_order = base_dict(
            "", "{} maximum branch order".format(n_name), max(branch_order)
        )
        values.append(max_branch_order)

        sec_len = nm.get("section_lengths", nrn, neurite_type=n_type)
        if len(sec_len) == 0:
            sec_len = [0]
        max_section_length = base_dict(
            "\u00b5m", "{} maximum section length".format(n_name), max(sec_len)
        )
        values.append(max_section_length)

    soma_r = nm.get("soma_radii", nrn)
    soma_diam = base_dict("\u00b5m", "soma diameter", 2 * soma_r[0])
    values.append(soma_diam)

    return {"values": values, "name": "Anatomy"}


def get_physiology_data(config):
    """Analyse the output of the RmpRiTau protocol."""
    # get parameters from config
    step_number = config.getint("Protocol", "run_step_number")
    stim_start = config.getint("Protocol", "stimulus_delay")
    stim_duration = config.getint("Protocol", "stimulus_duration")
    amp_filename = os.path.join(
        config.get("Paths", "protocol_amplitudes_dir"),
        config.get("Paths", "protocol_amplitudes_file"),
    )

    # get current amplitude data
    with open(amp_filename, "r") as f:
        data = json.load(f)
    current_amplitude = data["amps"][step_number - 1]

    # get data from run.py output
    fname = "soma_voltage_step{}.dat".format(step_number)
    fpath = os.path.join("python_recordings", fname)
    data = np.loadtxt(fpath)

    # Prepare the trace data
    trace = {}
    trace["T"] = data[:, 0]  # time
    trace["V"] = data[:, 1]  # soma voltage
    trace["stim_start"] = [stim_start]
    trace["stim_end"] = [stim_start + stim_duration]

    # Calculate the necessary eFeatures
    efel_results = efel.getFeatureValues(
        [trace],
        [
            "voltage_base",
            "steady_state_voltage_stimend",
            "decay_time_constant_after_stim",
        ],
    )

    voltage_base = efel_results[0]["voltage_base"][0]
    dct = efel_results[0]["decay_time_constant_after_stim"][0]

    # Calculate input resistance
    trace["decay_start_after_stim"] = efel_results[0]["voltage_base"]
    trace["decay_end_after_stim"] = efel_results[0]["steady_state_voltage_stimend"]
    trace["stimulus_current"] = [current_amplitude]
    efel_results = efel.getFeatureValues([trace], ["ohmic_input_resistance_vb_ssse"])
    input_resistance = efel_results[0]["ohmic_input_resistance_vb_ssse"][0]

    # build dictionnary to be returned
    names = ["resting membrane potential", "input resistance", "membrane time constant"]
    vals = [voltage_base, input_resistance, dct]
    units = ["mV", "MOhm", "ms"]
    values = []
    for name, val, unit in zip(names, vals, units):
        data = base_dict(unit, name, val)
        values.append(data)

    return {"values": values, "name": "Physiology"}


def get_mechanisms_data(config):
    """Return a dictionnary containing channel mechanisms for each section."""
    # get emodel
    constants_path = os.path.join(
        config.get("Paths", "constants_dir"), config.get("Paths", "constants_file")
    )
    with open(constants_path, "r") as f:
        data = json.load(f)
    emodel = data["template_name"]

    # get mechanisms file
    recipes_path = os.path.join(
        config.get("Paths", "recipes_dir"), config.get("Paths", "recipes_file")
    )
    mechs_filename = find_param_file(recipes_path, emodel)

    # get mechs data and fill in dictionnary
    with open(mechs_filename) as mechs_file:
        mech_definitions = json.load(
            mechs_file, object_pairs_hook=collections.OrderedDict
        )["mechanisms"]

    dend_mechs = []
    soma_mechs = []
    axon_mechs = []
    for sectionlist, channels in mech_definitions.items():
        # sectionlist "all" is ignored here
        if sectionlist == "alldend":
            dend_mechs += channels["mech"]
        elif sectionlist == "somadend":
            dend_mechs += channels["mech"]
            soma_mechs += channels["mech"]
        elif sectionlist == "somaxon":
            soma_mechs += channels["mech"]
            axon_mechs += channels["mech"]
        elif sectionlist == "allact":
            dend_mechs += channels["mech"]
            soma_mechs += channels["mech"]
            axon_mechs += channels["mech"]
        elif sectionlist == "apical":
            dend_mechs += channels["mech"]
        elif sectionlist == "basal":
            dend_mechs += channels["mech"]
        elif sectionlist == "axonal":
            axon_mechs += channels["mech"]
        elif sectionlist == "somatic":
            soma_mechs += channels["mech"]

    values = [
        {"values": dend_mechs, "name": "dendrite"},
        {"values": axon_mechs, "name": "axonal"},
        {"values": soma_mechs, "name": "somatic"},
    ]
    return {"values": values, "name": "Channel mechanisms"}


def write_metype_json(config):
    """Write the me-type fact json file."""
    anatomy = get_morph_data(config)
    physiology = get_physiology_data(config)
    channel_mechanisms = get_mechanisms_data(config)

    output = [anatomy, physiology, channel_mechanisms]

    output_fpath = "me_type_factsheeet.json"
    with open(output_fpath, "w") as out_file:
        json.dump(output, out_file, indent=4, cls=NpEncoder)
    print("me-type json file written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--c",
        default=None,
        help="the name of the config file",
    )
    args = parser.parse_args()

    config_file = args.c
    config = load_config(filename=config_file)

    write_metype_json(config)
