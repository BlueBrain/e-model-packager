"""Contains tests for the workflow."""
# pylint: disable=wrong-import-position
# pylint: disable=wrong-import-order
# pylint: disable=import-error
import configparser
import os
import re
import numpy as np
import pytest
from functools import partial

import bglibpy
from tests.decorators import launch_luigi
from e_model_packages.sscx2020.utils import (
    get_output_path,
    combine_names,
    cwd,
)
from e_model_packages.circuit import BluepyCircuit

from emodelrunner.json_loader import json_load
from emodelrunner.load import (
    load_config,
    find_param_file,
    get_release_params,
    load_amps,
)
from emodelrunner.write_factsheets import (
    get_morph_path,
    get_morph_data,
    get_physiology_data,
    get_morph_name_dict,
    get_emodel,
    get_recipe,
    get_prefix,
    get_exp_features_data,
    get_mechanisms_data,
    load_raw_exp_features,
    load_feature_units,
    load_fitness,
)

test_config = configparser.ConfigParser()
test_config.read(os.path.join("tests", "luigi_test.cfg"))
get_param = partial(test_config.get, "params")


@pytest.fixture(scope="session")
def prepare_test_synapses_config():
    """Prepares a test config with synapses that uses neuron's rng."""
    mtype = get_param("mtype")
    etype = get_param("etype")
    region = get_param("region")
    gidx = int(get_param("gidx"))
    configfile = "config_synapses.ini"

    output_path = os.path.join("tests", "output")
    memodel_path = get_output_path(mtype, etype, region, gidx, output_path)

    # re-write config file to have consistent randomness accross simulations
    config_path = os.path.join(memodel_path, "config", configfile)
    with open(config_path, "r") as config_f:
        config = config_f.read()
    new_config = re.sub("vecstim_random=.*", "vecstim_random=neuron", config)
    with open(config_path, "w") as config_f:
        config_f.write(new_config)


@launch_luigi(module="workflow", task="PrepareMEModelDirectory")
def test_directory_exists(
    mtype=get_param("mtype"),
    etype=get_param("etype"),
    region=get_param("region"),
    gid=int(get_param("gid")),
    gidx=int(get_param("gidx")),
):
    """Check that e-model directories have been created, given the attributes of a given test cell

    Attributes:
        mtype: morphological type
        etype: electrophysiological type
        region: circuit region
        gid: id of cell in the circuit
        gidx: index of cell
    """

    directories_to_be_checked = [
        "hoc_recordings",
        "python_recordings",
    ]

    memodel_files_to_be_checked = [
        "run_hoc.sh",
        "LICENSE.txt",
        "run_py.sh",
        "requirements.txt",
        "cell_info.json",
        "README.md",
    ]

    templates = [
        "cell_template_neurodamus.jinja2",
        "replace_axon_hoc.hoc",
        "createsimulation.jinja2",
        "run_hoc.jinja2",
        "synapses.jinja2",
    ]

    mechanisms = [
        "Ca_HVA.mod",
        "Ca_HVA2.mod",
        "Ca_LVAst.mod",
        "CaDynamics_DC0.mod",
        "Ih.mod",
        "K_Pst.mod",
        "K_Tst.mod",
        "KdShu2007.mod",
        "Nap_Et2.mod",
        "NaTg.mod",
        "NaTg2.mod",
        "notes.txt",
        "SK_E2.mod",
        "SKv3_1.mod",
        "StochKv2.mod",
        "StochKv3.mod",
    ]

    config_files = [
        "config_synapses.ini",
        "constants.json",
        "current_amps.json",
    ]

    synapses = ["synapses.tsv", "synconf.txt"]

    for item in config_files:
        memodel_files_to_be_checked.append(os.path.join("config", item))

    for item in templates:
        memodel_files_to_be_checked.append(os.path.join("templates", item))

    for item in mechanisms:
        memodel_files_to_be_checked.append(os.path.join("mechanisms", item))

    for item in synapses:
        memodel_files_to_be_checked.append(os.path.join("synapses", item))

    circuit_config_path = test_config["paths"]["circuit"]
    circuit = BluepyCircuit(circuit_config_path)
    cell = circuit.get_cell_attributes(gid)
    memodel_files_to_be_checked.append(
        os.path.join("morphology", f"{cell.morphology}.asc")
    )

    path_ = os.path.join("tests", "output", "memodel_dirs")
    memodel_path = os.path.join(
        path_, mtype, etype, region, "_".join([mtype, etype, str(gidx)])
    )

    for item in memodel_files_to_be_checked:
        if os.path.isfile(os.path.join(memodel_path, item)) is False:
            print("Test failed: " + os.path.join(memodel_path, item) + " not found.")
            print(f"cwd is {os.getcwd()} ")
            assert False

    for item in directories_to_be_checked:
        if os.path.isdir(os.path.join(memodel_path, item)) is False:
            print("Test failed: " + os.path.join(memodel_path, item) + " not found.")
            assert False


@launch_luigi(module="workflow", task="DoRecordings")
def test_voltages(
    mtype=get_param("mtype"),
    etype=get_param("etype"),
    region=get_param("region"),
    gid=int(get_param("gid")),
    gidx=int(get_param("gidx")),
    configfile="config_multistep_short.ini",
):
    """Test to compare the voltages produced via python and hoc.

    Attributes:
        mtype: morphological type
        etype: electrophysiological type
        region: circuit region
        gid: cell id
        gidx: index of cell
    """
    threshold = 1e-3

    inner_folder_name = combine_names(mtype, etype, gidx)
    recording_path = os.path.join(mtype, etype, region, inner_folder_name)
    script_path = os.path.join("tests", "output", "memodel_dirs", recording_path)

    for idx in range(3):
        hoc_path = os.path.join(
            script_path, "hoc_recordings", "soma_voltage_step%d.dat" % (idx + 1)
        )
        py_path = os.path.join(
            script_path, "python_recordings", "soma_voltage_step%d.dat" % (idx + 1)
        )

        hoc_voltage = np.loadtxt(hoc_path)
        py_voltage = np.loadtxt(py_path)

        rms = np.sqrt(np.mean((hoc_voltage[:, 1] - py_voltage[:, 1]) ** 2))
        assert rms < threshold


def run_bglibpy_cell(blueconfig_path, gid, sim_time, dt=0.025):
    """Run the cell in bglibpy with synapses"""
    bglibpy.set_verbose(0)

    ssim = bglibpy.SSim(blueconfig_path, record_dt=0.1)
    ssim.instantiate_gids([gid], add_synapses=True)
    cell = ssim.cells[gid]

    rng = bglibpy.neuron.h.Random(1)
    rng.uniform(50, sim_time)
    connections = []
    for _, synapse in cell.synapses.items():
        spike_train = np.array([rng.repick()])
        connection = bglibpy.Connection(synapse, spike_train, stim_dt=dt)
        connections.append(connection)

    ssim.run(sim_time, forward_skip=False, v_init=-80, dt=dt, cvode=False)

    return ssim.get_time_trace(), ssim.get_voltage_trace(gid)


@pytest.mark.usefixtures("prepare_test_synapses_config")
@launch_luigi(module="workflow", task="RunPyScript")
def test_synapses(
    mtype=get_param("mtype"),
    etype=get_param("etype"),
    region=get_param("region"),
    gid=int(get_param("gid")),
    gidx=int(get_param("gidx")),
    configfile="config_synapses.ini",
):
    """Test to compare the output of cell with synapses between our run.py and bglibpy.

    Attributes:
        mtype: morphological type
        etype: electrophysiological type
        region: circuit region
        gid: cell id
        gidx: index of cell
        configfile: the configuration file of the emodel.
    """

    threshold = 0.05

    # get circuit path for bglibpy
    circuit_config_path = test_config.get("paths", "circuit")

    # run cells from bglibpy
    sim_time = 3000
    _, bg_v = run_bglibpy_cell(circuit_config_path, gid, sim_time)

    base_path = f"memodel_dirs/{mtype}/{etype}/{region}/{mtype}_{etype}_{gidx}"

    np.savetxt(os.path.join("tests", "output", base_path, "bglibpy_voltage.dat"), bg_v)

    # load run.py output
    py_path = os.path.join(
        "tests", "output", base_path, "python_recordings", "soma_voltage_vecstim.dat"
    )
    py_v = np.loadtxt(py_path)

    # compare
    rms = np.sqrt(np.mean((bg_v - py_v[:, 1]) ** 2))
    print(rms)
    assert rms < threshold


@pytest.mark.usefixtures("prepare_test_synapses_config")
@launch_luigi(module="workflow", task="DoRecordings", reload_hoc=True)
def test_synapses_hoc_vs_py_script(
    mtype=get_param("mtype"),
    etype=get_param("etype"),
    region=get_param("region"),
    gid=int(get_param("gid")),
    gidx=int(get_param("gidx")),
    configfile="config_synapses.ini",
):
    """Test to compare the voltages produced via python and hoc.

    Attributes:
        mtype: morphological type
        etype: electrophysiological type
        region: circuit region
        gid: cell id
        gidx: index of cell
        configfile : name of config file in /config to use when running script / creating hoc
    """
    threshold = 0.1
    threshold_py_recs = 0.1

    output_path = os.path.join("tests", "output")
    memodel_path = get_output_path(mtype, etype, region, gidx, output_path)

    # load output
    hoc_path = os.path.join(memodel_path, "hoc_recordings", "soma_voltage_vecstim.dat")
    py_path = os.path.join(
        memodel_path, "python_recordings", "soma_voltage_vecstim.dat"
    )

    hoc_voltage = np.loadtxt(hoc_path)
    py_voltage = np.loadtxt(py_path)

    # check rms
    rms = np.sqrt(np.mean((hoc_voltage[:, 1] - py_voltage[:, 1]) ** 2))
    assert rms < threshold


@launch_luigi(module="workflow", task="CreateFactsheets")
def test_metype_factsheet_exists(
    mtype=get_param("mtype"),
    etype=get_param("etype"),
    region=get_param("region"),
    gid=int(get_param("gid")),
    gidx=int(get_param("gidx")),
    configfile="config_singlestep_short.ini",
):
    """Check that the me-type factsheet json file has been created.

    Attributes:
        mtype: morphological type
        etype: electrophysiological type
        region: circuit region
        gid: id of cell in the circuit
        gidx: index of cell
    """

    path_ = os.path.join("tests", "output", "memodel_dirs")
    memodel_path = os.path.join(
        path_, mtype, etype, region, "_".join([mtype, etype, str(gidx)])
    )

    metype_factsheet = os.path.join(
        memodel_path, "factsheets", "me_type_factsheeet.json"
    )
    etype_factsheet = os.path.join(memodel_path, "factsheets", "e_type_factsheeet.json")
    mtype_factsheet = os.path.join(
        memodel_path, "factsheets", "morphology_factsheeet.json"
    )
    assert os.path.isfile(metype_factsheet)
    assert os.path.isfile(etype_factsheet)
    assert os.path.isfile(mtype_factsheet)


def check_feature_mean_std(source, feat):
    """Checks that feature mean and std are equal to the original ones.

    Args:
        source (list): list of dict containing original features.
        feat (dict): feature to be checked.

    Returns True if mean and std were found in source and were equal to tested ones.
    """
    for item in source:
        if item["feature"] == feat["name"]:
            assert feat["values"][0]["mean"] == item["val"][0]
            assert feat["values"][0]["std"] == item["val"][1]
            return True
    return False


def check_features(config):
    """Checks factsheet features.

    Checks that there is no empty list or dictionary.
    Checks that features name, mean, and std are identical to the ones in feature file.
    Checks that units correspond to the ones in unit file.
    Checks that model fitnesses correspond to the ones in fitness file."""
    # original files data
    constants_path = os.path.join(
        config.get("Paths", "constants_dir"), config.get("Paths", "constants_file")
    )
    recipes_path = "/".join(
        (config.get("Paths", "recipes_dir"), config.get("Paths", "recipes_file"))
    )
    params_path = "/".join(
        (config.get("Paths", "params_dir"), config.get("Paths", "params_file"))
    )
    emodel = get_emodel(constants_path)
    recipe = get_recipe(recipes_path, emodel)
    original_feat = load_raw_exp_features(recipe)
    units = load_feature_units()
    fitness = load_fitness(params_path, emodel)
    prefix = get_prefix(recipe)
    # tested func
    feat_dict = get_exp_features_data(emodel, recipes_path, params_path)

    for items in feat_dict["values"]:
        assert items.items()
        for stim_name, stim_data in items.items():
            assert stim_data.items()
            for loc_name, loc_data in stim_data.items():
                assert loc_data
                for feat in loc_data["features"]:
                    original = original_feat[stim_name][loc_name]
                    key_fitness = ".".join((prefix, stim_name, loc_name, feat["name"]))

                    assert check_feature_mean_std(original, feat)
                    assert feat["unit"] == units[feat["name"]]
                    assert feat["model fitness"] == fitness[key_fitness]


def check_morph_name(config):
    """Checks that factsheet morph name corresponds to package morph file."""
    _, morph_fname = get_morph_path(config)
    morph_name_dict = get_morph_name_dict(morph_fname)
    assert os.path.isfile(os.path.join("morphology", morph_name_dict["value"] + ".asc"))


def get_locs_list(loc_name):
    """Return possible location list from a location name."""
    if loc_name == "all dendrites":
        return ["somadend", "alldend", "allact"]
    elif loc_name == "somatic":
        return ["somadend", "somatic", "allact", "somaxon"]
    elif loc_name == "axonal":
        return ["axonal", "allact", "somaxon"]
    elif loc_name == "apical":
        return ["apical", "allact"]
    elif loc_name == "basal":
        return ["basal", "allact"]
    return None


def get_loc_from_params(loc_name, mech_name_for_params, params):
    """Returns general location name and index of mechanism in the param dict.

    Args:
        loc_name (str): location name (dendrite, somatic, axonal)
        mech_name_for_params (str): mechanism name with the form mech_channel
            ex: "gCa_HVAbar_Ca_HVA2"
        params (dict): dictionary in which the mech is searched

    Returns:
        new_loc_name (str): general location name under which the channel can be found
            (somadend, alldend, somatic, axonal, allact, somaxon, apical, basal)
        idx (int): index of the channel in the list under location key
    """
    locs = get_locs_list(loc_name)

    for loc in locs:
        if loc in params.keys():
            for i, param in enumerate(params[loc]):
                if param["name"] == mech_name_for_params:
                    return loc, i

    return "", 0


def check_mechanisms(config):
    """Checks factsheet mechanisms.

    Checks that there is no empty list or dict.
    Checks that 'type' is either exponential or decay or uniform.
    Checks that if type is exponential,
        there is an according ['dist']='exp' field in parameter file.
    Idem if type is decay
    Checks that all values are identical to files.
    """
    # pylint: disable=too-many-locals
    # original data
    constants_path = os.path.join(
        config.get("Paths", "constants_dir"), config.get("Paths", "constants_file")
    )
    params_path = "/".join(
        (config.get("Paths", "params_dir"), config.get("Paths", "params_file"))
    )
    emodel = get_emodel(constants_path)
    recipes_path = "/".join(
        (config.get("Paths", "recipes_dir"), config.get("Paths", "recipes_file"))
    )
    params_filepath = find_param_file(recipes_path, emodel)
    definitions = json_load(params_filepath)
    parameters = definitions["parameters"]
    release_params = get_release_params(emodel)

    # output to check
    mech_dict = get_mechanisms_data(emodel, params_path, params_filepath)

    assert mech_dict["values"][0]["location_map"]
    for loc_name, loc in mech_dict["values"][0]["location_map"].items():
        assert loc["channels"]
        for channel_name, channel in loc["channels"].items():
            assert channel["equations"]
            for mech_name, mech in channel["equations"].items():
                mech_name_for_params = "_".join((mech_name, channel_name))
                new_loc_name, idx = get_loc_from_params(
                    loc_name, mech_name_for_params, parameters
                )
                mech_name_for_final_params = ".".join(
                    (mech_name_for_params, new_loc_name)
                )
                if mech["type"] == "exponential":
                    assert "dist" in parameters[new_loc_name][idx]
                    assert parameters[new_loc_name][idx]["dist"] == "exp"
                    assert (
                        str(release_params[mech_name_for_final_params]) in mech["plot"]
                    )
                    assert (
                        str(release_params[mech_name_for_final_params]) in mech["latex"]
                    )
                elif mech["type"] == "decay":
                    assert "dist" in parameters[new_loc_name][idx]
                    assert parameters[new_loc_name][idx]["dist"] == "decay"
                    assert (
                        str(release_params[mech_name_for_final_params]) in mech["plot"]
                    )
                    assert (
                        str(release_params[mech_name_for_final_params]) in mech["latex"]
                    )
                    assert (
                        str(release_params["constant.distribution_decay"])
                        in mech["plot"]
                    )
                    assert (
                        str(release_params["constant.distribution_decay"])
                        in mech["latex"]
                    )
                else:
                    assert mech["type"] == "uniform"
                    assert release_params[mech_name_for_final_params] == mech["latex"]
                    assert release_params[mech_name_for_final_params] == mech["plot"]


def check_anatomy(config):
    """Checks that all anatomy data is positive and exists.

    Fields include axon and soma.
    Fields can either include basal and apical, or just dendrite.

    Checks that there is no empty list or dict.
    Checks that data exists and is a float/int and is positive.
    Checks that there is no anatomy field missing.
    """
    morph_dir, morph_fname = get_morph_path(config)
    ana_dict = get_morph_data(os.path.join(morph_dir, morph_fname))
    left_to_check_1 = [
        "total axon length",
        "total axon volume",
        "axon maximum branch order",
        "axon maximum section length",
        "total apical length",
        "total apical volume",
        "apical maximum branch order",
        "apical maximum section length",
        "total basal length",
        "total basal volume",
        "basal maximum branch order",
        "basal maximum section length",
        "soma diameter",
    ]
    left_to_check_2 = [
        "total axon length",
        "total axon volume",
        "axon maximum branch order",
        "axon maximum section length",
        "total dendrite length",
        "total dendrite volume",
        "dendrite maximum branch order",
        "dendrite maximum section length",
        "soma diameter",
    ]
    lists_to_check = [left_to_check_1, left_to_check_2]

    assert ana_dict["values"]
    for item in ana_dict["values"]:
        assert isinstance(item["value"], (float, np.floating, int, np.integer))
        assert item["value"] > 0

        for l in lists_to_check:
            if item["name"] in l:
                l.remove(item["name"])

    assert len(lists_to_check[0]) == 0 or len(lists_to_check[1]) == 0


def check_physiology(config):
    """Checks that all physiology values exist.

    Checks that there is no empty list or dict.
    Checks that data exists and is a float and is positive (except for membrane pot.).
    Checks that there is no physiology field missing.
    """
    # get current amplitude
    amps_path = os.path.join(
        config.get("Paths", "protocol_amplitudes_dir"),
        config.get("Paths", "protocol_amplitudes_file"),
    )
    step_number = config.getint("Protocol", "run_step_number")
    amps, _ = load_amps(amps_path)
    current_amplitude = amps[step_number - 1]

    # get parameters from config
    stim_start = config.getint("Protocol", "stimulus_delay")
    stim_duration = config.getint("Protocol", "stimulus_duration")

    # get data path from run.py output
    fname = "soma_voltage_step{}.dat".format(step_number)
    data_path = os.path.join("python_recordings", fname)

    # load time, voltage
    data = np.loadtxt(data_path)

    phys_dict = get_physiology_data(
        time=data[:, 0],
        voltage=data[:, 1],
        current_amplitude=current_amplitude,
        stim_start=stim_start,
        stim_duration=stim_duration,
    )
    left_to_check = [
        "input resistance",
        "membrane time constant",
        "resting membrane potential",
    ]

    assert phys_dict["values"]
    for item in phys_dict["values"]:
        assert isinstance(item["value"], float)
        if item["name"] in ["input resistance", "membrane time constant"]:
            assert item["value"] >= 0
        left_to_check.remove(item["name"])

    assert len(left_to_check) == 0


@launch_luigi(module="workflow", task="RunPyScript")
def test_factsheets_fcts(
    mtype=get_param("mtype"),
    etype=get_param("etype"),
    region=get_param("region"),
    gid=int(get_param("gid")),
    gidx=int(get_param("gidx")),
    configfile="config_singlestep_short.ini",
):
    """Test dictionary output from functions used for factsheets."""
    path_ = os.path.join("tests", "output", "memodel_dirs")
    memodel_path = os.path.join(
        path_, mtype, etype, region, "_".join([mtype, etype, str(gidx)])
    )

    with cwd(memodel_path):
        config = load_config(config_dir="config", filename=configfile)
        check_features(config)
        check_morph_name(config)
        check_mechanisms(config)
        check_physiology(config)
        check_anatomy(config)
