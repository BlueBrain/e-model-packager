"""Contains tests for the workflow."""
# pylint: disable=wrong-import-position
# pylint: disable=wrong-import-order
# pylint: disable=import-error
import configparser
import os
import re
import numpy as np
import pytest
import sys
from functools import partial

import bglibpy
from tests.decorators import launch_luigi
from e_model_packages.sscx2020.utils import (
    get_morph_emodel_names,
    get_output_path,
    combine_names,
    cwd,
)

sys.path.append(os.path.join("e_model_packages", "sscx2020", "extra_data", "scripts"))
from load import load_config
from write_factsheets import (
    get_morph_data,
    get_physiology_data,
    get_morph_name,
    get_emodel,
    get_recipe,
    get_prefix,
    get_exp_features_data,
    get_mechanisms_data,
    load_raw_exp_features,
    load_feature_units,
    load_fitness,
    get_param_data,
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
        "run.py",
        "load.py",
        "recordings.py",
        "morphology.py",
        "synapse.py",
        "cell.py",
        "create_hoc.py",
        "create_hoc_tools.py",
        "GUI.py",
        "requirements.txt",
        "cell_info.json",
        "write_factsheets.py",
        "README.md",
        "create_cells.py",
        "protocols.py",
        "locations.py",
    ]

    templates = [
        "cell_template_neurodamus.jinja2",
        "replace_axon_hoc.hoc",
    ]

    py_rec_config = [
        "final.json",
        "int_delayed.json",
        "int_delayed_noise.json",
        "int.json",
        "int_noise.json",
        "pyr.json",
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

    GUI_files = [
        "interface.py",
        "style.py",
        "frames.py",
        "plotshape.py",
        "simulator.py",
    ]

    memodel_files_to_be_checked.append(
        os.path.join("config", "recipes", "recipes.json")
    )
    for item in config_files:
        memodel_files_to_be_checked.append(os.path.join("config", item))

    for item in templates:
        memodel_files_to_be_checked.append(os.path.join("templates", item))

    for item in py_rec_config:
        memodel_files_to_be_checked.append(os.path.join("config", "params", item))

    for item in mechanisms:
        memodel_files_to_be_checked.append(os.path.join("mechanisms", item))

    for item in synapses:
        memodel_files_to_be_checked.append(os.path.join("synapses", item))

    for item in GUI_files:
        memodel_files_to_be_checked.append(os.path.join("GUI_utils", item))

    morph_fname, _ = get_morph_emodel_names(gid, test_config)
    memodel_files_to_be_checked.append(os.path.join("morphology", morph_fname))

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
    threshold_py_recs = 1e-8

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
        old_py_path = os.path.join(
            script_path, "old_python_recordings", "soma_voltage_step%d.dat" % (idx + 1)
        )

        hoc_voltage = np.loadtxt(hoc_path)
        py_voltage = np.loadtxt(py_path)
        old_py_voltage = np.loadtxt(old_py_path)

        rms = np.sqrt(np.mean((hoc_voltage[:, 1] - py_voltage[:, 1]) ** 2))
        rms_py_recs = np.sqrt(np.mean((old_py_voltage[:, 1] - py_voltage[:, 1]) ** 2))
        assert rms < threshold
        assert rms_py_recs < threshold_py_recs


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
    old_py_path = os.path.join(
        memodel_path, "old_python_recordings", "soma_voltage_vecstim.dat"
    )

    hoc_voltage = np.loadtxt(hoc_path)
    py_voltage = np.loadtxt(py_path)
    old_py_voltage = np.loadtxt(old_py_path)

    # check rms
    rms = np.sqrt(np.mean((hoc_voltage[:, 1] - py_voltage[:, 1]) ** 2))
    rms_py_recs = np.sqrt(np.mean((old_py_voltage[:, 1] - py_voltage[:, 1]) ** 2))
    assert rms < threshold
    assert rms_py_recs < threshold_py_recs


@launch_luigi(module="workflow", task="CreateMETypeJson")
def test_metype_factsheet_exists(
    mtype=get_param("mtype"),
    etype=get_param("etype"),
    region=get_param("region"),
    gid=int(get_param("gid")),
    gidx=int(get_param("gidx")),
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
    emodel = get_emodel(config)
    recipe = get_recipe(config, emodel)
    original_feat = load_raw_exp_features(recipe)
    units = load_feature_units()
    fitness = load_fitness(config, emodel)
    prefix = get_prefix(config, recipe)
    # tested func
    feat_dict = get_exp_features_data(config)

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
    morph_name_dict = get_morph_name(config)
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
    # original data
    release_params, parameters, _, _ = get_param_data(config)
    # output to check
    mech_dict = get_mechanisms_data(config)

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
    ana_dict = get_morph_data(config)
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
        assert isinstance(item["value"], (float, int, np.integer))
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
    phys_dict = get_physiology_data(config)
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
    run_single_step=True,
):
    """Test dictionary output from functions used for factsheets."""
    path_ = os.path.join("tests", "output", "memodel_dirs")
    memodel_path = os.path.join(
        path_, mtype, etype, region, "_".join([mtype, etype, str(gidx)])
    )
    config = load_config()

    with cwd(memodel_path):
        check_features(config)
        check_morph_name(config)
        check_mechanisms(config)
        check_physiology(config)
        check_anatomy(config)
