"""Test file."""
import configparser
import os
import re
import numpy as np
import pytest

import bglibpy
from tests.decorators import launch_luigi
from e_model_packages.sscx2020.utils import (
    get_morph_emodel_names,
    get_output_path,
    combine_names,
)


@pytest.fixture(scope="session")
def prepare_test_synapses_config():
    """Prepares a test config with synapses that uses neuron's rng."""
    mtype = "L23_BP"
    etype = "bNAC"
    gidx = 150
    configfile = "config_synapses.ini"

    output_path = os.path.join("tests", "output")
    memodel_path = get_output_path(mtype, etype, gidx, output_path)

    # re-write config file to have consistent randomness accross simulations
    config_path = os.path.join(memodel_path, "config", configfile)
    with open(config_path, "r") as config_f:
        config = config_f.read()
    new_config = re.sub("vecstim_random=.*", "vecstim_random=neuron", config)
    with open(config_path, "w") as config_f:
        config_f.write(new_config)


@launch_luigi(module="workflow", task="PrepareMEModelDirectory")
def test_directory_exists(mtype="L23_BP", etype="bNAC", gid=111728, gidx=150):
    """Check that e-model directories have been created, given the attributes of a given test cell

    Attributes:
        mtype: morphological type
        etype: electrophysiological type
        gid: id of cell in the circuit
        gidx: index of cell
    """

    directories_to_be_checked = [
        "hoc_recordings",
        "python_recordings",
        "old_python_recordings",
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
        "config.ini",
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

    config = configparser.ConfigParser()
    config.read(os.path.join("tests", "luigi_test.cfg"))
    morph_fname, _ = get_morph_emodel_names(gid, config)
    memodel_files_to_be_checked.append(os.path.join("morphology", morph_fname))

    path_ = os.path.join("tests", "output", "memodel_dirs")
    memodel_path = os.path.join(
        path_, mtype, etype, "_".join([mtype, etype, str(gidx)])
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
def test_voltages(mtype="L23_BP", etype="bNAC", gid=111728, gidx=150):
    """Test to compare the voltages produced via python and hoc.

    Attributes:
        mtype: morphological type
        etype: electrophysiological type
        gid: cell id
        gidx: index of cell
    """
    threshold = 1e-3
    threshold_py_recs = 1e-8

    inner_folder_name = combine_names(mtype, etype, gidx)
    recording_path = os.path.join(mtype, etype, inner_folder_name)
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
    mtype="L23_BP", etype="bNAC", gid=111728, gidx=150, configfile="config_synapses.ini"
):
    """Test to compare the output of cell with synapses between our run.py and bglibpy.

    Attributes:
        mtype: morphological type
        etype: electrophysiological type
        gid: cell id
        gidx: index of cell
        configfile: the configuration file of the emodel.
    """

    threshold = 0.05

    # get circuit path for bglibpy
    config_circuit = configparser.ConfigParser()
    config_circuit.read(os.path.join("tests", "luigi_test.cfg"))
    circuit_config_path = config_circuit.get("paths", "circuit")

    # run cells from bglibpy
    sim_time = 3000
    _, bg_v = run_bglibpy_cell(circuit_config_path, gid, sim_time)

    base_path = f"memodel_dirs/{mtype}/{etype}/{mtype}_{etype}_{gidx}"

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
    mtype="L23_BP", etype="bNAC", gid=111728, gidx=150, configfile="config_synapses.ini"
):
    """Test to compare the voltages produced via python and hoc.

    Attributes:
        mtype: morphological type
        etype: electrophysiological type
        gid: cell id
        gidx: index of cell
        configfile : name of config file in /config to use when running script / creating hoc
    """
    threshold = 0.1
    threshold_py_recs = 0.1

    output_path = os.path.join("tests", "output")
    memodel_path = get_output_path(mtype, etype, gidx, output_path)

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
