"""Test file."""
import configparser
import os
import random
import re
import subprocess
import numpy as np

import bglibpy
from tests.decorators import launch_luigi
from e_model_packages.sscx2020.utils import (
    get_morph_emodel_names,
    combine_names,
    cwd,
)


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
        "constants.hoc",
        "current_amps.dat",
        "run_hoc.sh",
    ]

    output_files_to_be_checked = [
        "LICENSE.txt",
        "run_py.sh",
        "run.py",
        "run_old_py.sh",
        "old_run.py",
        "load.py",
        "recordings.py",
        "morphology.py",
        "synapse.py",
        "cell.py",
        "create_hoc.py",
        "create_hoc_tools.py",
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

    config_files = ["config_example.ini", "config.ini", "config_synapses.ini"]

    synapses = ["synapses.tsv", "synconf.txt"]

    output_files_to_be_checked.append(os.path.join("config", "recipes", "recipes.json"))
    for item in config_files:
        output_files_to_be_checked.append(os.path.join("config", item))

    for item in templates:
        output_files_to_be_checked.append(os.path.join("templates", item))

    for item in py_rec_config:
        output_files_to_be_checked.append(os.path.join("config", "params", item))

    for item in mechanisms:
        memodel_files_to_be_checked.append(os.path.join("mechanisms", item))

    for item in synapses:
        memodel_files_to_be_checked.append(os.path.join("synapses", item))

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
            assert False

    for item in directories_to_be_checked:
        if os.path.isdir(os.path.join(memodel_path, item)) is False:
            print("Test failed: " + os.path.join(memodel_path, item) + " not found.")
            assert False

    output_path = os.path.join("tests", "output")

    for item in output_files_to_be_checked:
        if os.path.isfile(os.path.join(output_path, item)) is False:
            print("Test failed: " + os.path.join(output_path, item) + " not found.")
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

    connections = []
    random.seed(1)
    for _, synapse in cell.synapses.items():
        spike_train = np.array([random.uniform(50, sim_time)])
        connection = bglibpy.Connection(synapse, spike_train, stim_dt=dt)
        connections.append(connection)

    ssim.run(sim_time, forward_skip=False, v_init=-80, dt=dt, cvode=False)

    return ssim.get_time(), ssim.get_voltage_traces()[gid]


@launch_luigi(module="workflow", task="PrepareMEModelDirectory")
def test_synapses(mtype="L23_BP", etype="bNAC", gid=111728, gidx=150):
    """Test to compare the output of cell with synapses between our run.py and bglibpy.

    Attributes:
        mtype: morphological type
        etype: electrophysiological type
        gid: cell id
        gidx: index of cell
    """

    threshold = 0.05

    # get circuit path for bglibpy
    config_circuit = configparser.ConfigParser()
    config_circuit.read(os.path.join("tests", "luigi_test.cfg"))
    circuit_config_path = config_circuit.get("paths", "circuit")

    # run run.py
    os.chdir("tests/output")
    base_path = f"memodel_dirs/{mtype}/{etype}/{mtype}_{etype}_{gidx}"
    path_to_mechanisms = os.path.join(base_path, "mechanisms")
    subprocess.call(["nrnivmodl", path_to_mechanisms])
    subprocess.call(["python", "run.py", "--c", "config_synapses.ini"])

    # run cells from bglibpy
    sim_time = 3000
    _, bg_v = run_bglibpy_cell(circuit_config_path, gid, sim_time)

    # load run.py output
    py_path = os.path.join(base_path, "python_recordings", "soma_voltage_vecstim.dat")
    py_v = np.loadtxt(py_path)

    os.chdir("../..")

    # compare
    rms = np.sqrt(np.mean((bg_v - py_v[:, 1]) ** 2))
    print(rms)
    assert rms < threshold


@launch_luigi(module="workflow", task="CreateHoc", reload_hoc=True)
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

    # re-write config file to have consistent randomness accross simulations
    new_config_file = "config_synapses_test.ini"
    config_path = os.path.join(output_path, "config", configfile)
    with open(config_path, "r") as config_f:
        config = config_f.read()
    new_config = re.sub("vecstim_random=.*", "vecstim_random=neuron", config)

    new_config_path = os.path.join(output_path, "config", new_config_file)
    with open(new_config_path, "w") as new_f:
        new_f.write(new_config)

    # get output path
    inner_folder_name = combine_names(mtype, etype, gidx)
    recording_path = os.path.join(mtype, etype, inner_folder_name)
    script_path = os.path.join(output_path, "memodel_dirs", recording_path)

    # run scripts
    with cwd(script_path):
        subprocess.call(["sh", "./run_hoc.sh"])
    with cwd(output_path):
        subprocess.call(["sh", "./run_py.sh", new_config_file])
        subprocess.call(["sh", "./run_old_py.sh", new_config_file])

    # load output
    hoc_path = os.path.join(script_path, "hoc_recordings", "soma_voltage_vecstim.dat")
    py_path = os.path.join(script_path, "python_recordings", "soma_voltage_vecstim.dat")
    old_py_path = os.path.join(
        script_path, "old_python_recordings", "soma_voltage_vecstim.dat"
    )

    hoc_voltage = np.loadtxt(hoc_path)
    py_voltage = np.loadtxt(py_path)
    old_py_voltage = np.loadtxt(old_py_path)

    # check rms
    rms = np.sqrt(np.mean((hoc_voltage[:, 1] - py_voltage[:, 1]) ** 2))
    rms_py_recs = np.sqrt(np.mean((old_py_voltage[:, 1] - py_voltage[:, 1]) ** 2))
    assert rms < threshold
    assert rms_py_recs < threshold_py_recs
