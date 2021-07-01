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

test_config = configparser.ConfigParser()
test_config.read(os.path.join("tests", "luigi_test.cfg"))
get_param = partial(test_config.get, "params")


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
        "compile_mechanisms.sh",
        "requirements.txt",
        "cell_info.json",
        "README.md",
        "LICENSE_CC-BY-CA-SA-4.0",
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


@launch_luigi(module="workflow", task="RunPyScript")
def test_synapses(
    mtype=get_param("mtype"),
    etype=get_param("etype"),
    region=get_param("region"),
    gid=int(get_param("gid")),
    gidx=int(get_param("gidx")),
    configfile="config_synapses_short.ini",
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
    sim_time = 600
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


@launch_luigi(module="workflow", task="DoRecordings", reload_hoc=True)
def test_synapses_hoc_vs_py_script(
    mtype=get_param("mtype"),
    etype=get_param("etype"),
    region=get_param("region"),
    gid=int(get_param("gid")),
    gidx=int(get_param("gidx")),
    configfile="config_synapses_short.ini",
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
        memodel_path, "factsheets", "me_type_factsheet.json"
    )
    etype_factsheet = os.path.join(memodel_path, "factsheets", "e_model_factsheet.json")
    assert os.path.isfile(metype_factsheet)
    assert os.path.isfile(etype_factsheet)
