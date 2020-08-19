"""Test file."""
import os
import numpy as np
from tests.decorators import launch_luigi
from e_model_packages.sscx2020.utils import (
    get_morph_emodel_names,
    combine_names,
)


@launch_luigi(module="workflow", task="PrepareMEModelDirectory")
def test_directory_exists(mtype="L1_DAC", etype="bNAC", gid=4, gidx=1):
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
        "createsimulation.hoc",
        "current_amps.dat",
        "run_hoc.sh",
        "run.hoc",
    ]

    output_files_to_be_checked = [
        "LICENSE.txt",
        "run_py.sh",
        "run.py",
        "run_old_py.sh",
        "old_run.py",
        "load.py",
        "myrecordings.py",
        "mymorphology.py",
        "create_hoc.py",
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

    output_files_to_be_checked.append(os.path.join("config", "recipes", "recipes.json"))
    output_files_to_be_checked.append(os.path.join("config", "config_example.ini"))
    output_files_to_be_checked.append(os.path.join("config", "config.ini"))

    for item in templates:
        output_files_to_be_checked.append(os.path.join("templates", item))

    for item in py_rec_config:
        output_files_to_be_checked.append(os.path.join("config", "params", item))

    for item in mechanisms:
        memodel_files_to_be_checked.append(os.path.join("mechanisms", item))

    morph_fname, _ = get_morph_emodel_names(
        os.path.join("e_model_packages", "sscx2020"), gid
    )
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
def test_voltages(mtype="L1_DAC", etype="bNAC", gidx=1):
    """Test to compare the voltages produced via python and hoc.

    Attributes:
        mtype: morphological type
        etype: electrophysiological type
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
