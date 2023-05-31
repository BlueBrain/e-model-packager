import os

import h5py
import numpy as np
import subprocess

from tests.decorators import launch_luigi_synaptic_plasticity

from e_model_packages.utils import cwd


@launch_luigi_synaptic_plasticity(module="workflow", task="PrepareMEModelDirectory")
def test_directory_exists(memodel_dir=None):
    """Check that cell directories have been created."""

    memodel_files_to_be_checked = [
        "run.sh",
        "run_pairsim.sh",
        "run_all.sh",
        "run_all_pairsim.sh",
        "LICENSE.txt",
        "requirements.txt",
        "README.md",
    ]

    mechanisms = [
        "Ca_HVA2.mod",
        "Ca_LVAst.mod",
        "CaDynamics_DC0.mod",
        "GluSynapse.mod",
        "Ih.mod",
        "K_Pst.mod",
        "K_Tst.mod",
        "Nap_Et2.mod",
        "NaTg.mod",
        "SK_E2.mod",
        "SKv3_1.mod",
        "StochKv3.mod",
        "VecStim.mod",
    ]

    config_files = [
        "config_10Hz_10ms.ini",
        "config_1Hz_10ms.ini",
        "config_20Hz_10ms.ini",
        "config_50Hz_10ms.ini",
        "params/final.json",
        "params/pyr.json",
    ]

    protocols = [
        "spiketrain_10Hz_10ms.dat",
        "spiketrain_1Hz_10ms.dat",
        "spiketrain_20Hz_10ms.dat",
        "spiketrain_50Hz_10ms.dat",
        "stimuli_10Hz.json",
        "stimuli_1Hz.json",
        "stimuli_20Hz.json",
        "stimuli_50Hz.json",
    ]

    synapses = [
        "synapses.tsv",
        "synconf.txt",
        "cpre_cpost.json",
        "synapse_properties.json",
        "syn_extra_params.json",
        "mtype_map.tsv",
    ]

    for item in config_files:
        memodel_files_to_be_checked.append(os.path.join("config", item))

    for item in mechanisms:
        memodel_files_to_be_checked.append(os.path.join("mechanisms", item))

    for item in protocols:
        memodel_files_to_be_checked.append(os.path.join("protocols", item))

    for item in synapses:
        memodel_files_to_be_checked.append(os.path.join("synapses", item))

    for item in memodel_files_to_be_checked:
        if os.path.isfile(os.path.join(memodel_dir, item)) is False:
            print("Test failed: " + os.path.join(memodel_dir, item) + " not found.")
            print(f"cwd is {os.getcwd()} ")
            assert False


@launch_luigi_synaptic_plasticity(module="workflow", task="RunPyScript")
def test_voltage_trace(memodel_dir=None, original_path=None):
    """Compare that the voltage trace of a random cell with the original one."""
    # Note that the args are set in the decorator.
    new_path = os.path.join(memodel_dir, "output_50Hz_10ms.h5")

    threshold_v = 1.5
    threshold_other = 1.0

    with h5py.File(original_path, "r") as original:
        with h5py.File(new_path, "r") as new:
            original_t = original["t"][()]
            new_t = new["t"][()]
            for key, data in original.items():
                if key == "prespikes":
                    assert np.all(data[()] == new[key][()])
                elif key == "v":
                    new_v = np.interp(original_t, new_t, new[key][()])
                    rms = np.sqrt(np.mean((data[()] - new_v) ** 2))
                    assert rms < threshold_v
                elif key != "t":
                    for i in range(len(data[()][0])):
                        new_data = np.interp(original_t, new_t, new[key][()][:, i])
                        rms = np.sqrt(np.mean((data[()][:, i] - new_data) ** 2))
                        assert rms < threshold_other

            for key, elem in original.attrs.items():
                assert np.all(elem[()] == new.attrs[key][()])


@launch_luigi_synaptic_plasticity(module="workflow", task="PrecellConfig")
def test_voltage_pairsim(memodel_dir=None, original_path=None):
    """Compare that the voltage trace of a random cell with the original one."""
    # Note that the args are set in the decorator.
    new_path = os.path.join(memodel_dir, "output_50Hz_10ms.h5")

    # there is a bit of a precision lost
    # due to the approximate timing of the precell spiking
    threshold_v = 1.5
    threshold_other = 10.0

    if os.path.isfile(new_path):
        os.remove(new_path)

    with cwd(memodel_dir):
        subprocess.call(["sh", "run_pairsim.sh", "config/config_50Hz_10ms.ini"])

    with h5py.File(original_path, "r") as original:
        with h5py.File(new_path, "r") as new:
            original_t = original["t"][()]
            new_t = new["t"][()]
            for key, data in original.items():
                if key == "prespikes":
                    assert np.all(data[()] == new[key][()])
                elif key == "v":
                    new_v = np.interp(original_t, new_t, new[key][()])
                    rms = np.sqrt(np.mean((data[()] - new_v) ** 2))
                    assert rms < threshold_v
                elif key != "t":
                    for i in range(len(data[()][0])):
                        new_data = np.interp(original_t, new_t, new[key][()][:, i])
                        rms = np.sqrt(np.mean((data[()][:, i] - new_data) ** 2))
                        assert rms < threshold_other

            for key, elem in original.attrs.items():
                assert np.all(elem[()] == new.attrs[key][()])
