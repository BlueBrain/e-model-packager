import configparser
import os
import sys
from tests.decorators import launch_luigi, erase_output
from e_model_packages.sscx2020.utils import (
    read_circuit,
    get_mecombo_emodels,
    get_morph_emodel_names,
)


@erase_output
@launch_luigi(module="workflow", task="PrepareMEModelDirectory")
def test_directory_exists(mtype="L1_DAC", etype="bNAC", gid=1, gidx=1):
    """Check that e-model directories have been created, given the attributes of a given test cell

    Attributes:
        mtype: morphological type
        etype: electrophysiological type
        gid: id of cell in the circuit
        gidx: index of cell
    """

    directories_to_be_checked = ["hoc_recordings", "python_recordings"]

    files_to_be_checked = [
        "constants.hoc",
        "createsimulation.hoc",
        "current_amps.dat",
        "LICENSE.txt",
        "run_hoc.sh",
        "run_py.sh",
        "run.hoc",
        "run.py",
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

    for item in mechanisms:
        files_to_be_checked.append(os.path.join("mechanisms", item))

    morph_fname, emodel_fname = get_morph_emodel_names(
        os.path.join("e_model_packages", "sscx2020"), gid
    )

    files_to_be_checked.append(os.path.join("morphology", morph_fname))
    files_to_be_checked.append(emodel_fname)

    path_ = os.path.join("e_model_packages", "sscx2020", "output", "memodel_dirs")
    path = os.path.join(path_, mtype, etype, "_".join([mtype, etype, str(gidx)]))

    for item in files_to_be_checked:
        if os.path.isfile(os.path.join(path, item)) is False:
            print("Test failed: " + os.path.join(path, item) + " not found.")
            assert False

    for item in directories_to_be_checked:
        if os.path.isdir(os.path.join(path, item)) is False:
            print("Test failed: " + os.path.join(path, item) + " not found.")
            assert False
