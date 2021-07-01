"""Functions to extract data needed for the glusynapse package."""
import json
import logging
import shutil
import os

import h5py
import bglibpy
from glusynapseutils.simulation.epg import ParamsGenerator
from glusynapseutils.simulation.simulator import (
    c_pre_finder,
    c_post_finder,
)
from bluepy_configfile.configfile import BlueConfigFile

# merge this into e_model_packages.circuit.synapse_extractor
from e_model_packages.circuit.glusynapse_extractor import GluSynapseExtractor
from e_model_packages.circuit import BluepyCircuit, BluepySimulation
from e_model_packages.sscx2020.utils import NpEncoder

# Configure logger
logger = logging.getLogger(__name__)


def extract_morph_data(
    output_dir, circuit, simulation, postgid, output_morph_dir="morphology"
):
    """Extract morphology."""
    # get cell data
    cell = circuit.get_cell_attributes(postgid)

    # copy morphology
    src = os.path.join(simulation.morph_dir, cell.morphology_fname)
    dst = os.path.join(output_dir, output_morph_dir, cell.morphology_fname)
    shutil.copyfile(src, dst)


def extract_constants(
    output_dir,
    circuit,
    postgid,
    bc_dict,
    out_file="constants.json",
    out_const_dir="config",
    celsius=34,
    v_init=-65,
    fastforward=None,
    invivo=False,
    fit_params=None,
    default_synrec=None,
):
    """Extract constants."""
    # pylint: disable=too-many-arguments, too-many-locals
    # get base_seed, dt and duration from BlueConfig
    base_seed = bc_dict["Run"]["Default"]["BaseSeed"]
    dt = bc_dict["Run"]["Default"]["Dt"]
    tstop = bc_dict["Run"]["Default"]["Duration"]

    # get constants
    cell = circuit.get_cell_attributes(postgid)
    cell_emodel = circuit.get_emodel_attributes(postgid)
    constants = {
        "celsius": celsius,
        "v_init": v_init,
        "morph_fname": cell.morphology_fname,
        "emodel": cell_emodel.name,
        "gid": int(postgid),
        "fastforward": fastforward,
        "invivo": invivo,
        "fit_params": fit_params,
        "base_seed": base_seed,
        "dt": dt,
        "tstop": tstop,
        "synrec": default_synrec,
    }

    # write constants
    output_path = os.path.join(output_dir, out_const_dir, out_file)
    with open(output_path, "w") as f:
        json.dump(constants, f)


# adapted from glusynapseutils.simulator._runconnectedpair_process
# adapted from e_model_packages.sscx2020.workflow.PrepareMEModelDirectory.run
def extract_synapses_data(output_dir, bcpath, syn_dir="synapses"):
    """Extract synapse data."""
    # set extractor
    syn_ext = GluSynapseExtractor(bcpath, None)

    # get gids
    postgid = syn_ext.get_postgid()
    pregids = syn_ext.get_pregids()
    # set gid
    syn_ext.set_gid(postgid)

    # extract synapse data
    syn_ext.load_synapses(
        add_stimuli=True,
        add_synapses=True,
        intersect_pre_gids=pregids,
    )

    # write data
    synapses_dir = os.path.join(output_dir, syn_dir)

    synapse_tsv_filename = os.path.join(synapses_dir, "synapses.tsv")
    with open(synapse_tsv_filename, "w") as synapse_tsv_file:
        synapse_tsv_file.write(syn_ext.synapse_tsv_content)

    mtype_filename = os.path.join(synapses_dir, "mtype_map.tsv")
    with open(mtype_filename, "w") as mtype_file:
        mtype_file.write(syn_ext.mtype_map_content)

    synconf_filename = os.path.join(synapses_dir, "synconf.txt")
    with open(synconf_filename, "w") as synconf_file:
        synconf_file.write(syn_ext.synconf)


def fix_tuple_keys_for_json(dict_to_fix):
    """Replace tuple keys by str: (a,b) -> '(a,b)'."""
    return {str(k): v for k, v in dict_to_fix.items()}


def fix_Dep_and_Fac_TM(basedir, syn_extra_params):
    """Take Dep_TM and Fac_TM values from original simulation."""
    # pylint: disable=unsubscriptable-object
    results_path = os.path.join(basedir, "simulation.h5")

    with h5py.File(results_path, "r") as original_results:
        original_Dep_TM = original_results.attrs["Dep_TM"][()]
        original_Fac_TM = original_results.attrs["Fac_TM"][()]
        for i, synapse in enumerate(syn_extra_params):
            syn_extra_params[synapse]["Dep_TM"] = original_Dep_TM[i]
            syn_extra_params[synapse]["Fac_TM"] = original_Fac_TM[i]

    return syn_extra_params


# adapted from glusynapseutils.simulation.simulator._runconnectedpair_process
def extract_syn_extra_params(
    output_dir,
    ssim,
    circuitpath,
    extra_recipe,
    basedir,
    syn_dir="synapses",
    output_file="syn_extra_params.json",
):
    """Extract synapse extra params."""
    # pylint: disable=protected-access
    # load postgid & pregids
    postgid = list(ssim.bc_circuit.cells.ids("PostCell"))[0]
    pregid = list(ssim.bc_circuit.cells.ids("PreCell"))[0]
    pregids = [pregid]
    # Special case: multiple connections
    if (
        "ExtraPreCell" in ssim.bc_circuit.cells._targets._targets
        or "ExtraPreCell" in ssim.bc_circuit.cells._targets._resolve_cache
    ):
        pregids = pregids + list(ssim.bc_circuit.cells.ids("ExtraPreCell"))

    # Generate supplementary model parameters
    pgen = ParamsGenerator(circuitpath, extra_recipe)
    syn_extra_params = {}
    for pg in pregids:
        syn_extra_params.update(pgen.generate_params(pg, postgid))

    # For some reason, Dep_TM and Fac_TM are close, but not exactly the same as in
    # the output of the original simulation, so there are fixed here.
    syn_extra_params = fix_Dep_and_Fac_TM(basedir, syn_extra_params)

    # replace tuple keys by str: (a,b) -> "(a,b)"
    syn_extra_params = fix_tuple_keys_for_json(syn_extra_params)

    # write syn_extra_params on file
    output_file = os.path.join(output_dir, syn_dir, output_file)
    with open(output_file, "w") as f:
        json.dump(syn_extra_params, f, indent=4)


def extract_cpre_and_cpost(
    output_dir,
    basedir,
    fit_params,
    invivo=False,
    syn_dir="synapses",
    output_file="cpre_cpost.json",
):
    """Extract c_pre and c_post."""
    c_pre_tuple_keys = c_pre_finder(basedir, fit_params, invivo)
    c_post_tuple_keys = c_post_finder(basedir, fit_params, invivo)

    c_pre = fix_tuple_keys_for_json(c_pre_tuple_keys)
    c_post = fix_tuple_keys_for_json(c_post_tuple_keys)

    # -- write syn_extra_params on file
    output_path = os.path.join(output_dir, syn_dir, output_file)
    with open(output_path, "w") as f:
        json.dump({"c_pre": c_pre, "c_post": c_post}, f, indent=4)


def extract_protocols(
    output_dir, bc_dict, prot_dir="protocols", output_file="stimuli.json"
):
    """Extract Pulse Stimuli from BlueConfig."""
    stimuli_dict = {}
    for key, item in bc_dict["Stimulus"].items():
        stimuli_dict[key] = dict(item)
    # for key, item in bc_dict["StimulusInject"].items():
    #     stimuli_dict[key] = dict(item)

    output_path = os.path.join(output_dir, prot_dir, output_file)
    with open(output_path, "w") as f:
        json.dump(stimuli_dict, f, indent=4)


# adapted from glusynapseutils.simulation.simulator._runconnectedpair_process
def extract_synprop(
    output_dir,
    basedir,
    syn_dir="synapses",
    output_file="synapse_properties.json",
):
    """Extract model properties to be stored in simulation output."""
    results_path = os.path.join(basedir, "simulation.h5")

    with h5py.File(results_path, "r") as original_results:
        synprop = dict(original_results.attrs)

    # # -- write syn_extra_params on file
    output_path = os.path.join(output_dir, syn_dir, output_file)
    with open(output_path, "w") as f:
        json.dump(synprop, f, indent=4, cls=NpEncoder)


def get_fit_params(basedir):
    """Parse simulation.batch to get the fit_params."""
    fit_params = {}
    fastforward = None
    invivo = False

    param_names = [
        "gca_bar_VDCC_GluSynapse",
        "tau_effca_GB_GluSynapse",
        "gamma_d_GB_GluSynapse",
        "gamma_p_GB_GluSynapse",
        "a00",
        "a01",
        "a10",
        "a11",
        "a20",
        "a21",
        "a30",
        "a31",
    ]

    file_path = os.path.join(basedir, "simulation.batch")
    with open(file_path, "r") as f:
        file_content = f.read()

    for line in file_content.splitlines():
        for chunk in line.split():
            param = chunk.split("=")
            if len(param) > 0:
                # remove '--' from parsed param name
                key = param[0].replace("-", "")

                if key == "invivo":
                    invivo = True
                elif key == "fastforward" and len(param) > 1:
                    fastforward = float(param[1])
                elif key in param_names and len(param) > 1:
                    fit_params[key] = float(param[1])

    return fit_params, fastforward, invivo


def get_blueconfig_dict(bcpath):
    """Get BlueConfig configuration data as a dict."""
    bc = BlueConfigFile()
    with open(bcpath) as blueconfig_handle:
        bc.parse(blueconfig_handle)
    return bc.to_dict()


def extract_all(basedir, output_dir, pregid, postgid, circuitpath, extra_recipe):
    """Extract everything."""
    bcpath = os.path.join(basedir, "BlueConfig")

    circuit = BluepyCircuit(bcpath)
    simulation = BluepySimulation(bcpath)
    ssim = bglibpy.SSim(bcpath)

    bc_dict = get_blueconfig_dict(bcpath)

    default_synrec = [
        "rho_GB",
        "Use_TM",
        "gmax_AMPA",
        "cai_CR",
        "vsyn",
        "ica_NMDA",
        "ica_VDCC",
        "effcai_GB",
    ]

    # params from simulation.batch
    fit_params, fastforward, invivo = get_fit_params(basedir)

    # extract
    extract_morph_data(output_dir, circuit, simulation, postgid)
    extract_constants(
        output_dir,
        circuit,
        postgid,
        bc_dict,
        fastforward=fastforward,
        invivo=invivo,
        fit_params=fit_params,
        default_synrec=default_synrec,
    )
    extract_synapses_data(output_dir, bcpath)
    extract_syn_extra_params(output_dir, ssim, circuitpath, extra_recipe, basedir)
    extract_cpre_and_cpost(output_dir, basedir, fit_params, invivo)
    extract_synprop(output_dir, basedir)
    extract_protocols(output_dir, bc_dict)

    # extract precell
    extract_morph_data(output_dir, circuit, simulation, pregid)
    extract_constants(
        output_dir,
        circuit,
        pregid,
        bc_dict,
        fastforward=fastforward,
        invivo=invivo,
        fit_params=fit_params,
        default_synrec=default_synrec,
        out_file="constants_precell.json",
    )
