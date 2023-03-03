"""Functions to extract data needed for the glusynapse package."""
import configparser
import json
import logging
from pathlib import Path
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

from e_model_packages.synaptic_plasticity.circuit import GluSynapseExtractor
from e_model_packages.synaptic_plasticity.circuit import BluepyCircuit, BluepySimulation
from e_model_packages.io import NpEncoder

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


def extract_config(
    output_dir,
    circuit,
    postgid,
    pregid,
    bc_dicts,
    recipes_path,
    outfile_basename="config",
    out_fit_params_file="fit_params.json",
    out_const_dir="config",
    celsius=34,
    v_init=-65,
    fastforward=None,
    invivo=False,
    fit_params=None,
    default_synrec=None,
    base_stimuli_name="stimuli",
    base_spiketrain_name="spiketrain",
):
    """Extract and write config and fit_params."""
    # pylint: disable=too-many-arguments, too-many-locals

    # get constants
    cell = circuit.get_cell_attributes(postgid)
    cell_emodel = circuit.get_emodel_attributes(postgid)
    precell = circuit.get_cell_attributes(pregid)
    precell_emodel = circuit.get_emodel_attributes(pregid)

    with open(recipes_path, "r", encoding="utf-8") as recipes_file:
        recipe = json.load(recipes_file)
    unopt_params_path = recipe[cell_emodel.name]["params"]
    precell_unopt_params_path = recipe[precell_emodel.name]["params"]

    # write fit_params
    fit_params_output_path = os.path.join(
        output_dir, out_const_dir, out_fit_params_file
    )
    with open(fit_params_output_path, "w", encoding="utf-8") as f:
        json.dump(fit_params, f, indent=4)

    for bc_dict in bc_dicts:
        base_seed = bc_dict["Run"]["Default"]["BaseSeed"]
        tstop = bc_dict["Run"]["Default"]["Duration"]

        original_dir = bc_dict["Run"]["Default"]["CurrentDir"]
        pulse_spiketrain_names = Path(original_dir).stem
        pulse_name, _ = pulse_spiketrain_names.split("_")

        stimuli_path = f"protocols/{base_stimuli_name}_{pulse_name}.json"
        spiketrain_path = (
            f"protocols/{base_spiketrain_name}_{pulse_spiketrain_names}.dat"
        )

        config = configparser.ConfigParser()
        config["Package"] = {"type": "synplas"}
        config["Cell"] = {
            "celsius": str(celsius),
            "v_init": str(v_init),
            "emodel": cell_emodel.name,
            "precell_emodel": precell_emodel.name,
            "gid": str(int(postgid)),
            "precell_gid": str(int(pregid)),
        }
        config["Paths"] = {
            "morph_path": os.path.join("morphology", cell.morphology_fname),
            "precell_morph_path": os.path.join("morphology", precell.morphology_fname),
            "unoptimized_params_path": unopt_params_path,
            "precell_unoptimized_params_path": precell_unopt_params_path,
            "spiketrain_path": spiketrain_path,
            "stimuli_path": stimuli_path,
            "synplas_output_path": f"output_{pulse_spiketrain_names}.h5",
            "pairsim_output_path": f"output_{pulse_spiketrain_names}.h5",
            "pairsim_precell_output_path": f"output_precell_{pulse_spiketrain_names}.h5",
            # default values
            "memodel_dir": ".",
            "params_path": "%(memodel_dir)s/config/params/final.json",
            "synplas_fit_params_path": "%(memodel_dir)s/config/fit_params.json",
            "syn_dir": "%(memodel_dir)s/synapses",
            "syn_data_file": "synapses.tsv",
            "syn_conf_file": "synconf.txt",
            "syn_prop_path": "%(syn_dir)s/synapse_properties.json",
        }
        config["Protocol"] = {"tstop": tstop, "precell_amplitude": "1.0"}
        config["SynapsePlasticity"] = {
            "fastforward": str(fastforward),
            "invivo": str(invivo).lower(),
            "base_seed": str(base_seed),
            "synrec": json.dumps(default_synrec),
        }
        config["Morphology"] = {
            "do_replace_axon": "True",
        }
        config["Synapses"] = {
            "seed": "846515",
            "rng_settings_mode": "Random123",
        }

        # write config file
        out_file = f"{outfile_basename}_{pulse_spiketrain_names}.ini"
        config_output_path = os.path.join(output_dir, out_const_dir, out_file)
        with open(config_output_path, "w", encoding="utf-8") as configfile:
            config.write(configfile)


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
        intersect_pre_gids=pregids,
    )

    # write data
    synapses_dir = os.path.join(output_dir, syn_dir)

    synapse_tsv_filename = os.path.join(synapses_dir, "synapses.tsv")
    with open(synapse_tsv_filename, "w", encoding="utf-8") as synapse_tsv_file:
        synapse_tsv_file.write(syn_ext.synapse_tsv_content)

    mtype_filename = os.path.join(synapses_dir, "mtype_map.tsv")
    with open(mtype_filename, "w", encoding="utf-8") as mtype_file:
        mtype_file.write(syn_ext.mtype_map_content)

    synconf_filename = os.path.join(synapses_dir, "synconf.txt")
    with open(synconf_filename, "w", encoding="utf-8") as synconf_file:
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
    if "ExtraPreCell" in ssim.bc_circuit.cells._targets._targets:
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
    with open(output_file, "w", encoding="utf-8") as f:
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
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"c_pre": c_pre, "c_post": c_post}, f, indent=4)


def extract_protocols(
    output_dir, bc_dicts, prot_dir="protocols", base_output_name="stimuli"
):
    """Extract Pulse Stimuli from BlueConfig."""
    for bc_dict in bc_dicts:
        original_dir = bc_dict["Run"]["Default"]["CurrentDir"]
        pulse_spiketrain_names = Path(original_dir).stem
        pulse_name = pulse_spiketrain_names.split("_")[0]

        output_file = f"{base_output_name}_{pulse_name}.json"
        output_path = os.path.join(output_dir, prot_dir, output_file)

        if not Path(output_path).is_file():
            stimuli_dict = {}
            for key, item in bc_dict["Stimulus"].items():
                stimuli_dict[key] = dict(item)
            with open(output_path, "w", encoding="utf-8") as f:
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
    with open(output_path, "w", encoding="utf-8") as f:
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
    with open(file_path, "r", encoding="utf-8") as f:
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
    with open(bcpath, encoding="utf-8") as blueconfig_handle:
        bc.parse(blueconfig_handle)
    return bc.to_dict()


def extract_all(
    basedirs, output_dir, pregid, postgid, circuitpath, extra_recipe, recipes_path
):
    """Extract everything."""
    # pylint: disable=too-many-locals
    basedir = basedirs[0]
    bcpath = os.path.join(basedir, "BlueConfig")

    circuit = BluepyCircuit(bcpath)
    simulation = BluepySimulation(bcpath)
    ssim = bglibpy.SSim(bcpath)

    bc_dicts = []
    for basedir_i in basedirs:
        bcpath_i = os.path.join(basedir_i, "BlueConfig")
        bc_dicts.append(get_blueconfig_dict(bcpath_i))

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
    base_stimuli_name = "stimuli"

    # params from simulation.batch
    fit_params, fastforward, invivo = get_fit_params(basedir)

    # extract
    extract_morph_data(output_dir, circuit, simulation, postgid)
    extract_synapses_data(output_dir, bcpath)
    extract_syn_extra_params(output_dir, ssim, circuitpath, extra_recipe, basedir)
    extract_cpre_and_cpost(output_dir, basedir, fit_params, invivo)
    extract_synprop(output_dir, basedir)
    extract_protocols(output_dir, bc_dicts, base_output_name=base_stimuli_name)

    # extract precell
    extract_morph_data(output_dir, circuit, simulation, pregid)

    extract_config(
        output_dir,
        circuit,
        postgid,
        pregid,
        bc_dicts,
        recipes_path,
        fastforward=fastforward,
        invivo=invivo,
        fit_params=fit_params,
        default_synrec=default_synrec,
        base_stimuli_name=base_stimuli_name,
    )
