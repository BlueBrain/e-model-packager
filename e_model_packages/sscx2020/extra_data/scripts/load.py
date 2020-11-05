"""Functions mainly for loading params for run.py."""

import collections

try:
    import ConfigParser as configparser  # for python2
except ImportError:
    import configparser  # for python3
import logging
import json
import os

import bluepyopt.ephys as ephys

from morphology import NrnFileMorphologyCustom
from recordings import RecordingCustom
from cell import CellModelCustom
from synapse import (
    NrnMODPointProcessMechanismCustom,
    NrnNetStimStimulusCustom,
    NrnVecStimStimulusCustom,
)

logger = logging.getLogger(__name__)


def load_config(config_dir="config", filename="config.ini"):
    """Set config from config file and set default value."""
    config_path = os.path.join(config_dir, filename)

    defaults = {
        # protocol
        "step_stimulus": "True",
        "run_all_steps": "True",
        "run_step_number": "1",
        "total_duration": "3000",
        "stimulus_delay": "700",
        "stimulus_duration": "2000",
        "hold_stimulus_delay": "0",
        "hold_stimulus_duration": "3000",
        "syn_stim_mode": "vecstim",
        "syn_total_duration": "%(total_duration)s",
        "syn_interval": "100",
        "syn_nmb_of_spikes": "5",
        "syn_start": "50",
        "syn_noise": "0",
        "syn_stim_seed": "1",
        "vecstim_random": "python",  # can be "python" or "neuron"
        # morphology
        "do_replace_axon": "True",
        "do_set_nseg": "40",
        # sim
        "cvcode_active": "False",
        # synapse
        "add_synapses": "False",
        "seed": "932156",
        "rng_settings_mode": "Random123",  # can be "Random123" or "Compatibility"
        # paths
        "memodel_dir": ".",
        "output_dir": "%(memodel_dir)s/python_recordings",
        "output_file": "soma_voltage_",
        "constants_dir": "config",
        "constants_file": "constants.json",
        "recipes_dir": "config/recipes",
        "recipes_file": "recipes.json",
        "params_dir": "config/params",
        "params_file": "final.json",
        "protocol_amplitudes_dir": "config",
        "protocol_amplitudes_file": "current_amps.json",
        "templates_dir": "templates",
        "create_hoc_template_file": "cell_template_neurodamus.jinja2",
        "replace_axon_hoc_dir": "%(templates_dir)s",
        "replace_axon_hoc_file": "replace_axon_hoc.hoc",
        "syn_dir_for_hoc": "synapses",
        "syn_dir": "%(memodel_dir)s/%(syn_dir_for_hoc)s",
        "syn_data_file": "synapses.tsv",
        "syn_conf_file": "synconf.txt",
        "syn_hoc_file": "synapses.hoc",
        "syn_mtype_map": "mtype_map.tsv",
        "simul_hoc_file": "createsimulation.hoc",
    }

    config = configparser.ConfigParser(defaults=defaults)
    config.read(config_path)

    # make sure that config has all sections
    secs = ["Cell", "Protocol", "Morphology", "Sim", "Synapses", "Paths"]
    for sec in secs:
        if not config.has_section(sec):
            config.add_section(sec)

    return config


def multi_locations(sectionlist):
    """Define mechanisms."""
    if sectionlist == "alldend":
        seclist_locs = [
            ephys.locations.NrnSeclistLocation("apical", seclist_name="apical"),
            ephys.locations.NrnSeclistLocation("basal", seclist_name="basal"),
        ]
    elif sectionlist == "somadend":
        seclist_locs = [
            ephys.locations.NrnSeclistLocation("apical", seclist_name="apical"),
            ephys.locations.NrnSeclistLocation("basal", seclist_name="basal"),
            ephys.locations.NrnSeclistLocation("somatic", seclist_name="somatic"),
        ]
    elif sectionlist == "somaxon":
        seclist_locs = [
            ephys.locations.NrnSeclistLocation("axonal", seclist_name="axonal"),
            ephys.locations.NrnSeclistLocation("somatic", seclist_name="somatic"),
        ]
    elif sectionlist == "allact":
        seclist_locs = [
            ephys.locations.NrnSeclistLocation("apical", seclist_name="apical"),
            ephys.locations.NrnSeclistLocation("basal", seclist_name="basal"),
            ephys.locations.NrnSeclistLocation("somatic", seclist_name="somatic"),
            ephys.locations.NrnSeclistLocation("axonal", seclist_name="axonal"),
        ]
    else:
        seclist_locs = [
            ephys.locations.NrnSeclistLocation(sectionlist, seclist_name=sectionlist)
        ]

    return seclist_locs


def find_param_file(recipes_path, etype):
    """Find the parameter file for unfrozen params."""
    with open(recipes_path, "r") as f:
        recipes = json.load(f)
    recipe = recipes[etype]

    return recipe["params"]


def get_global_param(name, value):
    """Return a global parameter."""
    return ephys.parameters.NrnGlobalParameter(
        name=name, param_name=name, frozen=True, bounds=None, value=value
    )


def load_constants(constants_path):
    """Get etype, morphology, timestep and gid."""
    with open(constants_path, "r") as f:
        data = json.load(f)

    emodel = data["template_name"]
    morph_dir = data["morph_dir"]
    morph_fname = data["morph_fname"]
    dt = data["dt"]
    gid = data["gid"]

    return emodel, morph_dir, morph_fname, dt, gid


def load_params(emodel, params_filename):
    """Get optimised parameters."""
    with open(params_filename, "r") as f:
        params_file = json.load(f)
    data = params_file[emodel]

    param_dict = data["params"]

    return param_dict


def load_mechanisms(mechs_filename):
    """Define mechanisms."""
    with open(mechs_filename) as mechs_file:
        mech_definitions = json.load(
            mechs_file, object_pairs_hook=collections.OrderedDict
        )["mechanisms"]

    mechanisms_list = []
    for sectionlist, channels in mech_definitions.items():

        seclist_locs = multi_locations(sectionlist)

        for channel in channels["mech"]:
            mechanisms_list.append(
                ephys.mechanisms.NrnMODMechanism(
                    name="%s.%s" % (channel, sectionlist),
                    mod_path=None,
                    suffix=channel,
                    locations=seclist_locs,
                    preloaded=True,
                )
            )

    return mechanisms_list


def load_syn_locs(cell):
    """Load synapse point process location."""
    syn_locs = []
    for mech in cell.mechanisms:
        if hasattr(mech, "pprocesses"):
            syn_locs.append(
                ephys.locations.NrnPointProcessLocation("synapse_locs", mech)
            )

    if not syn_locs:
        syn_locs = None

    return syn_locs


def get_syn_stim(syn_locs, config, syn_stim_mode):
    """Get synapse stimulus depending on mode."""
    # load config data
    syn_total_duration = config.getint("Protocol", "syn_total_duration")
    syn_interval = config.getint("Protocol", "syn_interval")
    syn_nmb_of_spikes = config.getint("Protocol", "syn_nmb_of_spikes")
    syn_start = config.getint("Protocol", "syn_start")
    syn_noise = config.getint("Protocol", "syn_noise")
    syn_stim_seed = config.getint("Protocol", "syn_stim_seed")
    vecstim_random = config.get("Protocol", "vecstim_random")

    if syn_stim_mode == "vecstim" and vecstim_random not in ["python", "neuron"]:
        logger.warning(
            "vecstim random not set to 'python' nor to 'neuron' in config file."
        )
        logger.warning("vecstim random will be re-set to 'python'.")
        vecstim_random = "python"

    if syn_stim_mode == "netstim":
        return NrnNetStimStimulusCustom(
            syn_locs,
            syn_total_duration,
            syn_nmb_of_spikes,
            syn_interval,
            syn_start,
            syn_noise,
        )
    if syn_stim_mode == "vecstim":
        return NrnVecStimStimulusCustom(
            syn_locs,
            syn_total_duration,
            syn_start,
            syn_stim_seed,
            vecstim_random,
        )
    else:
        return 0


def step_stimuli(config, soma_loc, cvcode_active=False, syn_stim=None):
    """Create Step Stimuli."""
    step_number = 3
    step_protocols = []

    # load config data
    total_duration = config.getint("Protocol", "total_duration")
    run_all_steps = config.getboolean("Protocol", "run_all_steps")
    run_step_number = config.getint("Protocol", "run_step_number")
    step_delay = config.getint("Protocol", "stimulus_delay")
    step_duration = config.getint("Protocol", "stimulus_duration")
    hold_step_delay = config.getint("Protocol", "hold_stimulus_delay")
    hold_step_duration = config.getint("Protocol", "hold_stimulus_duration")
    amp_filename = os.path.join(
        config.get("Paths", "protocol_amplitudes_dir"),
        config.get("Paths", "protocol_amplitudes_file"),
    )

    if run_all_steps:
        from_step = 0
        up_to = step_number
    elif run_step_number in range(1, step_number + 1):
        from_step = run_step_number - 1
        up_to = run_step_number
    else:
        logger.warning(
            " ".join(
                (
                    "Bad run_step_number parameter.",
                    "Should be between 1 and {}.".format(step_number),
                    "Only first step will be run.",
                )
            )
        )
        from_step = 0
        up_to = 1

    # protocol names
    protocol_names = ["step{}".format(x) for x in range(1, 4)]

    # get current amplitude data
    with open(amp_filename, "r") as f:
        data = json.load(f)
    amplitudes = data["amps"]
    hypamp = data["holding"]

    for protocol_name, amplitude in zip(
        protocol_names[from_step:up_to], amplitudes[from_step:up_to]
    ):
        # use RecordingCustom to sample time, voltage every 0.1 ms.
        rec = RecordingCustom(name=protocol_name, location=soma_loc, variable="v")

        # create step stimulus
        stim = ephys.stimuli.NrnSquarePulse(
            step_amplitude=amplitude,
            step_delay=step_delay,
            step_duration=step_duration,
            location=soma_loc,
            total_duration=total_duration,
        )

        # create holding stimulus
        hold_stim = ephys.stimuli.NrnSquarePulse(
            step_amplitude=hypamp,
            step_delay=hold_step_delay,
            step_duration=hold_step_duration,
            location=soma_loc,
            total_duration=total_duration,
        )

        # create protocol
        stims = [stim, hold_stim]
        if syn_stim is not None:
            stims.append(syn_stim)
        protocol = ephys.protocols.SweepProtocol(
            protocol_name, stims, [rec], cvcode_active
        )

        step_protocols.append(protocol)

    return step_protocols


def define_protocols(config, cell=None):
    """Define Protocols."""
    # load config
    cvcode_active = config.getboolean("Sim", "cvcode_active")
    step_stim = config.getboolean("Protocol", "step_stimulus")
    add_synapses = config.getboolean("Synapses", "add_synapses")
    syn_stim_mode = config.get("Protocol", "syn_stim_mode")

    # synapses location and stimuli
    if add_synapses and syn_stim_mode in ["vecstim", "netstim"]:
        if cell is not None:
            # locations
            syn_locs = load_syn_locs(cell)
            # get synpase stimuli
            syn_stim = get_syn_stim(syn_locs, config, syn_stim_mode)
        else:
            raise Exception("The cell is  missing in the define_protocol function.")
    else:
        syn_stim = None

    # recording location
    soma_loc = ephys.locations.NrnSeclistCompLocation(
        name="soma", seclist_name="somatic", sec_index=0, comp_x=0.5
    )

    # get step stimuli and make protocol(s)
    if step_stim:
        # get step protocols
        step_protocols = step_stimuli(config, soma_loc, cvcode_active, syn_stim)

    elif syn_stim:
        protocol_name = syn_stim_mode
        # use RecordingCustom to sample time, voltage every 0.1 ms.
        rec = RecordingCustom(name=protocol_name, location=soma_loc, variable="v")

        stims = [syn_stim]
        protocol = ephys.protocols.SweepProtocol(
            protocol_name, stims, [rec], cvcode_active
        )
        step_protocols = [protocol]
    else:
        raise Exception(
            "No valid protocol was found. step_stimulus is {}".format(step_stim)
            + " and syn_stim_mode ({}) not in ['vecstim', 'netstim'].".format(
                syn_stim_mode
            )
        )

    return ephys.protocols.SequenceProtocol("twostep", protocols=step_protocols)


def define_parameters(params_filename):
    """Define parameters."""
    parameters = []

    with open(params_filename) as params_file:
        definitions = json.load(params_file, object_pairs_hook=collections.OrderedDict)

    # set distributions
    distributions = collections.OrderedDict()
    distributions["uniform"] = ephys.parameterscalers.NrnSegmentLinearScaler()

    distributions_definitions = definitions["distributions"]
    for distribution, definition in distributions_definitions.items():

        if "parameters" in definition:
            dist_param_names = definition["parameters"]
        else:
            dist_param_names = None
        distributions[
            distribution
        ] = ephys.parameterscalers.NrnSegmentSomaDistanceScaler(
            name=distribution,
            distribution=definition["fun"],
            dist_param_names=dist_param_names,
        )

    params_definitions = definitions["parameters"]

    if "__comment" in params_definitions:
        del params_definitions["__comment"]

    for sectionlist, params in params_definitions.items():
        if sectionlist == "global":
            seclist_locs = None
            is_global = True
            is_dist = False
        elif "distribution_" in sectionlist:
            is_dist = True
            seclist_locs = None
            is_global = False
            dist_name = sectionlist.split("distribution_")[1]
            dist = distributions[dist_name]
        else:
            seclist_locs = multi_locations(sectionlist)
            is_global = False
            is_dist = False

        bounds = None
        value = None
        for param_config in params:
            param_name = param_config["name"]

            if isinstance(param_config["val"], (list, tuple)):
                is_frozen = False
                bounds = param_config["val"]
                value = None

            else:
                is_frozen = True
                value = param_config["val"]
                bounds = None

            if is_global:
                parameters.append(
                    ephys.parameters.NrnGlobalParameter(
                        name=param_name,
                        param_name=param_name,
                        frozen=is_frozen,
                        bounds=bounds,
                        value=value,
                    )
                )
            elif is_dist:
                parameters.append(
                    ephys.parameters.MetaParameter(
                        name="%s.%s" % (param_name, sectionlist),
                        obj=dist,
                        attr_name=param_name,
                        frozen=is_frozen,
                        bounds=bounds,
                        value=value,
                    )
                )

            else:
                if "dist" in param_config:
                    dist = distributions[param_config["dist"]]
                    use_range = True
                else:
                    dist = distributions["uniform"]
                    use_range = False

                if use_range:
                    parameters.append(
                        ephys.parameters.NrnRangeParameter(
                            name="%s.%s" % (param_name, sectionlist),
                            param_name=param_name,
                            value_scaler=dist,
                            value=value,
                            bounds=bounds,
                            frozen=is_frozen,
                            locations=seclist_locs,
                        )
                    )
                else:
                    parameters.append(
                        ephys.parameters.NrnSectionParameter(
                            name="%s.%s" % (param_name, sectionlist),
                            param_name=param_name,
                            value_scaler=dist,
                            value=value,
                            bounds=bounds,
                            frozen=is_frozen,
                            locations=seclist_locs,
                        )
                    )

    return parameters


def get_axon_hoc(replace_axon_hoc):
    """Returns string containing replace axon hoc."""
    with open(replace_axon_hoc, "r") as f:
        return f.read()


def load_syn_mechs(config, pre_mtypes=None, stim_params=None):
    """Load synapse mechanisms."""
    seed = config.getint("Synapses", "seed")
    rng_settings_mode = config.get("Synapses", "rng_settings_mode")
    syn_data_path = os.path.join(
        config.get("Paths", "syn_dir"), config.get("Paths", "syn_data_file")
    )
    syn_conf_path = os.path.join(
        config.get("Paths", "syn_dir"), config.get("Paths", "syn_conf_file")
    )

    # load synapse file data
    synapses_data = load_tsv_data(syn_data_path)

    # load synapse configuration
    synconf_dict = load_synapse_configuration_data(syn_conf_path)

    return NrnMODPointProcessMechanismCustom(
        "synapse_mechs",
        synapses_data,
        synconf_dict,
        seed,
        rng_settings_mode,
        pre_mtypes,
        stim_params,
    )


def create_cell(config):
    """Create a cell. Returns cell, release params and time step."""
    # load constants
    constants_path = os.path.join(
        config.get("Paths", "constants_dir"), config.get("Paths", "constants_file")
    )
    emodel, morph_dir, morph_fname, dt_tmp, gid = load_constants(constants_path)

    # load morphology path
    if config.has_option("Paths", "morph_dir"):
        morph_dir = config.get("Paths", "morph_dir")
    else:
        morph_dir = os.path.join(config.get("Paths", "memodel_dir"), morph_dir)
    if config.has_option("Paths", "morph_file"):
        morph_fname = config.get("Paths", "morph_file")

    morph_path = os.path.join(morph_dir, morph_fname)

    # load mechanisms
    recipes_path = os.path.join(
        config.get("Paths", "recipes_dir"), config.get("Paths", "recipes_file")
    )
    params_filename = find_param_file(recipes_path, emodel)
    mechs = load_mechanisms(params_filename)

    # add synapses mechs
    add_synapses = config.getboolean("Synapses", "add_synapses")
    if add_synapses:
        mechs += [load_syn_mechs(config)]

    # load parameters
    params_path = os.path.join(
        config.get("Paths", "params_dir"), config.get("Paths", "params_file")
    )
    release_params = load_params(params_filename=params_path, emodel=emodel)
    params = define_parameters(params_filename)

    # create morphology
    axon_hoc_path = os.path.join(
        config.get("Paths", "replace_axon_hoc_dir"),
        config.get("Paths", "replace_axon_hoc_file"),
    )
    replace_axon_hoc = get_axon_hoc(axon_hoc_path)
    do_replace_axon = config.getboolean("Morphology", "do_replace_axon")
    do_set_nseg = config.getint("Morphology", "do_set_nseg")
    morph = NrnFileMorphologyCustom(
        morph_path,
        do_replace_axon=do_replace_axon,
        replace_axon_hoc=replace_axon_hoc,
        do_set_nseg=do_set_nseg,
    )

    # create cell
    cell = CellModelCustom(
        name=emodel,
        morph=morph,
        mechs=mechs,
        params=params,
        gid=gid,
        add_synapses=add_synapses,
    )

    return cell, release_params, dt_tmp


def load_tsv_data(tsv_path):
    """Load synapse data from tsv."""
    synapses = []
    with open(tsv_path, "r") as f:
        # first line is dimensions
        for line in f.readlines()[1:]:
            syn = {}
            items = line.strip().split("\t")
            syn["sid"] = int(items[0])
            syn["pre_cell_id"] = int(items[1])
            syn["sectionlist_id"] = int(items[2])
            syn["sectionlist_index"] = int(items[3])
            syn["seg_x"] = float(items[4])
            syn["synapse_type"] = int(items[5])
            syn["dep"] = float(items[6])
            syn["fac"] = float(items[7])
            syn["use"] = float(items[8])
            syn["tau_d"] = float(items[9])
            syn["delay"] = float(items[10])
            syn["weight"] = float(items[11])
            syn["Nrrp"] = float(items[12])
            syn["pre_mtype"] = int(items[13])

            synapses.append(syn)

    return synapses


def load_synapse_configuration_data(synconf_path):
    """Load synapse configuration data into dict[command]=list(ids)."""
    synconf_dict = {}
    with open(synconf_path, "r") as f:
        synconfs = f.read().split("-1000000000000000.0")

    for synconf in synconfs:
        tmp = synconf.split("\n")
        if "" in tmp:
            tmp.remove("")
        if len(tmp) == 2:
            cmd, ids = tmp
            ids = ids.replace(") ", ");")
            ids = ids.split(";")
            if "" in ids:
                ids.remove("")
            synconf_dict[cmd] = ids

    return synconf_dict
