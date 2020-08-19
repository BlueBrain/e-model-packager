"""Functions mainly for loading params for run.py."""

import collections
import configparser
import json
import os

import bluepyopt.ephys as ephys

from mymorphology import NrnFileMorphologyCustom
from myrecordings import MyRecording


def load_config(config_dir="config", filename="config.ini"):
    """Set config from config file and set default value."""
    config_path = os.path.join(config_dir, filename)

    defaults = {
        # protocol
        "total_duration": "3000",
        "stimulus_delay": "700",
        "stimulus_duration": "2000",
        "hold_stimulus_delay": "0",
        "hold_stimulus_duration": "3000",
        "cvcode_active": "False",
        # morphology
        "do_replace_axon": "True",
        "do_set_nseg": "40",
        # paths
        "inner_dir": "${Cell:mtype}_${Cell:etype}_${Cell:gidx}",
        "memodel_dir": "memodel_dirs/${Cell:mtype}/${Cell:etype}/${inner_dir}",
        "output_dir": "${memodel_dir}/python_recordings",
        "output_file": "soma_voltage_",
        "constants_dir": "${memodel_dir}",
        "constants_file": "constants.hoc",
        "recipes_dir": "config/recipes",
        "recipes_file": "recipes.json",
        "params_dir": "config/params",
        "params_file": "final.json",
        "protocol_amplitudes_dir": "${memodel_dir}",
        "protocol_amplitudes_file": "current_amps.dat",
        "templates_dir": "templates",
        "create_hoc_template_file": "cell_template_neurodamus.jinja2",
        "replace_axon_hoc_dir": "${templates_dir}",
        "replace_axon_hoc_file": "replace_axon_hoc.hoc",
    }

    config = configparser.ConfigParser(
        defaults=defaults, interpolation=configparser.ExtendedInterpolation()
    )
    config.read(config_path)

    # make sure that config has all sections
    secs = ["Cell", "Protocol", "Morphology", "Sim", "Paths"]
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


def load_constants(constants_path="constants.hoc"):
    """Get etype, morphology, timestep and gid."""
    with open(constants_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.split("=")
        if line[0] == "template_name":
            fname = line[1].rstrip()
            emodel = fname.strip('"')
        elif line[0] == "morph_dir":
            fname = line[1].rstrip()
            morph_dir = fname.strip('"')
        elif line[0] == "morph_fname":
            fname = line[1].rstrip()
            morph_fname = fname.strip('"')
        elif line[0] == "dt":
            dt = float(line[1].rstrip())
        elif line[0] == "gid":
            gid = int(line[1].rstrip())

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


def define_protocols(amp_filename, config):
    """Define Protocols."""
    # load config
    total_duration = config.getint("Protocol", "total_duration")
    step_delay = config.getint("Protocol", "stimulus_delay")
    step_duration = config.getint("Protocol", "stimulus_duration")
    hold_step_delay = config.getint("Protocol", "hold_stimulus_delay")
    hold_step_duration = config.getint("Protocol", "hold_stimulus_duration")
    cvcode_active = config.getboolean("Protocol", "cvcode_active")

    # locations
    soma_loc = ephys.locations.NrnSeclistCompLocation(
        name="soma", seclist_name="somatic", sec_index=0, comp_x=0.5
    )

    # protocols
    protocol_names = ["step{}".format(x) for x in range(1, 4)]
    with open(amp_filename, "r") as f:
        data = f.read().rstrip()
    amps = data.split()
    amplitudes = [float(amp) for amp in amps[1:]]  # do not take 1st value (hypamp)
    hypamp = float(amps[0])

    step_protocols = []
    for protocol_name, amplitude in zip(protocol_names, amplitudes):
        stim = ephys.stimuli.NrnSquarePulse(
            step_amplitude=amplitude,
            step_delay=step_delay,
            step_duration=step_duration,
            location=soma_loc,
            total_duration=total_duration,
        )
        hold_stim = ephys.stimuli.NrnSquarePulse(
            step_amplitude=hypamp,
            step_delay=hold_step_delay,
            step_duration=hold_step_duration,
            location=soma_loc,
            total_duration=total_duration,
        )
        # use MyRecording to sample time, voltage every 0.1 ms.
        rec = MyRecording(name=protocol_name, location=soma_loc, variable="v")
        protocol = ephys.protocols.StepProtocol(
            protocol_name, stim, hold_stim, [rec], cvode_active=cvcode_active
        )
        step_protocols.append(protocol)

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
    cell = ephys.models.CellModel(
        name=emodel, morph=morph, mechs=mechs, params=params, gid=gid
    )

    return cell, release_params, dt_tmp
