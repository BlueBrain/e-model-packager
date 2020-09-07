"""Creates .hoc from cell."""
import os
from datetime import datetime

import jinja2

import bluepyopt
from bluepyopt.ephys.create_hoc import (
    _generate_parameters,
    _generate_channels_by_location,
    _generate_reinitrng,
)


def create_run_hoc(template_dir, template_filename, config):
    """Returns a string containing run.hoc."""
    # load config data
    step_stimulus = config.getboolean("Protocol", "step_stimulus")

    # laod template
    template_path = os.path.join(template_dir, template_filename)
    with open(template_path) as template_file:
        template = template_file.read()
        template = jinja2.Template(template)

    # edit template
    return template.render(
        step_stimulus=step_stimulus,
    )


def create_synapse_hoc(
    template_dir, template_filename, config, gid, synapses_template_name="synapses"
):
    """Returns a string containing the synapse hoc."""
    # load config data
    seed = config.get("Synapses", "seed")
    rng_settings_mode = config.get("Synapses", "rng_settings_mode")
    syn_stim_mode = config.get("Protocol", "syn_stim_mode")
    syn_interval = config.get("Protocol", "syn_interval")
    syn_start = config.get("Protocol", "syn_start")
    syn_noise = config.get("Protocol", "syn_noise")
    syn_nmb_of_spikes = config.get("Protocol", "syn_nmb_of_spikes")
    syn_total_duration = config.get("Protocol", "syn_total_duration")
    syn_stim_seed = config.get("Protocol", "syn_stim_seed")
    syn_dir = config.get("Paths", "syn_dir_for_hoc")
    syn_conf_file = config.get("Paths", "syn_conf_file")
    syn_data_file = config.get("Paths", "syn_data_file")

    # load template
    template_path = os.path.join(template_dir, template_filename)
    with open(template_path) as template_file:
        template = template_file.read()
        template = jinja2.Template(template)

    # edit template
    return template.render(
        TEMPLATENAME=synapses_template_name,
        GID=gid,
        SEED=seed,
        syn_stim_mode=syn_stim_mode,
        syn_interval=syn_interval,
        syn_start=syn_start,
        syn_noise=syn_noise,
        syn_nmb_of_spikes=syn_nmb_of_spikes,
        syn_total_duration=syn_total_duration,
        syn_stim_seed=syn_stim_seed,
        rng_settings_mode=rng_settings_mode,
        syn_dir=syn_dir,
        syn_conf_file=syn_conf_file,
        syn_data_file=syn_data_file,
    )


def create_hoc(
    mechs,
    parameters,
    morphology=None,
    ignored_globals=(),
    replace_axon=None,
    template_name="CCell",
    template_filename="cell_template.jinja2",
    disable_banner=None,
    template_dir=None,
    custom_jinja_params=None,
    add_synapses=False,
    synapses_template_name="hoc_synapses",
    syn_hoc_filename="synapses.hoc",
    syn_dir="synapses",
):
    """Return a string containing the hoc template.

    Args:
        mechs (): All the mechs for the hoc template
        parameters (): All the parameters in the hoc template
        morphology (str): Name of morphology
        ignored_globals (iterable str): HOC coded is added for each
        NrnGlobalParameter
        that exists, to test that it matches the values set in the parameters.
        This iterable contains parameter names that aren't checked
        replace_axon (str): String replacement for the 'replace_axon' command.
        Must include 'proc replace_axon(){ ... }
        template_name (str): name of cell class in hoc
        template_filename (str): file name of the jinja2 template
        template_dir (str): dir name of the jinja2 template
        disable_banner (bool): if not True: a banner is added to the hoc file
        custom_jinja_params (dict): dict of additional jinja2 params in case
        of a custom template
        add_synapses (bool): if True: synapses are loaded in the hoc
        synapses_template_name (str): synapse class name in hoc
        syn_hoc_filename (str): file name of synapse hoc file
        syn_dir (str): directory where the synapse data /files are
    """
    if template_dir is None:
        template_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "templates")
        )

    template_path = os.path.join(template_dir, template_filename)
    with open(template_path) as template_file:
        template = template_file.read()
        template = jinja2.Template(template)

    global_params, section_params, range_params, location_order = _generate_parameters(
        parameters
    )
    channels = _generate_channels_by_location(mechs, location_order)

    ignored_global_params = {}
    for ignored_global in ignored_globals:
        if ignored_global in global_params:
            ignored_global_params[ignored_global] = global_params[ignored_global]
            del global_params[ignored_global]

    if not disable_banner:
        banner = "Created by BluePyOpt(%s) at %s" % (
            bluepyopt.__version__,
            datetime.now(),
        )
    else:
        banner = None

    re_init_rng = _generate_reinitrng(mechs)

    if custom_jinja_params is None:
        custom_jinja_params = {}

    return template.render(
        template_name=template_name,
        banner=banner,
        channels=channels,
        morphology=morphology,
        section_params=section_params,
        range_params=range_params,
        global_params=global_params,
        re_init_rng=re_init_rng,
        replace_axon=replace_axon,
        ignored_global_params=ignored_global_params,
        add_synapses=add_synapses,
        synapses_template_name=synapses_template_name,
        syn_hoc_filename=syn_hoc_filename,
        syn_dir=syn_dir,
        **custom_jinja_params,
    )


def create_simul_hoc(template_dir, template_filename, config):
    """Create createsimulation.hoc file."""
    # load config data
    step_stimulus = config.getboolean("Protocol", "step_stimulus")
    stimulus_duration = config.get("Protocol", "stimulus_duration")
    stimulus_delay = config.get("Protocol", "stimulus_delay")
    hold_stimulus_delay = config.get("Protocol", "hold_stimulus_delay")
    hold_stimulus_duration = config.get("Protocol", "hold_stimulus_duration")
    total_duration = config.get("Protocol", "total_duration")
    add_synapses = config.getboolean("Synapses", "add_synapses")
    syn_stim_mode = config.get("Protocol", "syn_stim_mode")
    syn_dir = config.get("Paths", "syn_dir_for_hoc")
    syn_hoc_file = config.get("Paths", "syn_hoc_file")

    # laod template
    template_path = os.path.join(template_dir, template_filename)
    with open(template_path) as template_file:
        template = template_file.read()
        template = jinja2.Template(template)

    # edit template
    return template.render(
        step_stimulus=step_stimulus,
        stimulus_duration=stimulus_duration,
        stimulus_delay=stimulus_delay,
        hold_stimulus_delay=hold_stimulus_delay,
        hold_stimulus_duration=hold_stimulus_duration,
        total_duration=total_duration,
        add_synapses=add_synapses,
        syn_stim_mode=syn_stim_mode,
        syn_dir=syn_dir,
        syn_hoc_file=syn_hoc_file,
    )
