## Installation

... to complete when package is finished

## Dependencies

... to complete when package is finished

## How to run the cell using python

Running the cell with the default configuration should as easy as :

    sh run_py.sh

if you want to run the three standard step stimuli protocols with synapses desactivated,
or :

    sh run_py.sh config_synapses.ini

if you want to run the standard synapses stimuli only protocol.

If you want to change the default parameters, go see the
'How to change the configuration file'
section.


## How to run the cell using hoc

Running the cell with the default configuration
(three standard step stimuli protocols with synapses desactivated)
should as easy as :

    sh run_hoc.sh

If you want to change the default parameters,
change the config file according to the
'How to change the configuration file'
section of this README, and then run the create_hoc.py script using python :

    python create_hoc.py --c <your_config_file>.ini

The hoc files are then rewritten and you can run them as described above.


## How to use the GUI

Launching the GUI should be as simple as :

    python GUI.py

On the upper left panel, you can change the display of the figures.

On the bottom left panel of the GUI, you can change the parameters for the step and holding stimuli,
as well as the simulation running time.

On the right panel, you can change the parameters for the synapse stimuli.

On the center panel, the cell shape figures and the soma voltage plot are displayed.

To start a simulation, click on the Start button on the center panel.

If a simulation is running, you can pause it by clicking on the pause button.

A paused simulation can be restarted by clicking on the start button,
or resumed by clicking on the continue button.


## How to change the configuration file

The configuration file should always be in the config/ folder.
There, you can either modify an existing configuration file (config_synapses.ini)
or you can create your own.

Below are the default configuration parameters that you can change :


    [Protocol]
    # set to True to run the simulation with step stimuli
    run_step_protocol=True
    # set to True to run all three steps (run_step_protocol must be True)
    run_all_steps=True
    # if run_step_protocol is True and run_all_steps is False, run this step (must be 1,2 or 3)
    run_step_number=1
    # duration of the simulation [ms]
    total_duration=3000
    # time at which the step stimuli begin [ms]
    stimulus_delay=700
    # duration of the step stimuli [ms]
    stimulus_duration=2000
    # time at which the holding stimulus begin [ms]
    hold_stimulus_delay=0
    # suration of the holding stimulus [ms]
    hold_stimulus_duration=3000
    # synapse stimulus. can be vecstim (one random spike per synapse)
    # or netstim (synapses spike at regular interval for a given amount of spikes)
    syn_stim_mode=vecstim
    # time at which the synapses are deactivated (for vecstim)
    syn_stop=%(total_duration)s
    # interval between two synaptic spike [ms] (for netstim)
    syn_interval=100
    # number of spikes for each synapse (for netstim)
    syn_nmb_of_spikes=5
    # time at which to activate the synapses [ms]
    syn_start=50
    # synaptic noise. set to 0 to have no noise
    syn_noise=0
    # random seed for vecstim. affects the time at which synapses spike
    syn_stim_seed=1
    # which random generator to use. can be python or neuron. 
    vecstim_random=python

    [Morphology]
    # set to True to replace the axon by an AIS stub.
    do_replace_axon=True

    [Sim]
    # simulation timestep [ms]
    dt=0.025
    # set to True to allow adaptative timestep. 
    # attention: these scripts have not been designed to support adaptative timesteps.
    cvcode_active=False

    [Synapses]
    # set to True to add synapses
    add_synapses=False
    # random seed for the synapses
    seed=932156
    # random seed mode. Can be Random123 or Compatibility
    rng_settings_mode=Random123

    [Paths]
    # paths to files.
    memodel_dir=.
    output_path=%(memodel_dir)s/python_recordings
    output_file=soma_voltage_
    constants_dir=config
    constants_file=constants.json
    recipes_dir=config/recipes
    recipes_file=recipes.json
    params_dir=config/params
    params_file=final.json
    protocol_amplitudes_dir=config
    protocol_amplitudes_file=current_amps.json
    templates_dir=templates
    create_hoc_template_file=cell_template_neurodamus.jinja2
    replace_axon_hoc_dir=%(templates_dir)s
    replace_axon_hoc_file=replace_axon_hoc.hoc
    syn_dir_for_hoc=synapses
    syn_dir=%(memodel_dir)s/%(syn_dir_for_hoc)s
    syn_data_file=synapses.tsv
    syn_conf_file=synconf.txt
    syn_hoc_file=synapses.hoc
    syn_mtype_map=mtype_map.tsv
    simul_hoc_file=createsimulation.hoc
