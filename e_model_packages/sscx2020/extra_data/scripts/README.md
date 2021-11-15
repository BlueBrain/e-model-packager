## Installation

... to complete when package is finished

## Dependencies

... to complete when package is finished

## How to run the cell using python

Running the cell with the default configuration should as easy as:

    sh run_py.sh config_path

Where config_path is the path to a specific config file. You will find the available config files in the config folder.
Note that the protocol used will depend on the contents of the config file.

If you want to change the contents of the config files, go see the
'How to change the configuration file'
section.

The output can be found under python_recordings.


## How to run the cell using hoc

You can also run the simulation using hoc. In order to do that, you will have to first create the hoc files with the following line:

    python create_hoc.py --config_path config_path

Where config_path is the path to a specific config file. You will find the available config files in the config folder.
Note that the protocol used will depend on the contents of the config file.

If you want to change the contents of the config files, go see the
'How to change the configuration file'
section.

Then run the simulation with:

    sh run_hoc.sh

The output can be found under python_recordings.


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

The configuration files present in the config folder already contain all the available fields. Feel free to change them of create new configuration files.

Below is a configuration file with some explanatory comments:

    [Synapses]
    add_synapses = False
    # RNG seed for synapses
    seed = 846515
    # RNG mode for synapses. Can be "Ramdom123" or "Compatibility"
    rng_settings_mode = Random123
    hoc_synapse_template_name = hoc_synapses

    [Paths]
    # path to the protocol file
    prot_path = config/protocols/RmpRiTau.json
    features_path = config/features/cADpyr_L5PC.json
    # path to the unoptimized parameters
    unoptimized_params_path = config/params/pyr.json
    # directory of the cell package. A lot of paths depend on this.
    memodel_dir = .
    output_dir = %(memodel_dir)s/python_recordings
    # path to the optimized parameters
    params_path = %(memodel_dir)s/config/params/final.json
    units_path = %(memodel_dir)s/config/features/units.json
    # path to the templates folder. Contains hoc templates and the replacement axon.
    templates_dir = %(memodel_dir)s/templates
    cell_template_path = %(templates_dir)s/cell_template_neurodamus.jinja2
    run_hoc_template_path = %(templates_dir)s/run_hoc.jinja2
    createsimulation_template_path = %(templates_dir)s/createsimulation.jinja2
    synapses_template_path = %(templates_dir)s/synapses.jinja2
    replace_axon_hoc_path = %(templates_dir)s/replace_axon_hoc.hoc
    # folder where the synapses data files are
    syn_dir_for_hoc = %(memodel_dir)s/synapses
    # folder where to put the synapse hoc file when created
    syn_dir = %(memodel_dir)s/synapses
    syn_data_file = synapses.tsv
    syn_conf_file = synconf.txt
    syn_hoc_file = synapses.hoc
    syn_mtype_map = mtype_map.tsv
    simul_hoc_file = createsimulation.hoc
    cell_hoc_file = cell.hoc
    run_hoc_file = run.hoc
    # morphology path
    morph_path = morphology/dend-C270999B-P3_axon-C060110A3_-_Scale_x1.000_y0.950_z1.000.asc

    [Morphology]
    # is only used for naming the output files
    mtype = L5TPCa
    # whether to replace the axon by a stub axon
    do_replace_axon = True

    [Cell]
    # temperature in celsius
    celsius = 34
    # initial voltage
    v_init = -80
    # name of the emodel
    emodel = cADpyr_L5TPC
    # id of the cell
    gid = 4138379

    [Sim]
    # timestep if timesteps are constant
    dt = 0.025
    # whether to activate adaptable timesteps.
    cvode_active = False

    [Protocol]
    # index of the section of the apical point
    apical_point_isec = 5


## Funding & Acknowledgment

The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

Copyright (c) BBP/EPFL 2021. This work is licenced under Creative Common CC BY-NC-SA-4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
