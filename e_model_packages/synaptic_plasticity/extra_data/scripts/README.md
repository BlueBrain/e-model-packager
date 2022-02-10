## Dependencies

- [Python 3.6+](https://www.python.org/downloads/release/python-360/)
- [Pip](https://pip.pypa.io) (installed by default in newer versions of Python)
- [Neuron 7.4+](http://neuron.yale.edu/) (compiled with Python support)
- EModelRunner: to be released

## Installation

Install [NEURON](http://neuron.yale.edu/) with Python support on your machine.

Then, install EModelRunner:

    pip install emodelrunner

## How to run the simulation using python

Running the simulation should be as easy as:

    sh run.sh config_path

Where config_path is the path to a specific config file. You will find the available config files in the config folder.
It will run the post-synaptic cell using pre-defined spike train of the pre-synaptic cell to stimulate the synapses.

You can also do a true pair simulation, where both the pre-synaptic and the post-synaptic cells. 
This should be as easy as:

    sh run_pairsim.sh config_path

Where config_path is the path to a specific config file. You will find the available config files in the config folder.

Once the simulation is done, the output is stored as output_{protocol_details}.h5.
If the precell has been simulated too, its output is stored as output_precell_{protocol_details}.h5.

Please, bear in mind that, since it is difficult to make the pre-synaptic cell spike at exactly the same time as in the pre-recorded spike-train file
(especially when the pre-synaptic cell has to spike multiple times in a row),
the results of the 'true pair' simulation might differ slightly from those of the 'post-synaptic cell only' simulation.

All the config files are working for both the 'post-synaptic cell only' and the 'true pair' simulations.

## Funding & Acknowledgment

The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

Copyright (c) BBP/EPFL 2020-2022. This work is licenced under Creative Common CC BY-NC-SA-4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
