## Dependencies

- [Python 2.7+](https://www.python.org/download/releases/2.7/) or [Python 3.6+](https://www.python.org/downloads/release/python-360/)
- [Pip](https://pip.pypa.io) (installed by default in newer versions of Python)
- [Neuron 7.4+](http://neuron.yale.edu/) (compiled with Python support)
- EModelRunner: to be released

## Installation

Install [NEURON](http://neuron.yale.edu/) with Python support on your machine.

Then, install EModelRunner:

    pip install emodelrunner

## How to run the simulation using python

Running the simulation should be as easy as:

    sh run.sh

It will run the post-synaptic cell using pre-defined spike train of the pre-synaptic cell to stimulate the synapses.

You can also do a true pair simulation, where both the pre-synaptic and the post-synaptic cells. 
This should be as easy as:

    sh run_pairsim.sh

Once the simulation is done, the output is stored as output.h5. If the precell has been simulated too, its output is stored as output_precell.h5.
