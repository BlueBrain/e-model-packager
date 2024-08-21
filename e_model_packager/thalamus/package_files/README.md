Instructions
============

The downloaded Thalamus package can be run with the EModelRunner software released under the Apache 2.0 License on PyPI.

EModelRunner is a Python package designed to run the cell models provided by the Blue Brain portals simply and straightforwardly.

The source code of EModelRunner is available on GitHub.

    https://pypi.org/project/EModelRunner/


Installing EModelRunner
=======================

The usual way to install EModelRunner is using pip. In that case, you probably want to use a python virtual environment.

Install using ``pip``

    pip install emodelrunner


Installing from source 
----------------------

If you want to make changes to emodelrunner, you might want to install it using the source repository.

   git clone https://github.com/BlueBrain/EModelRunner.git

and run pip from inside the newly created emodelrunner subdirectory 
(do not forget the dot at the end of the command)

    pip install -e .

Supported systems
-----------------

The code of emodelrunner can be installed on any POSIX system that supports 
pip-installable python code.


Installing Neuron
=================

Neuron is required to run the downloaded packages.
It is also a dependency to EModelRunner. There are multiple ways of installing Neuron. Here we list 3 ways to install Neuron.

Using the NEURON PyPI package
------------------------------

The easiest way of installing NEURON is through PyPI.
Neuron can be installed through pip using

    pip install NEURON


Installing Neuron from source
-----------------------------

The up-to-date installation instructions can be found at 

    https://github.com/neuronsimulator/nrn

Using a Linux package for Neuron
---------------------------------

On RPM systems one can install NEURON and its python interface using the following command

    sudo dnf install python3-neuron

On Debian systems the corresponding command is

    sudo apt-get install python3-neuron

Running the packages
=====================

Once you installed EModelRunner and Neuron, go to the package directory and call the run script.

    run_py.sh PATH_TO_CONFIG

For example

    ./run_py.sh config/config_recipe_prots.ini 

The voltage and current recordings will appear in the python_recordings directory after a successful run.


Package contents
=================

The package contains the following files and directories.


- LICENSE.txt
- LICENSE_CC-BY-CA-SA-4.0
- VPL_IN_bAC_IN_14128.nwb
- cell_info.json
- compile_mechanisms.sh
- config
- factsheets
- mechanisms
- morphology
- python_recordings
- run_py.sh
- synapses
- x86_64

The nwb file stands for the Neurodata Without Borders format. 
More information about the nwb format can be found at

    https://www.nwb.org/

cell_info.json contains the cell properties such as its morphology and e-type.

compile_mechanisms.sh gets called by run_py.sh. It compiles the mechanisms listed
 at the mechanisms dir.


config directory contains the configuration of the protocols to be applied to the cell.

It also contains the optimised parameter values and feature values extracted for 
corresponding e-type.

factsheets directory contains the morphology and electrophysiology factsheets data.

synapses directory contains the hoc readable synapses information extracted from the circuit.

x86_64 directory gets created after compiling the mod files.

Further Read
============

If you would like to learn more about EModelRunner, please refer to EModelRunner's documentation.

    https://emodelrunner.readthedocs.io/en/latest/?badge=latest

Support
=======

If you notice a bug or if you would like to make a feature request, please use EModelRunner's issue tracker.

    https://github.com/BlueBrain/EModelRunner/issues


Funding & Acknowledgment
========================

The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

Copyright (c) BBP/EPFL 2020-2022. This work is licenced under Creative Common CC BY-NC-SA-4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
