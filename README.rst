e-model-packager
================

Creates e-model packages from circuits.

The implementation uses the Luigi Workflow Management System.

Can generate packages for various e-models, including: SSCX, Glusynapse (Synaptic Plasticity) and Thalamus.

This software is dependent on private data and private software.
It has been released on GitHub "as is" for anyone wanting to see the code that has created Blue brain e-model packages such as the `synaptic plasticity ones <https://zenodo.org/records/6352774>`_.


Install
-------

You will not be able to install this software, unless you have access to the private software 'bluepy'.
If you have it, you can install this software by cloning this repository locally using

.. code-block:: console

    git clone https://github.com/BlueBrain/e-model-packager.git

Then going to the created software repository, and installing it as follow:

.. code-block:: console

    pip install -e .


If you want to generate the synaptic plasticity packages, you will also need to have installed the following private software: glusynapseutils and bglibpy version 4.4.51.


Examples
--------

Here is an example to run the whole pipeline for a single e-model package from the somatosensory (SSCX) circuit.
You need to have access to BB5 data (private Blue Brain data) to run this example.

Simply go to the software main repository and execute this code:

.. code-block:: console

    CURRENT_DIR=$PWD
    export PYTHONPATH=${PYTHONPATH}:$CURRENT_DIR:$CURRENT_DIR/e_model_packages/sscx2020

    LUIGI_CONFIG_PATH=e_model_packages/sscx2020/luigi.cfg luigi --module workflow PrepareMEModelDirectory --mtype=L5_TPC:A --etype=cADpyr --gid=4138379 --region=S1ULp --gidx=79597 --configfile=config_synapses.ini --local-scheduler

When the code has finished running, you should see a new ./output folder with an e-model package in it.

Here is how it works. First, you need to re-define your python path so that luigi can access the workflow module in e_model_packages/sscx2020.
If you want to execute the thalamus workflow or the synapse plasticity workflow, you will have to change this path.
Then, you give to luigi the path to its config file, the module you want to execute (here, workflow), and the luigi Task you want to be complete.
Next you can give the package specifications (mtype, etype, gid, region, gidx). If you want to create all the packages, you can remove all of these fields.
Then, you have to give the packager config file, that will determine which protocol the e-model will be able to run and if synapses should be added.
Finally, we ask luigi to use a local scheduler to run the workflow.

Other example launch scripts:

  - e_model_packages/sscx2020/prepare_packages.sh
  - e_model_packages/sscx2020/run_recordings.sh
  - e_model_packager/synaptic_plasticity/prepare_packages.sh


Citation
--------

Add a zenodo doi after open sourcing.


Funding & Acknowledgment
------------------------

The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.
Copyright (c) 2024 Blue Brain Project/EPFL