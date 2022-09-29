e-model-packages
================

Creates e-model packages from circuits.

The implementation uses the Luigi Workflow Management System.

Can generate packages for various e-models, including: SSCX, Glusynapse (Synaptic Plasticity) and Thalamus.


Usage notes
------------

To generate the synaptic plasticity packages e-model-packages needs to be installed with the EXTRA_GLUSYNAPSE flag.

.. code-block:: console

    $ pip install -i https://bbpteam.epfl.ch/repository/devpi/simple/ -e .[glusynapse]


For all other use-cases, refer to the following command:

.. code-block:: console

    $ pip install -i https://bbpteam.epfl.ch/repository/devpi/simple/ -e .[recent_bglibpy]
