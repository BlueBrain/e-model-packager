#!/bin/sh

set -e

SRC_DIR=$1
INSTALL_DIR=$2
PYTHON_BIN=$3

if [ ! -e ${INSTALL_DIR}/.install_finished ]
then
    echo 'Neuron was not fully installed in previous build, installing ...'
    mkdir -p ${SRC_DIR}
    cd ${SRC_DIR}

    if [ ! -d nrn ]
    then
        echo "Downloading NEURON from github ..."
        git clone --depth 10 https://github.com/nrnhines/nrn.git >download.log 2>&1
    else
        echo "Neuron already downloaded"
    fi
    cd nrn
    echo "Preparing NEURON ..."
    ./build.sh >buildsh.log 2>&1
    echo "Configuring NEURON ..."
    PYTHON_BLD=${PYTHON_BIN} ./configure --prefix=${INSTALL_DIR} --without-x --with-nrnpython=${PYTHON_BIN} --disable-rx3d >configure.log 2>&1
    echo "Installing NEURON ..."
    make -j4 install >makeinstall.log 2>&1

    export PATH="${INSTALL_DIR}/x86_64/bin":${PATH}
    export PYTHONPATH="${INSTALL_DIR}/lib/python":${PYTHONPATH}

    echo "Testing NEURON import ...."
    python -c 'import neuron' >testimport.log 2>&1

    touch -f ${INSTALL_DIR}/.install_finished
    echo "NEURON successfully installed"
else
    echo 'Neuron was successfully installed in previous build, not rebuilding'
fi
