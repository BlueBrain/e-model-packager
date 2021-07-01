#!/usr/bin/env python

import imp
import sys

from setuptools import setup, find_packages

if sys.version_info < (2, 7):
    sys.exit("Sorry, Python < 2.7 is not supported")

# read the contents of the README file
if sys.version_info < (3, 0):
    import io

    with io.open("README.rst", encoding="utf-8") as f:
        README = f.read()
else:
    with open("README.rst", encoding="utf-8") as f:
        README = f.read()

VERSION = imp.load_source("", "e_model_packages/version.py").__version__

setup(
    name="e-model-packages",
    version=VERSION,
    author="Anil Tuncel, Aurélien Jaquier",
    author_email="bbp-ou-cell@groupes.epfl.ch",
    description="Creates e-model packages from circuits",
    long_description=README,
    long_description_content_type="text/x-rst",
    url="https://bbpteam.epfl.ch/documentation/projects/e-model-packages",
    project_urls={
        "Tracker": "https://bbpteam.epfl.ch/project/issues/projects/EMODELPKGS/issues",
        "Source": "ssh://bbpcode.epfl.ch/cells/e-model-packages",
    },
    license="BBP-internal-confidential",
    dependency_links=[
        "https://bbpteam.epfl.ch/repository/devpi/bbprelman/" "dev/+simple/bluepy/",
    ],
    install_requires=[
        "luigi",
        "numpy",
        "bluepy>=v2.3.0",
        # force bluepyopt to be the same as in BPEM.
        # Can revert to regular BPO when we switch to proj38 code.
        "bluepyopt @ git+http://github.com/BlueBrain/BluePyOpt@CMA_clean#egg=bluepyopt",
        "bglibpy",
        "bluepysnap",
        "pandas",
        "pynwb",
        "ndx-icephys-meta @ git+https://git@github.com/oruebel/ndx-icephys-meta.git@2bd06bd152901ae5853253358d7efc66714f9a22",
        "EModelRunner>=0.0.16",
        "luigi-tools>=0.0.6",
        "h5py",
        "glusynapseutils @ file://localhost//gpfs/bbp.cscs.ch/project/proj32/ajaquier/GluSynapseUtils/dist/glusynapseutils-0.0.1.dev0-py3-none-any.whl",
        "efel",
    ],
    packages=find_packages(),
    python_requires=">=2.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
