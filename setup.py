#!/usr/bin/env python

import imp
import sys

from setuptools import setup, find_packages

if sys.version_info < (3, 7):
    sys.exit("Sorry, Python < 3.7 is not supported")


with open("README.rst", encoding="utf-8") as f:
    README = f.read()

VERSION = imp.load_source("", "e_model_packages/version.py").__version__

EXTRA_RECENT_BGLIBPY = ["bglibpy>=4.10.1"]

EXTRA_GLUSYNAPSE = [
    "bglibpy==4.4.51",
    "glusynapseutils @ file://localhost//gpfs/bbp.cscs.ch/project/proj32/ajaquier/GluSynapseUtils/20210903/GluSynapseUtils/dist/glusynapseutils-0.0.1.dev0-py3-none-any.whl",
]

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
        "bluepyopt>=1.13.168",
        "pandas",
        "pynwb >= 2.0.0",
        "EModelRunner>=1.1.7",
        "luigi-tools>=0.0.6",
        "h5py",
        "efel",
        "schema",
        "click>=8.0.0",
    ],
    extras_require={
        "recent_bglibpy": EXTRA_RECENT_BGLIBPY,
        "glusynapse": EXTRA_GLUSYNAPSE,
    },
    packages=find_packages(),
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
