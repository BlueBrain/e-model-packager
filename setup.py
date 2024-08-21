#!/usr/bin/env python

import imp
import sys

from setuptools import setup, find_packages

if sys.version_info < (3, 8):
    sys.exit("Sorry, Python < 3.8 is not supported")


with open("README.rst", encoding="utf-8") as f:
    README = f.read()

VERSION = imp.load_source("", "e_model_packager/version.py").__version__

EXTRA_GLUSYNAPSE = [
    "numpy<1.24",  # RNG are changed for numpy>=1.24
    "bglibpy==4.4.51",  # not open-sourced
    "glusynapseutils",  # not open-sourced
]

setup(
    name="e-model-packager",
    version=VERSION,
    author="Blue Brain Project, EPFL",
    author_email="bbp-ou-cell@groupes.epfl.ch",
    description="Creates e-model packages from circuits",
    long_description=README,
    long_description_content_type="text/x-rst",
    url="https://github.com/BlueBrain/e-model-packager",
    license="BBP-internal-confidential",
    dependency_links=[
        "https://bbpteam.epfl.ch/repository/devpi/bbprelman/" "dev/+simple/bluepy/",
    ],
    install_requires=[
        "luigi",
        "numpy",
        "bluepy>=v2.3.0",  # not open-sourced: this is NOT the bluepy package available on PyPi
        "bluepyopt>=1.13.168",
        "pandas",
        "pynwb >= 2.0.0",
        "EModelRunner>=1.1.9",
        "luigi-tools>=0.0.6",
        "h5py",
        "efel",
        "schema",
        "click>=8.0.0",
        "bluecellulab",
    ],
    extras_require={
        "glusynapse": EXTRA_GLUSYNAPSE,
    },
    packages=find_packages(),
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
