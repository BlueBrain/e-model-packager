#!/usr/bin/env python

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

setup(
    name="e-model-packages",
    author="Anil Tuncel",
    author_email="anil.tuncel@epfl.ch",
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
        "https://bbpteam.epfl.ch/repository/devpi/bbprelman/" "dev/+simple/bluepy/"
    ],
    install_requires=[
        "luigi",
        "numpy",
        "bluepy",
        "bluepyopt",
        "bglibpy",
        "bluepysnap",
        "pandas<1.1",
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
    use_scm_version={
        "local_scheme": "no-local-version",
    },
    setup_requires=["setuptools_scm"],
)
