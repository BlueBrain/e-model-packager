"""Decorator launching luigi."""
"""
Copyright 2024 Blue Brain Project / EPFL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
from functools import wraps
from inspect import signature

import json

from e_model_packager.sscx2020.utils import get_output_path
from e_model_packager.synaptic_plasticity.utils import (
    get_output_path as synaptic_plasticity_output_path,
)


def create_command_line_args(args_dict_items):
    """Creates command line argument string for luigi.

    Args:
        args_dict_items: input arguments

    Returns:
        str: arguments string.
    """
    arguments = ""
    for key, value in args_dict_items:
        # to change run_single_step into run-single-step
        if "_" in key:
            key = "-".join(key.split("_"))
        # add --key=value if not boolean. if bool, add --key if true, add nothing otherwise
        if value is True:
            arguments += "--{} ".format(key)
        elif value is not False:
            arguments += "--{}={} ".format(key, value)
    return arguments


def launch_luigi(module, task, reload_hoc=False):
    """Decorator launching luigi before executing function."""

    def launching(func):
        """Decorator."""
        dirs = ["e_model_packager", "sscx2020"]
        test_config = os.path.join("tests", "luigi_test_sscx.cfg")
        path_to_luigi = os.path.join(*dirs)
        path_to_module = ".".join(dirs)
        sig = signature(func)

        @wraps(func)
        def wrapped_function(*args, **kwargs):
            """Inner function."""

            # read arguments, including default ones
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # delete hoc files so that they will be reloaded by luigi
            if reload_hoc:
                mtype = bound_args.arguments["mtype"]
                etype = bound_args.arguments["etype"]
                region = bound_args.arguments["region"]
                gidx = bound_args.arguments["gidx"]
                hoc_path = get_output_path(
                    mtype, etype, region, gidx, "tests/output/sscx"
                )
                for fname in ["run.hoc", "createsimulation.hoc"]:
                    file_path = os.path.join(hoc_path, fname)
                    if os.path.exists(file_path):
                        os.remove(file_path)

            arguments = create_command_line_args(bound_args.arguments.items())

            # change PYTHONPATH
            os.system("export PYTHONPATH=${{PYTHONPATH}}:{}".format(path_to_luigi))

            # launch luigi
            os.system(
                "LUIGI_CONFIG_PATH='{}' luigi --module {} {} --local-scheduler {}".format(
                    test_config, ".".join((path_to_module, module)), task, arguments
                )
            )

            func(*args, **kwargs)

        return wrapped_function

    return launching


def launch_luigi_thalamus(module, task, reload_hoc=False):
    """Decorator launching luigi before executing function."""

    def launching(func):
        """Decorator."""
        dirs = ["e_model_packager", "thalamus"]
        test_config = os.path.join("tests", "luigi_test_thalamus.cfg")
        path_to_luigi = os.path.join(*dirs)
        path_to_module = ".".join(dirs)
        sig = signature(func)

        @wraps(func)
        def wrapped_function(*args, **kwargs):
            """Inner function."""

            # read arguments, including default ones
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            arguments = create_command_line_args(bound_args.arguments.items())

            # change PYTHONPATH
            os.system("export PYTHONPATH=${{PYTHONPATH}}:{}".format(path_to_luigi))

            # launch luigi
            os.system(
                "LUIGI_CONFIG_PATH='{}' luigi --module {} {} --local-scheduler {}".format(
                    test_config, ".".join((path_to_module, module)), task, arguments
                )
            )

            func(*args, **kwargs)

        return wrapped_function

    return launching


def launch_luigi_synaptic_plasticity(module, task):
    """Decorator launching synaptic_plasticity luigi with random cell."""

    def launching(func):
        """Decorator."""
        dirs = ["e_model_packager", "synaptic_plasticity"]
        test_config_path = os.path.join("tests", "luigi_test_synaptic_plasticity.cfg")
        output_dir = "tests/output/synplas"
        path_to_luigi = os.path.join(*dirs)
        path_to_module = ".".join(dirs)

        source_dirs = []
        base_source_dir = (
            "/gpfs/bbp.cscs.ch/project/proj32/"
            + "glusynapse_20190926_release/testing/egger_1999/"
            + "simulations/L4SS_L4SS/111202-111376"
        )
        spiketrains = ["10ms"]
        stims = ["50Hz", "20Hz", "10Hz", "1Hz"]
        config_path = "config/config_50Hz_10ms.ini"
        for stim in stims:
            for train in spiketrains:
                source_dirs.append(os.path.join(base_source_dir, stim + "_" + train))
        cell_data = {
            "layers": "L4SS_L4SS",
            "pregid": 111202,
            "postgid": 111376,
            # be sure that json list is inside ' '
            # so that argparse detect it as one argument
            "source_dirs": "'" + json.dumps(source_dirs) + "'",
        }
        if task == "RunPyScript":
            cell_data["config_path"] = config_path

        sig = signature(func)

        @wraps(func)
        def wrapped_function(*args, **kwargs):
            """Inner function."""
            arguments = create_command_line_args(cell_data.items())

            # change PYTHONPATH
            os.system("export PYTHONPATH=${{PYTHONPATH}}:{}".format(path_to_luigi))
            print(arguments)
            # launch luigi
            os.system(
                "LUIGI_CONFIG_PATH='{}' luigi --module {} {} --local-scheduler {}".format(
                    test_config_path,
                    ".".join((path_to_module, module)),
                    task,
                    arguments,
                )
            )

            # prepare function arguments
            memodel_dir = synaptic_plasticity_output_path(
                output_dir,
                cell_data["layers"],
                cell_data["pregid"],
                cell_data["postgid"],
            )
            original_path = os.path.join(source_dirs[0], "simulation.h5")

            # read arguments, including default ones
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # set arguments
            if "memodel_dir" in bound_args.arguments:
                kwargs["memodel_dir"] = memodel_dir
            if "original_path" in bound_args.arguments:
                kwargs["original_path"] = original_path

            func(*args, **kwargs)

        return wrapped_function

    return launching
