"""Decorator launching luigi."""
import configparser
import os
from functools import wraps
from inspect import signature

from e_model_packages.sscx2020.utils import get_output_path
from e_model_packages.synaptic_plasticity.utils import (
    get_output_path as synaptic_plasticity_output_path,
)


def launch_luigi(module, task, reload_hoc=False):
    """Decorator launching luigi before executing function."""

    def launching(func):
        """Decorator."""
        dirs = ["e_model_packages", "sscx2020"]
        test_config = os.path.join("tests", "luigi_test.cfg")
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
                hoc_path = get_output_path(mtype, etype, region, gidx, "tests/output")
                for fname in ["run.hoc", "createsimulation.hoc"]:
                    file_path = os.path.join(hoc_path, fname)
                    if os.path.exists(file_path):
                        os.remove(file_path)

            arguments = ""
            for key, value in bound_args.arguments.items():
                # to change run_single_step into run-single-step
                if "_" in key:
                    key = "-".join(key.split("_"))
                # add --key=value if not boolean. if bool, add --key if true, add nothing otherwise
                if value is True:
                    arguments += "--{} ".format(key)
                elif value is not False:
                    arguments += "--{}={} ".format(key, value)

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
        dirs = ["e_model_packages", "synaptic_plasticity"]
        test_config_path = os.path.join("tests", "luigi_test_synaptic_plasticity.cfg")
        output_dir = "tests/output"
        path_to_luigi = os.path.join(*dirs)
        path_to_module = ".".join(dirs)

        cell_data = {
            "layers": "L4SS_L4SS",
            "pregid": 111202,
            "postgid": 111376,
            "source_dir": "/gpfs/bbp.cscs.ch/project/proj32/"
            + "glusynapse_20190926_release/testing/egger_1999/"
            + "simulations/L4SS_L4SS/111202-111376/50Hz_10ms",
        }

        sig = signature(func)

        @wraps(func)
        def wrapped_function(*args, **kwargs):
            """Inner function."""
            arguments = ""
            for key, value in cell_data.items():
                # to change run_single_step into run-single-step
                if "_" in key:
                    key = "-".join(key.split("_"))
                # add --key=value if not boolean. if bool, add --key if true, add nothing otherwise
                if value is True:
                    arguments += "--{} ".format(key)
                elif value is not False:
                    arguments += "--{}={} ".format(key, value)

            # change PYTHONPATH
            os.system("export PYTHONPATH=${{PYTHONPATH}}:{}".format(path_to_luigi))

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
            original_path = os.path.join(cell_data["source_dir"], "simulation.h5")

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
