"""Decorator launching luigi."""
import os
from functools import wraps
from inspect import signature

from e_model_packages.sscx2020.utils import get_output_path


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
                gidx = bound_args.arguments["gidx"]
                hoc_path = get_output_path(mtype, etype, gidx, "tests/output")
                for fname in ["run.hoc", "createsimulation.hoc"]:
                    file_path = os.path.join(hoc_path, fname)
                    if os.path.exists(file_path):
                        os.remove(file_path)

            arguments = ""
            for key, value in bound_args.arguments.items():
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
