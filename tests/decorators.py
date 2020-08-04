import os
import shutil
from functools import wraps
from inspect import signature


def launch_luigi(module, task):
    """Decorator launching luigi before executing function."""

    def launching(func):
        """Decorator."""
        dirs = ["e_model_packages", "sscx2020"]
        path_to_luigi = os.path.join(*dirs)
        path_back = os.path.join("..", "..")
        path_to_module = ".".join(dirs)

        sig = signature(func)

        @wraps(func)
        def wrapped_function(*args, **kwargs):
            """Inner function."""

            # read arguments, including default ones
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            arguments = ""
            for key, value in bound_args.arguments.items():
                arguments += "--{}={} ".format(key, value)

            # change directory
            os.chdir(path_to_luigi)

            # change PYTHONPATH
            os.system("export PYTHONPATH=${{PYTHONPATH}}:{}".format(os.getcwd()))

            # launch luigi
            os.system(
                "luigi --module {} {} --local-scheduler {}".format(
                    ".".join((path_to_module, module)), task, arguments
                )
            )

            # return to original directory
            os.chdir(path_back)

            func(*args, **kwargs)

        return wrapped_function

    return launching


def erase_output(func):
    """Decorator to delete luigi output."""

    def wrapped_function(*args, **kwargs):
        """Inner function."""
        path_to_output = os.path.join(
            "e_model_packages", "sscx2020", "output", "memodel_dirs"
        )

        if os.path.isdir(path_to_output):
            shutil.rmtree(path_to_output)

        func(*args, **kwargs)

    return wrapped_function
