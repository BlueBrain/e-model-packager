"""Contains the utility functions needed for the workflow."""

import glob
import json
import os
from contextlib import contextmanager
import numpy as np

import luigi

# pylint: disable=super-with-arguments


class NpEncoder(json.JSONEncoder):
    """Class to encode numpy object as python object."""

    def default(self, o):
        """Convert numpy integer to int."""
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        else:
            return super(NpEncoder, self).default(o)


def combine_names(mtype, etype, gidx):
    """Returns the combined metype and cell index."""
    return "_".join([mtype, etype, str(gidx)])


def get_output_path(mtype, etype, region, gidx, workflow_output_dir):
    """Returns the path to the outputs directory of one cell model."""
    inner_folder_name = combine_names(mtype, etype, gidx)
    recording_path = os.path.join(mtype, etype, region, inner_folder_name)

    return os.path.join(workflow_output_dir, "memodel_dirs", recording_path)


@contextmanager
def cwd(path):
    """Cwd function that can be used in a context manager."""
    old_dir = os.getcwd()
    os.chdir(path)
    yield
    os.chdir(old_dir)


class LocalTargetCustom(luigi.LocalTarget):
    """Allow '*' in file path when checking for existence."""

    def exists(self):
        """Returns ``True`` if the path exists; ``False`` otherwise.

        Path can have '*' in it.
        """
        if "*" in self.path:
            return bool(glob.glob(self.path))
        return super(LocalTargetCustom, self).exists()
