"""Contains the utility functions needed for the workflow."""

import glob
import os

import luigi


def combine_names(mtype, etype, gidx):
    """Returns the combined metype and cell index."""
    return "_".join([mtype, etype, str(gidx)])


def get_output_path(mtype, etype, region, gidx, workflow_output_dir):
    """Returns the path to the outputs directory of one cell model."""
    inner_folder_name = combine_names(mtype, etype, gidx)
    recording_path = os.path.join(mtype, etype, region, inner_folder_name)

    return os.path.join(workflow_output_dir, "memodel_dirs", recording_path)


class LocalTargetCustom(luigi.LocalTarget):
    """Allow '*' in file path when checking for existence."""

    def exists(self):
        """Returns ``True`` if the path exists; ``False`` otherwise.

        Path can have '*' in it.
        """
        if "*" in self.path:
            return bool(glob.glob(self.path))
        return super().exists()
