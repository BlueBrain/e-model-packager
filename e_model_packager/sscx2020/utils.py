"""Contains the utility functions needed for the workflow."""

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
    recording_path = recording_path.replace(":", "-")

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
