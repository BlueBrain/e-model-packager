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

import os


def get_output_path(output_dir, layers, pregid, postgid):
    """Return cell output path given layers, pregid and postgid."""
    gids = str(pregid) + "-" + str(postgid)
    return os.path.join(output_dir, layers, gids)
