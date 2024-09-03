"""Test the thalamus workflow."""

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

import configparser
import os
from functools import partial
from pathlib import Path

from hdmf.common.hierarchicaltable import (
    drop_id_columns,
    flatten_column_index,
    to_hierarchical_dataframe,
)
from pynwb import NWBHDF5IO

from tests.decorators import launch_luigi_thalamus

test_config = configparser.ConfigParser()
test_config.read(os.path.join("tests", "luigi_test_thalamus.cfg"))
get_param = partial(test_config.get, "params")


@launch_luigi_thalamus(module="workflow", task="CreateNWB")
def test_nwb_denormalized_dataframe_conversion(
    mtype=get_param("mtype"),
    etype=get_param("etype"),
    gid=int(get_param("gid")),
    configfile="config_recipe_prots_short.ini",
):
    """Checks if the NWB output can successfully be converted into the denormalized dataframe."""
    output_path = test_config.get("paths", "output")
    fname = f"{mtype}_{etype}_{gid}.nwb"
    memodel_dir = Path(output_path) / mtype / etype / str(gid)
    memodel_dir = Path(str(memodel_dir).replace(":", "-"))

    assert_complete_package_files(memodel_dir)

    with NWBHDF5IO(memodel_dir / fname, "r") as io:
        nwbfile = io.read()
        root_table = nwbfile.get_icephys_meta_parent_table()
        icephys_meta_df = to_hierarchical_dataframe(root_table)
        icephys_meta_df.reset_index(inplace=True)
        flatten_column_index(dataframe=icephys_meta_df, max_levels=2, inplace=True)
        drop_id_columns(dataframe=icephys_meta_df, inplace=False)
        # fmt: off
        prot_keys_gt = {
            'IV_-140', 'RMP', 'Rin_dep', 'Rin_hyp', 'Step_150',
            'Step_200', 'Step_200_hyp', 'Step_250', 'hold_dep', 'hold_hyp'
        }
        # fmt: on
        prot_keys = {
            x.timeseries.name for x in icephys_meta_df[("responses", "response")].values
        }
        assert prot_keys == prot_keys_gt
        assert not icephys_meta_df.empty


def assert_complete_package_files(memodel_dir):
    """Make sure the package files are copied."""
    static_files = test_config.get("files", "package_files")

    for f in static_files.split(","):
        assert Path(memodel_dir / f).stat().st_size > 0

    assert Path(memodel_dir / "cell_info.json").stat().st_size > 0

    synapse_files = ["mtype_map.tsv", "synapses.tsv", "synconf.tsv"]

    for syn_file in synapse_files:
        assert Path(memodel_dir / "synapses" / syn_file).stat().st_size > 0


@launch_luigi_thalamus(module="workflow", task="CreateFactsheets")
def test_metype_factsheet(
    mtype=get_param("mtype"),
    etype=get_param("etype"),
    gid=int(get_param("gid")),
    configfile="config_recipe_prots_short.ini",
):
    """Check if the factsheets get created."""
    output_path = test_config.get("paths", "output")
    factsheets_dir = Path(output_path) / mtype / etype / str(gid) / "factsheets"
    factsheets_dir = Path(str(factsheets_dir).replace(":", "-"))
    assert Path(factsheets_dir / "morphology_factsheet.json").stat().st_size > 0
    assert Path(factsheets_dir / "etype_factsheet.json").stat().st_size > 0
