"""Tests for the NWB module."""

import configparser
import os
from functools import partial

from hdmf.common.hierarchicaltable import (
    drop_id_columns,
    flatten_column_index,
    to_hierarchical_dataframe,
)
from pynwb import NWBHDF5IO

from e_model_packages.sscx2020.utils import get_output_path
from tests.decorators import launch_luigi

test_config = configparser.ConfigParser()
test_config.read(os.path.join("tests", "luigi_test.cfg"))
get_param = partial(test_config.get, "params")


@launch_luigi(module="workflow", task="CreateNWB")
def test_nwb_denormalized_dataframe_conversion(
    mtype=get_param("mtype"),
    etype=get_param("etype"),
    region=get_param("region"),
    gid=int(get_param("gid")),
    gidx=int(get_param("gidx")),
    configfile="config_multistep_short.ini",
):
    """Checks if the NWB output can successfully be converted into the denormalized dataframe.

    Args:
        mtype: morphological type
        etype: electrophysiological type
        region: circuit region
        gid: id of cell in the circuit
        gidx: index of cell
    """
    output_path = os.path.join("tests", "output")
    memodel_path = get_output_path(mtype, etype, region, gidx, output_path)
    nwb_file_name = f"{region}_{mtype}_{etype}_{gidx}.nwb"
    nwb_path = os.path.join(memodel_path, nwb_file_name)

    with NWBHDF5IO(nwb_path, "r") as io:
        nwbfile = io.read()
        root_table = nwbfile.get_icephys_meta_parent_table()
        icephys_meta_df = to_hierarchical_dataframe(root_table)
        # Reset the index of the dataframe and turn the values into columns instead
        icephys_meta_df.reset_index(inplace=True)
        # Flatten the column-index, turning the pandas.MultiIndex into a pandas.Index of tuples
        flatten_column_index(dataframe=icephys_meta_df, max_levels=2, inplace=True)
        # Remove the id columns. By setting inplace=False allows us to visualize the result of this
        # action while keeping the id columns in our main icephys_meta_df table
        drop_id_columns(dataframe=icephys_meta_df, inplace=False)
        assert not icephys_meta_df.empty
