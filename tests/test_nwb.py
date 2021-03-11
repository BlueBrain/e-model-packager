"""Tests for the NWB module."""

import configparser
import os
from functools import partial

from pynwb import NWBHDF5IO
from ndx_icephys_meta.icephys import ICEphysFile

from e_model_packages.sscx2020.utils import get_output_path
from tests.decorators import launch_luigi

test_config = configparser.ConfigParser()
test_config.read(os.path.join("tests", "luigi_test.cfg"))
get_param = partial(test_config.get, "params")


@launch_luigi(module="workflow", task="ApplyProtocols")
def test_nwb_denormalized_dataframe_conversion(
    mtype=get_param("mtype"),
    etype=get_param("etype"),
    region=get_param("region"),
    gid=int(get_param("gid")),
    gidx=int(get_param("gidx")),
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
    nwb_path = os.path.join(memodel_path, "recordings.nwb")

    with NWBHDF5IO(nwb_path, "r") as io:
        nwbfile = io.read()
        df = nwbfile.get_icephys_meta_parent_table().to_denormalized_dataframe()
        assert not df.empty
