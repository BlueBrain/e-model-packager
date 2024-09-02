"""Script to create an nwb file given stimuli and responses."""
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

from datetime import datetime
from dateutil.tz import tzlocal

import numpy as np
from pynwb import NWBHDF5IO
from pynwb.file import NWBFile
from pynwb.icephys import CurrentClampStimulusSeries, CurrentClampSeries

from e_model_packager.nwb.units import UnitConverter


def interpolate(time, voltage, new_dt):
    """Interpolate voltage to new dt."""
    interp_time = np.arange(time[0], time[-1], new_dt)
    interp_voltage = np.interp(interp_time, time, voltage)

    return interp_time, interp_voltage


def create_nwb(emodel_name, protocol_responses, file_description):
    """Creates and NWB object from the given stimuli/responses.

    Args:
        emodel_name (str): name of the emodel
        protocol_responses (list of namedtuple): with fields: name, voltage, current
        file_description (str): description string to be added to the file

    Returns:
        pynwb.file.NWBFile: the resulting NWB file
    """
    # pylint: disable=no-member, redefined-outer-name, too-many-locals
    nwbfile = NWBFile(
        session_description=file_description,
        identifier=emodel_name,
        session_start_time=datetime.now(tzlocal()),
    )
    # Add a device
    device = nwbfile.create_device(name="BBP-single-cell-simulator")
    # Add an intracellular electrode
    electrode = nwbfile.create_icephys_electrode(
        name="Simulation electrode",
        description="a placeholder intracellular electrode",
        device=device,
    )

    sampling_rate = 1 / 25e-6  # for a sampling period of 25 microsec
    milivolt_volt = UnitConverter(1e-3)
    nanoamp_amp = UnitConverter(1e-9)
    for protocol_res in protocol_responses:
        stimulus_array = protocol_res.current[:, 1]
        voltage_array = protocol_res.voltage[:, 1]

        protocol_name = protocol_res.name
        voltage_array = milivolt_volt.convert_array(voltage_array)
        stimulus_array = nanoamp_amp.convert_array(stimulus_array)

        # Create an ic-ephys stimulus
        stimulus = CurrentClampStimulusSeries(
            name=protocol_name,
            data=stimulus_array,
            rate=sampling_rate,
            electrode=electrode,
            gain=-1.0,
        )

        # Create an ic-response
        response = CurrentClampSeries(
            name=protocol_name,
            data=voltage_array,
            rate=sampling_rate,
            electrode=electrode,
            gain=-1.0,
        )

        # (A) Add an intracellular recording to the file
        ir_index = nwbfile.add_intracellular_recording(
            electrode=electrode, stimulus=stimulus, response=response
        )

        # (B) Add a list of sweeps to the sweeps table
        sweep_index = nwbfile.add_icephys_simultaneous_recording(
            recordings=[
                ir_index,
            ]
        )

        # (C) Add a list of simultaneous recordings table indices as a sequential recording
        sequence_index = nwbfile.add_icephys_sequential_recording(
            simultaneous_recordings=[
                sweep_index,
            ],
            stimulus_type=protocol_name,
        )

        # (D) Add a list of sequential recordings table indices as a repetition
        nwbfile.add_icephys_repetition(
            sequential_recordings=[
                sequence_index,
            ]
        )

    return nwbfile


def write_nwb(nwbfile, output_path):
    """Writes nwb to the output_path."""
    # pylint: disable=too-many-function-args
    with NWBHDF5IO(output_path, "w") as io:
        io.write(nwbfile)
