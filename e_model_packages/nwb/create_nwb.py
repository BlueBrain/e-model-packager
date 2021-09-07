"""Script to create an nwb file given stimuli and responses."""

import argparse

from datetime import datetime
from pathlib import Path
from dateutil.tz import tzlocal

import numpy as np
from pynwb import NWBHDF5IO
from pynwb.file import NWBFile
from pynwb.icephys import CurrentClampStimulusSeries, CurrentClampSeries

from e_model_packages.nwb.units import UnitConverter


def interpolate(time, voltage, new_dt):
    """Interpolate voltage to new dt."""
    interp_time = np.arange(time[0], time[-1], new_dt)
    interp_voltage = np.interp(interp_time, time, voltage)

    return interp_time, interp_voltage


def create_nwb(emodel_name, voltage_recordings, current_recordings):
    """Creates and NWB object from the given stimuli/responses."""
    # pylint: disable=no-member, redefined-outer-name, too-many-locals
    nwbfile = NWBFile(
        session_description="SSCX Simulation Data",
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
    for idx, (current, voltage) in enumerate(
        zip(current_recordings, voltage_recordings)
    ):
        stimulus_array = current[:, 1]
        voltage_array = voltage[:, 1]

        protocol_name = f"step_{idx+1}"
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--emodel_name", help="name of the emodel")
    parser.add_argument(
        "--recordings_path",
        help="path containing stimuli and responses",
    )
    parser.add_argument("--output_file", help="output NWB file")
    args = parser.parse_args()

    voltage_recording_paths = sorted(
        Path(args.recordings_path).glob("soma_voltage_*.dat")
    )
    current_recording_paths = sorted(
        Path(args.recordings_path).glob("current_step*.dat")
    )

    voltage_recordings = [np.loadtxt(volt) for volt in voltage_recording_paths]
    current_recordings = [np.loadtxt(curr) for curr in current_recording_paths]

    emodel_name = args.emodel_name
    output_file = args.output_file

    nwb = create_nwb(emodel_name, voltage_recordings, current_recordings)

    write_nwb(nwb, output_file)
