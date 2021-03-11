"""Script to create an nwb file given protocols, stimuli and responses."""

import pickle
import argparse

from datetime import datetime
from dateutil.tz import tzlocal

from pynwb import NWBHDF5IO
from pynwb.icephys import CurrentClampStimulusSeries, CurrentClampSeries
from ndx_icephys_meta.icephys import ICEphysFile


def create_nwb(emodel_name, protocols, stimuli, responses):
    """Creates and NWB object from the given stimuli/responses."""
    # pylint: disable=no-member, redefined-outer-name
    nwbfile = ICEphysFile(
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

    for protocol in protocols:
        stimulus_array = stimuli[protocol].stimulus.generate()[1]
        voltage_array = responses[f"{protocol}.soma.v"]["voltage"].values

        # Create an ic-ephys stimulus
        stimulus = CurrentClampStimulusSeries(
            name=protocol,
            data=stimulus_array,
            rate=0.0,
            electrode=electrode,
            gain=0.0,
        )

        # Create an ic-response
        response = CurrentClampSeries(
            name=protocol,
            data=voltage_array,
            resolution=0.0,
            rate=0.0,
            electrode=electrode,
            gain=0.0,
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
            stimulus_type=protocol,
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
        "--pickle_recordings",
        help="path to pickle containing protocols, stimuli and responses",
    )
    parser.add_argument("--output_file", help="output NWB file")
    args = parser.parse_args()

    with open(args.pickle_recordings, "rb") as pickle_file:
        recordings = pickle.load(pickle_file)

    emodel_name = args.emodel_name
    output_file = args.output_file

    nwb = create_nwb(
        emodel_name,
        recordings["protocols"],
        recordings["stimuli"],
        recordings["responses"],
    )

    write_nwb(nwb, output_file)
