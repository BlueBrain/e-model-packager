"""Configure the pre-synaptic cell's protocols."""

import efel
from bluepyopt import ephys
from emodelrunner.load import get_release_params
from emodelrunner.load import load_config
from emodelrunner.create_cells import get_precell
from e_model_packages.sscx2020.utils import cwd


def get_protocol(
    amp,
    step_delay,
    step_duration,
    cvode_active,
    burst_interval,
    n_spikes=5,
    protocol_name="precell step protocol",
):
    """Get one step protocol."""
    # location
    soma_loc = ephys.locations.NrnSeclistCompLocation(
        name="soma", seclist_name="somatic", sec_index=0, comp_x=0.5
    )

    # recording
    presyn_rec = ephys.recordings.CompRecording(
        name=protocol_name, location=soma_loc, variable="v"
    )

    # stimulus
    presyn_stim = []
    for i in range(n_spikes):
        presyn_stim.append(
            ephys.stimuli.NrnSquarePulse(
                step_amplitude=amp,
                step_delay=step_delay + i * burst_interval,
                step_duration=step_duration,
                location=soma_loc,
                total_duration=step_delay + n_spikes * burst_interval + 30,
            )
        )

    # protocol
    return ephys.protocols.SweepProtocol(
        protocol_name, presyn_stim, [presyn_rec], cvode_active
    )


def run_precell(
    amp, step_delay, step_duration, burst_interval=50, n_spikes=5, fixhp=True
):
    """Run precell (to be run in the memodel repo)."""
    cvode_active = True

    config = load_config(filename="config_pairsim.ini")

    # load cell
    precell = get_precell(
        config,
        fixhp=fixhp,
    )

    # simulator
    sim = ephys.simulators.NrnSimulator(
        dt=config.getfloat("Sim", "dt"), cvode_active=cvode_active
    )
    # set dynamic timestep tolerance
    sim.neuron.h.cvode.atolscale("v", 0.1)

    # parameters
    pre_release_params = get_release_params(config.get("Cell", "emodel"))

    # protocol
    protocol = get_protocol(
        amp,
        step_delay,
        step_duration,
        cvode_active,
        burst_interval,
        n_spikes,
    )

    # run
    responses = protocol.run(
        cell_model=precell,
        param_values=pre_release_params,
        sim=sim,
    )

    # return 1st (and only) response (since only one recording was used)
    return next(iter(responses.values()))


def get_spikedelay_and_duration(
    memodel_dir, long_step_duration, amp, step_delay, add_duration=1
):
    """Get spikedelay and step duration for a given amplitude."""
    with cwd(memodel_dir):
        response = run_precell(
            amp=amp,
            step_delay=step_delay,
            step_duration=long_step_duration,
            n_spikes=1,
        )

    # get spike time after applying the stimulus
    trace = {
        "V": response["voltage"],
        "T": response["time"],
        "stim_start": [step_delay],
        "stim_end": [step_delay + long_step_duration],
        "Threshold": [-30],  # threshold used by synapses
    }
    efel_results = efel.getFeatureValues(
        [trace],
        ["peak_time"],
    )

    peak_times = efel_results[0]["peak_time"] - step_delay
    spikedelay = peak_times[0]
    # replace duration if it is so long that the cell spikes twice
    if len(peak_times) > 1:
        step_duration = int(peak_times[0] + (peak_times[1] - peak_times[0]) / 2.0)
    else:
        step_duration = int(spikedelay + add_duration)

    return step_duration, spikedelay


def check_repeated_spikes(
    memodel_dir, step_duration, amp, burst_interval, n_spikes=5, step_delay=100
):
    """Return peak times after multiple step protocols."""
    # run the cell
    with cwd(memodel_dir):
        response = run_precell(
            amp=amp,
            step_delay=step_delay,
            step_duration=step_duration,
            burst_interval=burst_interval,
            n_spikes=n_spikes,
        )

    # get spike time after applying the stimulus
    trace = {
        "V": response["voltage"],
        "T": response["time"],
        "stim_start": [step_delay],
        "stim_end": [step_delay + step_duration],
        "Threshold": [-30],  # threshold used by synapses
    }
    efel_results = efel.getFeatureValues(
        [trace],
        ["peak_time"],
    )

    peak_times = efel_results[0]["peak_time"]

    return peak_times


def get_amp_duration_spikedelay(
    memodel_dir,
    amp=1.0,
    max_step_duration=15,
    burst_interval=50,
    n_spikes=5,
    max_amp=8.0,
    long_step_duration=50,
    delay=100,
):
    """Return protocol variables s.t. the cell spikes as expected."""
    # initialize variables s.t. there is no error when while loop starts
    # these variables are redefined later during the loop.
    # these values are not important.
    peak_times = []
    step_duration = max_step_duration

    while (
        len(peak_times) < n_spikes or step_duration > max_step_duration
    ) and amp < max_amp:

        step_duration, spikedelay = get_spikedelay_and_duration(
            memodel_dir, long_step_duration, amp, delay
        )

        peak_times = check_repeated_spikes(
            memodel_dir, step_duration, amp, burst_interval, n_spikes, delay
        )

        if len(peak_times) < n_spikes or step_duration > max_step_duration:
            amp += 0.5

    return step_duration, amp, spikedelay
