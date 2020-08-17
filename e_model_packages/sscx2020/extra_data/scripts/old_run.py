"""Python script to run cell model.

@remarks Copyright (c) BBP/EPFL 2018; All rights reserved.
         Do not distribute without further notice.

"""


from __future__ import print_function


# pylint: disable=C0325, W0212, F0401, W0612, F0401

import os
import neuron
import numpy

recordings_dir = "old_python_recordings"


def create_cell():
    """Create the cell model."""
    # Load main cell template
    neuron.h.load_file("%s.hoc" % neuron.h.template_name)

    # Instantiate the cell from the template

    print("Loading cell")
    template = getattr(neuron.h, neuron.h.template_name)
    cell = template(neuron.h.gid, neuron.h.morph_dir, neuron.h.morph_fname)
    return cell


def create_stimuli(cell, step_number):
    """Create the stimuli."""
    print("Attaching stimulus electrodes")

    stimuli = []
    step_amp = [0] * 3

    with open("current_amps.dat", "r") as current_amps_file:
        first_line = current_amps_file.read().split("\n")[0].strip()
        hyp_amp, step_amp[0], step_amp[1], step_amp[2] = first_line.split(" ")

    iclamp = neuron.h.IClamp(0.5, sec=cell.soma[0])
    iclamp.delay = 700
    iclamp.dur = 2000
    iclamp.amp = float(step_amp[step_number - 1])
    print(
        "Setting up step current clamp: "
        "amp=%f nA, delay=%f ms, duration=%f ms"
        % (iclamp.amp, iclamp.delay, iclamp.dur)
    )

    stimuli.append(iclamp)

    hyp_iclamp = neuron.h.IClamp(0.5, sec=cell.soma[0])
    hyp_iclamp.delay = 0
    hyp_iclamp.dur = 3000
    hyp_iclamp.amp = float(hyp_amp)
    print(
        "Setting up hypamp current clamp: "
        "amp=%f nA, delay=%f ms, duration=%f ms"
        % (hyp_iclamp.amp, hyp_iclamp.delay, hyp_iclamp.dur)
    )

    stimuli.append(hyp_iclamp)

    return stimuli


def create_recordings(cell):
    """Create the recordings."""
    print("Attaching recording electrodes")

    recordings = {}

    recordings["time"] = neuron.h.Vector()
    recordings["soma(0.5)"] = neuron.h.Vector()

    recordings["time"].record(neuron.h._ref_t, 0.1)
    recordings["soma(0.5)"].record(cell.soma[0](0.5)._ref_v, 0.1)

    return recordings


def run_step(step_number, plot_traces=None):
    """Run step current simulation with index step_number."""
    cell = create_cell()

    stimuli = create_stimuli(cell, step_number)  # NOQA
    recordings = create_recordings(cell)

    # Overriding default 30s simulation,
    print("Setting simulation time to 3s for the step currents")
    neuron.h.tstop = 3000

    print("Disabling variable timestep integration")
    neuron.h.cvode_active(0)

    print("Running for %f ms" % neuron.h.tstop)
    neuron.h.run()

    time = numpy.array(recordings["time"])
    soma_voltage = numpy.array(recordings["soma(0.5)"])

    soma_voltage_filename = os.path.join(
        recordings_dir, "soma_voltage_step%d.dat" % step_number
    )
    numpy.savetxt(
        soma_voltage_filename, numpy.transpose(numpy.vstack((time, soma_voltage)))
    )

    print(
        "Soma voltage for step %d saved to: %s" % (step_number, soma_voltage_filename)
    )

    if plot_traces:
        import pylab

        pylab.figure()
        pylab.plot(recordings["time"], recordings["soma(0.5)"])
        pylab.xlabel("time (ms)")
        pylab.ylabel("Vm (mV)")
        pylab.gcf().canvas.set_window_title("Step %d" % step_number)


def init_simulation():
    """Initialise simulation environment."""
    neuron.h.load_file("stdrun.hoc")
    neuron.h.load_file("import3d.hoc")

    print("Loading constants")
    neuron.h.load_file("constants.hoc")

    if not os.path.exists(recordings_dir):
        os.mkdir(recordings_dir)


def main(plot_traces=False):
    """Main."""
    # Import matplotlib to plot the traces
    if plot_traces:
        import matplotlib

        matplotlib.rcParams["path.simplify"] = False

    init_simulation()

    for step_number in range(1, 4):
        run_step(step_number, plot_traces=plot_traces)

    if plot_traces:
        import pylab

        pylab.show()


if __name__ == "__main__":
    main()
