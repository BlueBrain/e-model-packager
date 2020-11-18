"""Main Interface."""

import tkinter as tk
from tkinter import ttk
import time

from GUI_utils.simulator import NeuronSimulation
from GUI_utils.frames import FrameMain, FrameConfig, FrameSynapses
from GUI_utils.style import define_style, set_matplotlib_style


class GUI:
    """GUI class. Contains all frames and simulation.

    Attributes:
        simulation (NeuronSimulation): contains BluePyOpt simulation (and cell) data
        play (bool): if True, runs the simulation
        refresh_display_dt (float): timestep (s) for the display of figures
        plot_3d (bool): set to True to plot the cell shapes in 3D
        toolbar_on (bool): set to True to display the matplotlib toolbars
        figsize (str): figures size. can be "small", "medium", or "large".
        root (tk.Tk): root of the GUI
        style (ttk.Style): style of the tkinter objects
        frames (dict of ttk.Frames): main frames embedded in root
        reload (bool): if True, the simulation has to be reloaded
    """

    def __init__(self, fps=15):
        """Constructor.

        Args:
            fps (int): frames per second for the figure display
        """
        # init simulation
        self.simulation = NeuronSimulation()

        # load cell, simulation, and protocol(s)
        self.simulation.load_cell_sim()
        self.simulation.load_protocol()

        self.simulation.instantiate()
        self.simulation.load_synapse_display_data()

        # display params
        self.play = False
        self.refresh_display_dt = self.get_refresh_from_fps(fps)
        self.plot_3d = False
        self.toolbar_on = False
        self.figsize = "medium"

        # Tkinter
        self.root = tk.Tk()
        self.root.title("ME-type models launcher & visualisation interface")

        # ttk style
        self.style = ttk.Style()
        self.style.theme_use("clam")  # to be able to change background, etc. on macos
        define_style(self.style)
        set_matplotlib_style()

        # frames
        self.frames = {}
        self.create_frames()

        self.reload = False

    def get_refresh_from_fps(self, fps):
        """Get refresh rate from fps."""
        return 1.0 / fps

    def create_frames(self):
        """Creates the frames."""
        self.frames["FrameMain"] = FrameMain(self.root, self)
        self.frames["FrameConfig"] = FrameConfig(self.root, self)

        title_synapses = ttk.Label(self.root, text="Synapse Stimuli configuration")
        self.frames["FrameSynapses"] = FrameSynapses(self.root, self, title_synapses)

        self.frames["FrameConfig"].grid(
            row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=2, pady=2
        )
        self.frames["FrameMain"].grid(
            row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=2
        )
        self.frames["FrameSynapses"].grid(
            row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=2, pady=2
        )

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(0, weight=1)

    def update_figures(self):
        """Update the figures."""
        self.frames["FrameMain"].display(self.root, self.simulation)

    def clear_voltage_figure(self):
        """Clear the voltage figure."""
        self.frames["FrameMain"].restart_volt()

    def config_has_changed(self):
        """Stop the simulation when the user has changed configuration."""
        self.reload = True
        self.end_simul()  # stop simul and disable continue button

    def run_simul(self, v_threshold=-65, peak_refresh_dt=1.5):
        """Main loop for running simulation.

        Args:
            v_threshold (int): if voltage is larger than this value,
                increase fps to have a more detailed display of spikes
            peak_refresh_dt (float): timestep (ms) between two
                display actualisations in a detailed display (around a spike)
        """
        # for refreshing rate
        t1 = time.time()
        # for fine display of peaks
        last_t = self.simulation.sim.neuron.h.t  # = 0

        while (
            self.play
            and self.simulation.sim.neuron.h.t
            < self.simulation.sim.neuron.h.tstop - self.simulation.sim.neuron.h.dt / 2
        ):
            self.simulation.sim.neuron.h.fadvance()
            if time.time() - t1 > self.refresh_display_dt or (
                self.simulation.cell.icell.soma[0](0.5).v > v_threshold
                and self.simulation.sim.neuron.h.t > last_t + peak_refresh_dt
            ):
                self.update_figures()
                last_t = self.simulation.sim.neuron.h.t
                t1 = time.time()

        self.update_figures()
        self.play = False

        if (
            self.simulation.sim.neuron.h.t
            >= self.simulation.sim.neuron.h.tstop - self.simulation.sim.neuron.h.dt / 2
        ):
            # change buttons state
            self.end_simul()

    def start(self):
        """Start the simulation from beginning. Reload simulation config if needed."""
        # if config has changed: reload cell, sim, protocol, and figure frame
        if self.reload:
            self.reload_params()
        else:
            # destroy last simulation (i.e. this is not a start but a restart)
            if self.simulation.sim.neuron.h.t > 0:
                self.simulation.destroy()
                self.simulation.sim.neuron.h.t = 0
                self.clear_voltage_figure()

            # instantiate
            if self.simulation.cell.icell is None:
                self.simulation.instantiate()

        # change buttons state
        self.frames["FrameMain"].simul_running()

        # run
        self.play = True
        self.run_simul()

    def reload_params(self):
        """Reload cell, protocol, simulation, figure frame."""
        # destroy before reload
        self.simulation.destroy()  # destroy cell, sim and protocol, not config data

        # reload cell & protocol
        self.simulation.load_cell_sim()
        self.simulation.load_protocol()
        self.simulation.instantiate()

        # clear voltage data
        self.clear_voltage_figure()
        # display figures after simulation has been reset
        self.frames["FrameMain"].display(self.root, self.simulation)

        self.root.update()

        # remember that all changes have been loaded
        self.reload = False

    def reload_figure_frame(self):
        """Reload figure frame."""
        self.end_simul()
        self.frames["FrameMain"].destroy()

        # reload figure frame
        self.frames["FrameMain"] = FrameMain(self.root, self)
        self.frames["FrameMain"].grid(row=0, column=1)
        self.frames["FrameMain"].update_syn_display(self.root, self.simulation)

        self.root.update()

    def pause(self):
        """Pause the simulation."""
        # change buttons state
        self.frames["FrameMain"].simul_on_pause()

        self.play = False

    def continue_simul(self):
        """Unpause the simulation."""
        # change buttons state
        self.frames["FrameMain"].simul_running()

        # run
        self.play = True
        self.run_simul()

    def end_simul(self):
        """End the simulation."""
        self.frames["FrameMain"].simul_ended()
        self.play = False
