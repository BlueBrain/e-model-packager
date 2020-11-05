"""Class containing simulation for the GUI."""

import os
import numpy as np

import bluepyopt.ephys as ephys
from recordings import RecordingCustom
from morphology import NrnFileMorphologyCustom
from recordings import RecordingCustom
from cell import CellModelCustom
from synapse import (
    NrnMODPointProcessMechanismCustom,
    NrnNetStimStimulusCustom,
    NrnVecStimStimulusCustom,
)
from load import (
    load_config,
    define_protocols,
    load_syn_locs,
    load_syn_mechs,
    get_axon_hoc,
    define_parameters,
    load_params,
    load_mechanisms,
    find_param_file,
    load_constants,
)


def section_coordinate_3d(sec, seg_pos, syn_type):
    """Returns the 3d coordinate of a point in a section.

    Args:
        sec: neuron section
        seg_pos (float): postion of the segment os the section
            (should be between 0 and 1)
        syn_type (int): synaptic type. excitatory if >100,
            inhibitory if <100
    """
    n3d = sec.n3d()

    arc3d = [sec.arc3d(i) for i in range(n3d)]
    x3d = [sec.x3d(i) for i in range(n3d)]
    y3d = [sec.y3d(i) for i in range(n3d)]
    z3d = [sec.z3d(i) for i in range(n3d)]

    if seg_pos in arc3d:
        idx = arc3d.index(seg_pos)
        local_x = x3d[idx]
        local_y = y3d[idx]
        local_z = z3d[idx]
    else:
        for i, arc in enumerate(arc3d):
            if arc > seg_pos:
                proportion = (seg_pos - arc3d[i - 1]) / (arc - arc3d[i - 1])
                local_x = x3d[i - 1] + proportion * (x3d[i] - x3d[i - 1])
                local_y = y3d[i - 1] + proportion * (y3d[i] - y3d[i - 1])
                local_z = z3d[i - 1] + proportion * (z3d[i] - z3d[i - 1])

    if syn_type > 100:
        syn_type_ = 1
    else:
        syn_type_ = 0

    return [local_x, local_y, local_z, syn_type_]


class NeuronSimulation:
    """Class containing BPO cell, simulation & protocol.

    Attributes:
        config (dict): dictionnary containing configuration data
        cell_path (str): path to cell repo. should be "."
        total_duration (int): duration of cell simulation (ms)
        steps (list of floats): default step stimuli (mV)
        default_hypamp (float): default holding stimulus (mV)
        step_stim (float): selected step stimulus (mV)
        hypamp (float): selected holding stimulus (mV)
        step_delay (int): delay before applying step stimulus (ms)
        step_duration (int): duration of step stimulus (ms)
        hold_step_delay (int): delay of holding stimulus (ms)
        hold_step_duration (int): duration of holding stimulus (ms)
        available_pre_mtypes (dict): all synapses pre_mtypes
            {mtypeidx: mtype_name, ...}
        pre_mtypes (list of int): selected pre_mtypes to run
            [mtypeidx, ...]
        netstim_params (dict): netstim parameters for synapses of each mtype
            {mtypeidx:[start, interval, number, noise]}
        syn_start (int): default time (ms) at which the synapse starts firing
        syn_interval (int): default interval (ms) between two synapse firing
        syn_nmb_of_spikes (int): default number of synapse firing
        syn_noise (int): default synapse noise
        cell (CellModelCustom): BluePyOpt-based cell
        release_params (dict): optimised cell parameters to fill in
            the cell's free parameters
        sim (ephys.simulators.NrnSimulator): BluePyOpt simulator
            can access neuron data from it
        syn_display_data (dict): synapse data (position and type) for display
            syn_display_data[pre_mtype] = [x,y,z,type],
            type=0 if inhib, type=1 if excit
    """

    def __init__(self, config_file="config.ini"):
        """Constructor. Load default params from config file."""
        # load config file
        self.config = load_config(filename=config_file)
        self.cell_path = self.config.get("Paths", "memodel_dir")

        # get default params
        self.load_protocol_params()
        self.load_synapse_params()

    def load_protocol_params(self):
        """Load default protocol params."""
        self.total_duration = self.config.getint("Protocol", "total_duration")

        # step protocol params
        self.steps, self.default_hypamp = self.load_default_step_stim()
        self.step_stim = self.steps[0]
        self.hypamp = self.default_hypamp

        self.step_delay = self.config.getint("Protocol", "stimulus_delay")
        self.step_duration = self.config.getint("Protocol", "stimulus_duration")
        self.hold_step_delay = self.config.getint("Protocol", "hold_stimulus_delay")
        self.hold_step_duration = self.config.getint(
            "Protocol", "hold_stimulus_duration"
        )

    def load_synapse_params(self):
        """Load default synapse params."""
        # mtypes to be chosen from {mtypeidx: mtype_name, ...}
        self.available_pre_mtypes = self.load_available_pre_mtypes()
        # mtypes to be loaded [mtypeidx, ...]
        self.pre_mtypes = []
        # synapse netstim param depending on mtype {mtypeidx:[start, interval, number, noise]}
        self.netstim_params = {}

        # default synapse params
        self.syn_start = self.config.getint("Protocol", "syn_start")
        self.syn_interval = self.config.getint("Protocol", "syn_interval")
        self.syn_nmb_of_spikes = self.config.getint("Protocol", "syn_nmb_of_spikes")
        self.syn_noise = self.config.getint("Protocol", "syn_noise")

    def reload_config_paths(self, mtype, etype, gidx):
        """When the cell has changed, re-set config paths."""
        self.config.set("Cell", "mtype", mtype)
        self.config.set("Cell", "etype", etype)
        self.config.set("Cell", "gidx", str(gidx))
        self.cell_path = self.config.get("Paths", "memodel_dir")

    def load_available_pre_mtypes(self):
        """Load the list of pre mtype cells to which are connected the synapses."""
        mtype_path = os.path.join(
            self.config.get("Paths", "syn_dir"),
            self.config.get("Paths", "syn_mtype_map"),
        )
        with open(mtype_path, "r") as mtype_file:
            raw_mtypes = mtype_file.readlines()

        # mtypes[id] = m-type name
        mtypes = {}
        for line in raw_mtypes:
            line = line.rstrip().split()
            if line:
                mtypes[int(line[0])] = line[1]

        return mtypes

    def load_default_step_stim(self):
        """Load the default step & holding stimuli."""
        amp_filename = os.path.join(
            self.config.get("Paths", "protocol_amplitudes_dir"),
            self.config.get("Paths", "protocol_amplitudes_file"),
        )

        with open(amp_filename, "r") as f:
            data = f.read().rstrip()
        amps = data.split()
        amplitudes = [float(amp) for amp in amps[1:]]  # do not take 1st value (hypamp)
        hypamp = float(amps[0])

        return amplitudes, hypamp

    def get_syn_stim(self):
        """Create syanpse stimuli."""
        if self.pre_mtypes:
            syn_locs = load_syn_locs(self.cell)
            syn_total_duration = self.total_duration
            # netstim
            return NrnNetStimStimulusCustom(
                syn_locs,
                syn_total_duration,
            )
        return None

    def load_protocol(self, protocol_name="protocol"):
        """Load BPO protocol."""
        syn_stim = self.get_syn_stim()

        soma_loc = ephys.locations.NrnSeclistCompLocation(
            name="soma", seclist_name="somatic", sec_index=0, comp_x=0.5
        )

        rec = RecordingCustom(name=protocol_name, location=soma_loc, variable="v")

        # create step stimulus
        stim = ephys.stimuli.NrnSquarePulse(
            step_amplitude=self.step_stim,
            step_delay=self.step_delay,
            step_duration=self.step_duration,
            location=soma_loc,
            total_duration=self.total_duration,
        )

        # create holding stimulus
        hold_stim = ephys.stimuli.NrnSquarePulse(
            step_amplitude=self.hypamp,
            step_delay=self.hold_step_delay,
            step_duration=self.hold_step_duration,
            location=soma_loc,
            total_duration=self.total_duration,
        )

        # create protocol
        stims = [stim, hold_stim]
        if syn_stim is not None:
            stims.append(syn_stim)

        self.protocol = ephys.protocols.SweepProtocol(
            protocol_name, stims, [rec], False
        )

    def create_cell_custom(self):
        """Create a cell. Returns cell, release params and time step."""
        # load constants
        constants_path = os.path.join(
            self.config.get("Paths", "constants_dir"),
            self.config.get("Paths", "constants_file"),
        )
        emodel, morph_dir, morph_fname, dt_tmp, gid = load_constants(constants_path)

        # load morphology path
        if self.config.has_option("Paths", "morph_dir"):
            morph_dir = self.config.get("Paths", "morph_dir")
        else:
            morph_dir = os.path.join(self.config.get("Paths", "memodel_dir"), morph_dir)
        if self.config.has_option("Paths", "morph_file"):
            morph_fname = self.config.get("Paths", "morph_file")

        morph_path = os.path.join(morph_dir, morph_fname)

        # load mechanisms
        recipes_path = os.path.join(
            self.config.get("Paths", "recipes_dir"),
            self.config.get("Paths", "recipes_file"),
        )
        params_filename = find_param_file(recipes_path, emodel)
        mechs = load_mechanisms(params_filename)

        # add synapses mechs
        # always load synapse data for synapse display.
        # -> do not need to reload syn data each time user toggle synapse checkbox
        mechs += [load_syn_mechs(self.config, self.pre_mtypes, self.netstim_params)]

        # load parameters
        params_path = os.path.join(
            self.config.get("Paths", "params_dir"),
            self.config.get("Paths", "params_file"),
        )
        release_params = load_params(params_filename=params_path, emodel=emodel)
        params = define_parameters(params_filename)

        # create morphology
        axon_hoc_path = os.path.join(
            self.config.get("Paths", "replace_axon_hoc_dir"),
            self.config.get("Paths", "replace_axon_hoc_file"),
        )
        replace_axon_hoc = get_axon_hoc(axon_hoc_path)
        do_replace_axon = self.config.getboolean("Morphology", "do_replace_axon")
        do_set_nseg = self.config.getint("Morphology", "do_set_nseg")
        morph = NrnFileMorphologyCustom(
            morph_path,
            do_replace_axon=do_replace_axon,
            replace_axon_hoc=replace_axon_hoc,
            do_set_nseg=do_set_nseg,
        )

        # create cell
        cell = CellModelCustom(
            name=emodel,
            morph=morph,
            mechs=mechs,
            params=params,
            gid=gid,
        )

        return cell, release_params, dt_tmp

    def load_cell_sim(self):
        """Load BPO cell & simulation."""
        self.cell, self.release_params, dt_tmp = self.create_cell_custom()

        if self.config.has_section("Sim") and self.config.has_option("Sim", "dt"):
            dt = self.config.getfloat("Sim", "dt")
        else:
            dt = dt_tmp
        self.sim = ephys.simulators.NrnSimulator(dt=dt, cvode_active=False)

    def load_synapse_display_data(self):
        """Load dict containing x,y,z of each synapse & inhib/excit."""
        # self.syn_display_data[pre_mtype] = [x,y,z,type], type=0 if inhib, type=1 if excit
        self.syn_display_data = {}
        for key in self.available_pre_mtypes:
            self.syn_display_data[key] = []

        for mech in self.cell.mechanisms:
            if hasattr(mech, "pprocesses"):
                for syn in mech.synapses_data:
                    pre_mtype = syn["pre_mtype"]
                    seg_pos = syn["seg_x"]
                    # check if a synapse of the same mtype has already the same position
                    # and add synapse only if a new position has to be displayed
                    syn_section = mech.get_cell_section_for_synapse(
                        syn, self.cell.icell
                    )
                    syn_display_data = section_coordinate_3d(
                        syn_section, seg_pos, syn["synapse_type"]
                    )
                    if syn_display_data not in self.syn_display_data[pre_mtype]:
                        self.syn_display_data[pre_mtype].append(syn_display_data)

    def instantiate(self):
        """Instantiate cell, simulation & protocol."""
        self.cell.freeze(self.release_params)
        self.cell.instantiate(sim=self.sim)
        self.protocol.instantiate(sim=self.sim, icell=self.cell.icell)
        self.sim.neuron.h.tstop = self.protocol.total_duration
        self.sim.neuron.h.stdinit()

    def destroy(self):
        """Destroy cell & protocol."""
        self.protocol.destroy(sim=self.sim)
        self.cell.destroy(sim=self.sim)
        self.cell.unfreeze(self.release_params.keys())

    def get_voltage(self):
        """Returns voltage response."""
        responses = {
            recording.name: recording.response for recording in self.protocol.recordings
        }
        key = list(responses.keys())[0]
        resp = responses[key]
        return np.array(resp["time"]), np.array(resp["voltage"])
