"""Custom synapse-related classes."""
import random
import bluepyopt.ephys as ephys


class MySynapse:
    """Attach a synapse to the simulation."""

    def __init__(
        self, sim, icell, synapse, section, seed, rng_settings_mode, synconf_dict
    ):
        """Constructor.

        Args:
            sim (NrnSimulator): simulator
            icell (Hoc Cell): cell to which attach the synapse
            synapse (dict) : synapse data
            section (): cell location where the synapse is attached to
            seed (int) : random seed number
            rng_settings_mode (str) : mode of the random number generator
            synconf_dict (dict) : synapse configuration
        """
        self.seed = seed
        self.rng_settings_mode = rng_settings_mode

        # the synapse is inhibitory
        if synapse["synapse_type"] < 100:
            self.hsynapse = sim.neuron.h.ProbGABAAB_EMS(synapse["seg_x"], sec=section)
            self.hsynapse.tau_d_GABAA = synapse["tau_d"]

            self.set_tau_r(sim, icell, synapse)

        # the synapse is excitatory
        elif synapse["synapse_type"] > 100:
            self.hsynapse = sim.neuron.h.ProbAMPANMDA_EMS(synapse["seg_x"], sec=section)
            self.hsynapse.tau_d_AMPA = synapse["tau_d"]

        self.hsynapse.Use = abs(synapse["use"])
        self.hsynapse.Dep = abs(synapse["dep"])
        self.hsynapse.Fac = abs(synapse["fac"])

        # set random number generator
        self.set_random_nmb_generator(sim, icell, synapse)

        self.hsynapse.synapseID = synapse["sid"]

        self.hsynapse.Nrrp = synapse["Nrrp"]

        self.execute_synapse_configuration(synconf_dict, synapse, sim)

        self.delay = synapse["delay"]
        self.weight = synapse["weight"]

    def set_random_nmb_generator(self, sim, icell, synapse):
        """Sets the random number generator."""
        if self.rng_settings_mode == "Random123":
            self.randseed1 = icell.gid + 250
            self.randseed2 = synapse["sid"] + 100
            self.randseed3 = 300

            self.hsynapse.setRNG(self.randseed1, self.randseed2, self.randseed3)

        if self.rng_settings_mode == "Compatibility":
            self.rndd = sim.neuron.h.Random()
            self.rndd.MCellRan4(
                synapse["sid"] * 100000 + 100,
                icell.gid + 250 + self.seed,
            )
            self.rndd.uniform(0, 1)

            self.hsynapse.setRNG(self.rndd)

    def set_tau_r(self, sim, icell, synapse):
        """Set tau_r_GABAA using random nmb generator."""
        self.rng = sim.neuron.h.Random()
        if self.rng_settings_mode == "Random123":
            self.rng.Random123(icell.gid + 250, synapse["sid"] + 100, 450)
        elif self.rng_settings_mode == "Compatibility":
            self.rng.MCellRan4(
                synapse["sid"] * 100000 + 100,
                icell.gid + 250 + self.seed,
            )
        self.rng.lognormal(0.2, 0.1)
        self.hsynapse.tau_r_GABAA = self.rng.repick()

    def execute_synapse_configuration(self, synconf_dict, synapse, sim, exec_all=False):
        """Create a hoc file configuring synapse."""
        for cmd, ids in synconf_dict.items():
            if synapse["sid"] in ids and (exec_all or "*" not in cmd):
                cmd = cmd.replace("%s", "\n%(syn)s")
                hoc_cmd = cmd % {"syn": self.hsynapse.hname()}
                hoc_cmd = "{%s}" % hoc_cmd
                sim.neuron.h(hoc_cmd)


class MyNrnMODPointProcessMechanism(ephys.mechanisms.Mechanism):
    """Class containing all the synapses."""

    def __init__(
        self, name, synapses_data, synconf_dict, seed, rng_settings_mode, comment=""
    ):
        """Constructor.

        Args:
            name (str): name of this object
            synapses_data (dict) : synapse data
            synconf_dict (dict) : synapse configuration
            seed (int) : random seed number
            rng_settings_mode (str) : mode of the random number generator
            comment (str) : comment
        """
        super(MyNrnMODPointProcessMechanism, self).__init__(name, comment)
        self.synapses_data = synapses_data
        self.synconf_dict = synconf_dict
        self.seed = seed
        self.rng_settings_mode = rng_settings_mode
        self.rng = None
        self.pprocesses = None

    @staticmethod
    def get_cell_section_for_synapse(synapse, icell):
        """Returns the cell section on which is the synapse."""
        if synapse["sectionlist_id"] == 0:
            section = icell.soma[synapse["sectionlist_index"]]
        elif synapse["sectionlist_id"] == 1:
            section = icell.dend[synapse["sectionlist_index"]]
        elif synapse["sectionlist_id"] == 2:
            section = icell.apic[synapse["sectionlist_index"]]
        elif synapse["sectionlist_id"] == 3:
            section = icell.axon[synapse["sectionlist_index"]]

        return section

    def instantiate(self, sim=None, icell=None):
        """Instantiate the synapses."""
        if self.rng_settings_mode == "Random123":
            self.rng = sim.neuron.h.Random()
            self.rng.Random123_globalindex(self.seed)

        self.pprocesses = []
        for synapse in self.synapses_data:
            # get section
            section = self.get_cell_section_for_synapse(synapse, icell)

            my_synapse = MySynapse(
                sim,
                icell,
                synapse,
                section,
                self.seed,
                self.rng_settings_mode,
                self.synconf_dict,
            )

            self.pprocesses.append(my_synapse)

    def destroy(self, sim=None):
        """Destroy mechanism instantiation."""
        self.pprocesses = None


class MyNrnNetStimStimulus(ephys.stimuli.Stimulus):
    """Current stimulus based on current amplitude and time series."""

    def __init__(
        self,
        locations=None,
        total_duration=None,
        interval=None,
        number=None,
        start=None,
        noise=0,
    ):
        """Constructor.

        Args:
            locations: synapse point process location to connect to
            total_duration: duration of run (ms)
            interval: time between spikes (ms)
            number: average number of spikes
            start: most likely start time of first spike (ms)
            noise: fractional randomness (0 deterministic,
                   1 negexp interval distrubtion)
        """
        super(MyNrnNetStimStimulus, self).__init__()
        if total_duration is None:
            raise ValueError("NrnNetStimStimulus: Need to specify a total duration")
        else:
            self.total_duration = total_duration

        self.locations = locations
        self.interval = interval
        self.number = number
        self.start = start
        self.noise = noise
        self.connections = {}

    def instantiate(self, sim=None, icell=None):
        """Run stimulus."""
        for location in self.locations:
            self.connections[location.name] = []
            for synapse in location.instantiate(sim=sim, icell=icell):
                netstim = sim.neuron.h.NetStim()
                netstim.interval = self.interval
                netstim.number = self.number
                netstim.start = self.start
                netstim.noise = self.noise
                netcon = sim.neuron.h.NetCon(
                    netstim, synapse.hsynapse, -30, synapse.delay, synapse.weight
                )

                self.connections[location.name].append((netcon, netstim))

    def destroy(self, sim=None):
        """Destroy stimulus."""
        self.connections = None

    def __str__(self):
        """String representation."""
        return (
            "Netstim at %s" % ",".join(location for location in self.locations)
            if self.locations is not None
            else "Netstim"
        )


class MyNrnVecStimStimulus(ephys.stimuli.Stimulus):
    """Current stimulus based on stochastic current amplitude."""

    def __init__(
        self,
        locations=None,
        total_duration=None,
        interval=None,
        start=None,
        seed=1,
        vecstim_random="python",
    ):
        """Constructor.

        Args:
            locations: synapse point process location to connect to
            total_duration: duration of run (ms)
            interval: time between spikes (ms)
            start: most likely start time of first spike (ms)
            seed: seed for random number generator
            vecstim_random: origin of the random nmb gener. for vecstim. can be python or neuron
        """
        super(MyNrnVecStimStimulus, self).__init__()
        if total_duration is None:
            raise ValueError("NrnNetStimStimulus: Need to specify a total duration")
        else:
            self.total_duration = total_duration

        self.locations = locations
        self.interval = interval
        self.start = start
        self.seed = seed
        self.vecstim_random = vecstim_random
        self.connections = {}

    def instantiate(self, sim=None, icell=None):
        """Run stimulus."""
        if self.vecstim_random == "python":
            random.seed(self.seed)
        else:
            rand = sim.neuron.h.Random(self.seed)
            rand.uniform(self.start, self.total_duration)

        for location in self.locations:
            self.connections[location.name] = []
            for synapse in location.instantiate(sim=sim, icell=icell):
                if self.vecstim_random == "python":
                    spike_train = [random.uniform(self.start, self.total_duration)]
                else:
                    spike_train = [rand.repick()]

                t_vec = sim.neuron.h.Vector(spike_train)
                vecstim = sim.neuron.h.VecStim()
                vecstim.play(t_vec, self.interval)
                netcon = sim.neuron.h.NetCon(
                    vecstim, synapse.hsynapse, -30, synapse.delay, synapse.weight
                )

                self.connections[location.name].append((netcon, vecstim, t_vec))

    def destroy(self, sim=None):
        """Destroy stimulus."""
        self.connections = None

    def __str__(self):
        """String representation."""
        return (
            "Vecstim at %s" % ",".join(location for location in self.locations)
            if self.locations is not None
            else "Vecstim"
        )
