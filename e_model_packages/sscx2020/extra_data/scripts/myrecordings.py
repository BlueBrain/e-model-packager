"""Custom Recording class."""

import logging

import bluepyopt.ephys as ephys

logger = logging.getLogger(__name__)


class MyRecording(ephys.recordings.CompRecording):
    """Response to stimulus with recording every 0.1 ms."""

    def __init__(self, name=None, location=None, variable="v"):
        """Constructor.

        Args:
            name (str): name of this object
            location (Location): location in the model of the recording
            variable (str): which variable to record from (e.g. 'v')
        """
        super(MyRecording, self).__init__(
            name=name, location=location, variable=variable
        )

    def instantiate(self, sim=None, icell=None):
        """Instantiate recording."""
        logger.debug(
            "Adding compartment recording of %s at %s", self.variable, self.location
        )

        self.varvector = sim.neuron.h.Vector()
        seg = self.location.instantiate(sim=sim, icell=icell)
        self.varvector.record(getattr(seg, "_ref_%s" % self.variable), 0.1)

        self.tvector = sim.neuron.h.Vector()
        self.tvector.record(sim.neuron.h._ref_t, 0.1)  # pylint: disable=W0212

        self.instantiated = True
