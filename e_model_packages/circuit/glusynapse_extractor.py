"""For the retrieval of synapses using bglibpy."""
import bglibpy

# use glu synapses
from glusynapseutils.simulation.synapse import Synapse as GluSynapse

from e_model_packages.circuit.synapse_extractor import SynapseExtractor

bglibpy.Synapse = GluSynapse


class GluSynapseExtractor(SynapseExtractor):
    """Extracts glusynapses of a cell from the circuit."""

    def get_postgid(self):
        """Get the post-synaptic cell's gid."""
        return list(self.ssim.bc_circuit.cells.ids("PostCell"))[0]

    def get_pregids(self):
        """Get the pre-synaptic cells' gid."""
        # pylint: disable=protected-access
        pregid = list(self.ssim.bc_circuit.cells.ids("PreCell"))[0]
        pregids = [pregid]
        # Special case: multiple connections
        # if "ExtraPreCell" in self.ssim.bc_circuit.cells.ids:
        if "ExtraPreCell" in self.ssim.bc_circuit.cells._targets._targets:
            pregids = pregids + list(self.ssim.bc_circuit.cells.ids("ExtraPreCell"))

        return pregids

    def set_gid(self, gid):
        """Set the gid of the cell from which the synapses are to be extracted."""
        self.gid = gid

    @staticmethod
    def get_Nrrp(hsynapse):
        """Get Nrrp from hsynapse."""
        return hsynapse.Nrrp_TM
