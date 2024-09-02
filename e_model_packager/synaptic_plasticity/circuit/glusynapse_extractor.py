"""For the retrieval of synapses using bglibpy."""

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

import bglibpy

# use glu synapses
from glusynapseutils.simulation.synapse import Synapse as GluSynapse

from e_model_packager.synaptic_plasticity.circuit import SynapseExtractor

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
