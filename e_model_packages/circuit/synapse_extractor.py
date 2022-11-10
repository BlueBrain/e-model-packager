"""For the retrieval of synapses using bglibpy."""

import re
import collections
import bglibpy


class SynapseExtractor:
    """Extracts synapses of a cell from the circuit."""

    def __init__(self, blueconfig, gid):
        """Constructor, inits bglibpy ssim object.

        Args:
            blueconfig (obj): blueconfig object or path to bloeconfig object
            gid ([type]): [description]
        """
        self.ssim = bglibpy.SSim(blueconfig, record_dt=0.1)
        self.circuit = self.ssim.circuit_access
        self.gid = gid
        self.synconf = None
        self.synapse_tsv_content = None
        self.mtype_map_content = None

    @staticmethod
    def convert_sec_name(sec_name):
        """Convert section name into sectionlist_id and section_list_index."""
        match = re.match(r"(.*)\[(.*)\]", sec_name)
        if match is None:
            raise Exception(f"Couldnt match section name {sec_name}")

        sectionlist_name = match.groups()[0]
        sectionlist_index = int(match.groups()[1])

        sectionlist_names = ["soma", "dend", "apic", "axon"]

        return sectionlist_names.index(sectionlist_name), sectionlist_index

    @staticmethod
    def generate_synconf_content(synconf_dict, synconf_ordering):
        """Generate content for synconf.txt."""
        # pylint: disable=consider-using-f-string
        synconf_content = ""
        for command in synconf_ordering:
            gids = synconf_dict[command]
            synconf_content += "%s\n%s\n" % (
                command,
                " ".join([str(x) for x in gids] + [str(-1e15)]),
            )

        return synconf_content

    @staticmethod
    def get_tau_d(synapse_dict):
        """Return tau_d given synapse type."""
        is_inhibitory = synapse_dict["syn_type"] < 100
        if is_inhibitory:
            return synapse_dict["synapse_parameters"]["tau_d_GABAA"]
        else:
            return synapse_dict["synapse_parameters"]["tau_d_AMPA"]

    def get_pre_mtype_id(self, mtype_map, pre_gid):
        """Assign pre-cell mtype to an id."""
        pre_mtype = self.circuit.get_cell_properties(pre_gid, "mtype").mtype
        if pre_mtype in mtype_map:
            # can use index. one occurence of pre_mtype & list is not long
            return mtype_map.index(pre_mtype)
        else:
            pre_mtype_id = len(mtype_map)
            mtype_map.append(pre_mtype)
            return pre_mtype_id

    @staticmethod
    def get_Nrrp(hsynapse):
        """Get Nrrp from hsynapse."""
        return hsynapse.Nrrp

    def load_synapses(
        self, add_stimuli=False, add_synapses=None, intersect_pre_gids=None
    ):
        """Loads synapses information."""
        # pylint: disable=too-many-locals, consider-using-f-string
        self.ssim.instantiate_gids(
            [self.gid],
            synapse_detail=2,
            add_replay=True,
            add_stimuli=add_stimuli,
            add_synapses=add_synapses,
            intersect_pre_gids=intersect_pre_gids,
        )

        cell_info_dict = self.ssim.cells[self.gid].info_dict
        cell = self.ssim.cells[self.gid]

        n_of_synapses = len(cell_info_dict["synapses"].items())

        # n_of_cols is actually not related to nmb of keys
        n_of_cols = 14

        self.synapse_tsv_content = f"{n_of_synapses} {n_of_cols}\n"

        synconf_dict = collections.defaultdict(list)
        synconf_ordering = []
        mtype_map = []

        for (synapse_id, synapse_dict), (_, synapse) in zip(
            cell_info_dict["synapses"].items(), cell.synapses.items()
        ):
            tau_d = self.get_tau_d(synapse_dict)

            delay = cell_info_dict["connections"][synapse_id]["post_netcon"]["delay"]
            weight = cell_info_dict["connections"][synapse_id]["post_netcon"]["weight"]

            pre_gid = synapse_dict["pre_cell_id"]
            post_sec_sectionlist_id, post_sec_sectionlist_index = self.convert_sec_name(
                synapse_dict["post_sec_name"]
            )

            # assign pre-cell mtype to an id
            pre_mtype_id = self.get_pre_mtype_id(mtype_map, pre_gid)

            # get synapse id without the ('', ) part.
            _, sid = synapse_id

            # do not save in scientific notation : hoc files can't read it.
            self.synapse_tsv_content += "%s\n" % "\t".join(
                [
                    str(x)
                    for x in [
                        sid,
                        pre_gid,
                        post_sec_sectionlist_id,
                        post_sec_sectionlist_index,
                        "%.3f" % synapse_dict["post_segx"],
                        synapse_dict["syn_type"],
                        synapse_dict["synapse_parameters"]["Dep"],
                        synapse_dict["synapse_parameters"]["Fac"],
                        synapse_dict["synapse_parameters"]["Use"],
                        tau_d,
                        delay,
                        weight,
                        self.get_Nrrp(synapse.hsynapse),
                        pre_mtype_id,
                    ]
                ]
            )
            for command in synapse_dict["synapseconfigure_cmds"]:
                if command not in synconf_ordering:
                    synconf_ordering.append(command)
                synconf_dict[command].append(sid)

        self.synconf = self.generate_synconf_content(synconf_dict, synconf_ordering)

        self.mtype_map_content = ""
        for idx, pre_mtype in enumerate(mtype_map):
            self.mtype_map_content += f"{idx} {pre_mtype}\n"

    def write_synapses_to_files(self, synapse_tsv_path, mtype_map_path, synconf_path):
        """Write the synapses information to files.

        Args:
            synapse_tsv_path (Path or string): output filepath for synapses
            mtype_map_path (Path or string): output filepath for mtype_map
            synconf_path (Path or string): output filepath for synconf
        """
        with open(synapse_tsv_path, "w", encoding="utf-8") as synapse_tsv_file:
            synapse_tsv_file.write(self.synapse_tsv_content)

        with open(mtype_map_path, "w", encoding="utf-8") as mtype_file:
            mtype_file.write(self.mtype_map_content)

        with open(synconf_path, "w", encoding="utf-8") as synconf_file:
            synconf_file.write(self.synconf)
