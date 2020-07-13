"""Workflow to build e-model packages."""

import os
import json
import collections
import luigi
from utils import read_circuit, NpEncoder
from bluepy.v2 import Cell as bpcell


class BuildCircuit(luigi.ExternalTask):
    """
    The external task to makes sure the circuit config is present.
    """

    def output(self):
        """
        :return: Path to the circuit config file.
        """
        config = luigi.configuration.get_config()
        circuit_config_path = config.get("paths", "circuit")
        return luigi.LocalTarget(path=circuit_config_path)


class SelectGids(luigi.Task):
    """
    Selects the gids from the circutes and saves their me types to a json file.
    """

    gids_per_metype = luigi.IntParameter(default=5)

    @property
    def config(self):
        """Returns the Luigi config"""
        return luigi.configuration.get_config()

    def requires(self):
        return BuildCircuit()

    def output(self):
        output_dir = self.config.get("paths", "output")
        return luigi.LocalTarget(os.path.join(output_dir, "metype_gids.json"))

    def run(self):

        circuit_config_path = self.config.get("paths", "circuit")

        circuit, _ = read_circuit(circuit_config_path)

        metype_gids = {}

        metypes_df = circuit.cells.get(
            properties=[bpcell.MTYPE, bpcell.ETYPE, bpcell.LAYER]
        ).drop_duplicates()
        metypes = [(row["mtype"], row["etype"]) for _, row in metypes_df.iterrows()]
        print("Found %d me-types" % len(metypes))

        for mtype, etype in metypes:
            metype_gids[(mtype, etype)] = list(
                circuit.cells.ids(
                    {bpcell.MTYPE: mtype, bpcell.ETYPE: etype},
                    limit=self.gids_per_metype,
                )
            )
            print(
                "Found %d %s gids for: %s_%s"
                % (
                    len(metype_gids[(mtype, etype)]),
                    metype_gids[(mtype, etype)],
                    mtype,
                    etype,
                )
            )

        mtype_etype_gids = collections.defaultdict(dict)
        for (mtype, etype), gids in metype_gids.items():
            mtype_etype_gids[mtype][etype] = gids

        with self.output().open("w") as out_file:
            json.dump(mtype_etype_gids, out_file, indent=4, cls=NpEncoder)


class SSCX2020(luigi.Task):
    """The skeleton task to perform the workflow."""

    def requires(self):
        return SelectGids()

    def run(self):
        config = luigi.configuration.get_config()
        print(config)
        print(config.get("SSCX2020", "param"))
        print("SSCX2020 analysis is complete!")
