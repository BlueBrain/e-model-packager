"""Workflow to build e-model packages."""

import os
import json
import collections
import luigi
import shutil
from luigi.contrib.simulate import RunAnywayTarget
from utils import read_circuit, NpEncoder, combine_names
from bluepy.v2 import Cell as bpcell


workflow_config = luigi.configuration.get_config()


class ParseCircuit(luigi.Task):
    """Parse the circuit to get the number of mtypes etypes and cells."""

    gids_per_metype = luigi.IntParameter(default=5)
    mtype_etype_gids = collections.defaultdict(dict)

    def requires(self):
        """The BuildCircuit task is a dependency of this task."""
        circuit_config_path = workflow_config.get("paths", "circuit")
        circuit, _ = read_circuit(circuit_config_path)
        metype_gids = {}

        metypes_df = circuit.cells.get(
            properties=[bpcell.MTYPE, bpcell.ETYPE, bpcell.LAYER]
        ).drop_duplicates()
        metypes = [(row["mtype"], row["etype"]) for _, row in metypes_df.iterrows()]

        for mtype, etype in metypes:
            metype_gids[(mtype, etype)] = list(
                circuit.cells.ids(
                    {bpcell.MTYPE: mtype, bpcell.ETYPE: etype},
                    limit=self.gids_per_metype,
                )
            )

        tasks = []
        for (mtype, etype), gids in metype_gids.items():
            self.mtype_etype_gids[mtype][etype] = gids
            for gid in gids:
                tasks.append(CopyMechanisms(mtype, etype, gid))

        return tasks

    def output(self):
        """The JSON output."""
        output_dir = workflow_config.get("paths", "output")
        return luigi.LocalTarget(os.path.join(output_dir, "metype_gids.json"))

    def run(self):
        """Write the JSON."""
        with self.output().open("w") as out_file:
            json.dump(self.mtype_etype_gids, out_file, indent=4, cls=NpEncoder)


class CopyMechanisms(luigi.Task):
    """Task to copy mechanisms to each memodel directory."""

    mtype = luigi.Parameter()
    etype = luigi.Parameter()
    gid = luigi.IntParameter()

    def requires(self):
        """The path needs to be created."""
        return CreateDir(self.mtype, self.etype, self.gid)

    def output(self):
        """Does not produce output."""
        return RunAnywayTarget(self)

    def run(self):
        """Copies the mechanisms to corresponding directories."""
        output_dir = os.path.join(self.input().path, "mechanisms")
        mecha_dir = workflow_config.get("paths", "mechanisms")
        mecha_files = workflow_config.get("mechanisms", "required_files").split(",")

        for mecha_file in mecha_files:
            mecha_file_path = os.path.join(mecha_dir, mecha_file)
            output_path = os.path.join(output_dir, mecha_file)
            shutil.copy(mecha_file_path, output_path)

        self.output().done()


class CreateDir(luigi.Task):
    """Task to create a model directory given mtype, etype and gid."""

    mtype = luigi.Parameter()
    etype = luigi.Parameter()
    gid = luigi.IntParameter()

    def run(self):
        """Creates the nested directory."""
        output_path = self.output().path
        try:
            os.makedirs(output_path)
            os.makedirs(os.path.join(output_path, "mechanisms"))
        # handled this way to have py2.7 support
        except OSError:
            pass

    def output(self):
        """The nested directory."""
        output_dir = workflow_config.get("paths", "output")
        output_path = os.path.join(
            output_dir,
            "memodel_dirs",
            self.mtype,
            self.etype,
            combine_names(self.mtype, self.etype, self.gid),
        )
        return luigi.LocalTarget(output_path)


class CopyMorphology(luigi.Task):
    """Task to copy the morphology to each memodel directory."""

    mtype = luigi.Parameter()
    etype = luigi.Parameter()
    gid = luigi.IntParameter()


class CopyScripts(luigi.Task):
    """Task to copy scripts to each memodel directory."""

    mtype = luigi.Parameter()
    etype = luigi.Parameter()
    gid = luigi.IntParameter()


class SSCX2020(luigi.WrapperTask):
    """The skeleton task to perform the workflow."""

    def requires(self):
        """The ParseCircuit method is required."""
        return ParseCircuit()
