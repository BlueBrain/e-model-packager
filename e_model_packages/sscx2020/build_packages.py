"""Workflow to build e-model packages."""

import os
import json
import collections
import luigi
import shutil
from luigi.contrib.simulate import RunAnywayTarget
from utils import read_circuit, NpEncoder, combine_names
from bluepy.v2 import Cell as bpcell


class BuildCircuit(luigi.ExternalTask):
    """The external task to makes sure the circuit config is present."""

    def output(self):
        """:return: Path to the circuit config file."""
        config = luigi.configuration.get_config()
        circuit_config_path = config.get("paths", "circuit")
        return luigi.LocalTarget(path=circuit_config_path)


class SelectGids(luigi.Task):
    """Selects gids from circuits and saves their ME types to a json file."""

    gids_per_metype = luigi.IntParameter(default=5)

    @property
    def config(self):
        """Returns the Luigi config."""
        return luigi.configuration.get_config()

    def requires(self):
        """The BuildCircuit task is a dependency of this task."""
        return BuildCircuit()

    def output(self):
        """The JSON output."""
        output_dir = self.config.get("paths", "output")
        return luigi.LocalTarget(os.path.join(output_dir, "metype_gids.json"))

    def run(self):
        """Creates me combos and dumps to json."""
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


class CreateMEmodelDirs(luigi.Task):
    """Creates me-model directories."""

    @property
    def config(self):
        """Returns the Luigi config."""
        return luigi.configuration.get_config()

    def requires(self):
        """ME type gids must be provided therefore SelectGids is required."""
        return SelectGids()

    def run(self):
        """Create directories for each metype and cell combo."""
        combo_tasks = []
        with self.input().open("r") as metype_gids_file:
            metype_gids = json.load(metype_gids_file)
            for mtype in metype_gids.keys():
                for etype in metype_gids[mtype]:
                    gids = metype_gids[mtype][etype]
                    for gid in gids:
                        combo_tasks.append(CopyMechanisms(mtype, etype, gid))

        self.output().done()
        yield combo_tasks

    def output(self):
        """Does not produce output."""
        return RunAnywayTarget(self)


class CopyMechanisms(luigi.Task):
    """Task to copy mechanisms to each memodel directory."""

    mtype = luigi.Parameter()
    etype = luigi.Parameter()
    gid = luigi.IntParameter()

    @property
    def config(self):
        """Returns the Luigi config."""
        return luigi.configuration.get_config()

    def requires(self):
        """The path needs to be created."""
        return CreateDir(self.mtype, self.etype, self.gid)

    def output(self):
        """Does not produce output."""
        return RunAnywayTarget(self)

    def run(self):
        """Copies the mechanisms to corresponding directories."""
        output_dir = os.path.join(self.input().path, "mechanisms")
        mecha_dir = self.config.get("paths", "mechanisms")
        mecha_files = self.config.get("mechanisms", "required_files").split(",")

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

    @property
    def config(self):
        """Returns the Luigi config."""
        return luigi.configuration.get_config()

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
        output_dir = self.config.get("paths", "output")
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
        """The CreateMEmodelDirs method is required."""
        return CreateMEmodelDirs()
