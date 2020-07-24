"""Workflow to build e-model packages."""

import os
import json
import collections
import shutil
import luigi
from luigi.contrib.simulate import RunAnywayTarget
from utils import read_circuit, NpEncoder, get_mecombo_emodels
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
            for gidx, gid in enumerate(gids):
                gidx = gidx + 1  # 1 indexed for users
                tasks.append(PrepareMEModelDirectory(mtype, etype, gid, gidx))

        return tasks

    def output(self):
        """The JSON output."""
        output_dir = workflow_config.get("paths", "output")
        return luigi.LocalTarget(os.path.join(output_dir, "metype_gids.json"))

    def run(self):
        """Write the JSON."""
        with self.output().open("w") as out_file:
            json.dump(self.mtype_etype_gids, out_file, indent=4, cls=NpEncoder)


class PrepareMEModelDirectory(luigi.Task):
    """Task to prepare the e-model directory.

    Attributes:
        mtype: morphological type
        etype: electrophysiological type
        gid: id of cell in the circuit
        gidx: index of cell
    """

    mtype = luigi.Parameter()
    etype = luigi.Parameter()
    gid = luigi.IntParameter()
    gidx = luigi.IntParameter()

    def output(self):
        """Does not produce output."""
        return RunAnywayTarget(self)

    def run(self):
        """Create me-model directories."""
        output_dir = workflow_config.get("paths", "output")
        memodels_dir = os.path.join(output_dir, "memodel_dirs")

        circuit_config_path = workflow_config.get("paths", "circuit")
        circuit, blueconfig = read_circuit(circuit_config_path)

        circ_morph_dir = os.path.join(blueconfig.Run["MorphologyPath"], "ascii")
        circ_emodel_dir = blueconfig.Run["METypePath"]

        scripts_dir = workflow_config.get("paths", "scripts_dir")
        script_files = workflow_config.get("files", "scripts")
        script_files = script_files.split(",")
        script_paths = []

        templates_dir = workflow_config.get("paths", "templates_dir")

        for script_file in script_files:
            script_path = os.path.join(scripts_dir, script_file)
            script_paths.append(script_path)

        mecombo_emodels, mecombo_thresholds, mecombo_hypamps = get_mecombo_emodels(
            blueconfig
        )

        mtype_dir = os.path.join(memodels_dir, self.mtype)
        metype_dir = os.path.join(mtype_dir, self.etype)

        metype = "%s_%s" % (self.mtype, self.etype)

        memodel_name = "%s_%d" % (metype, int(self.gidx))
        memodel_dir = os.path.join(metype_dir, memodel_name)
        os.makedirs(memodel_dir)
        memodel_morph_dir = os.path.join(memodel_dir, "morphology")
        os.makedirs(memodel_morph_dir)

        hocrec_dir = os.path.join(memodel_dir, "hoc_recordings")
        pyrec_dir = os.path.join(memodel_dir, "python_recordings")

        os.makedirs(hocrec_dir)
        os.makedirs(pyrec_dir)

        cell = circuit.cells.get(self.gid)
        morph = cell.morphology
        mecombo = cell.me_combo

        threshold = mecombo_thresholds[mecombo]
        holding = mecombo_hypamps[mecombo]

        morph_fname = "%s.asc" % morph
        morph_path = os.path.join(circ_morph_dir, morph_fname)

        emodel = mecombo_emodels[mecombo]
        emodel_fname = "%s.hoc" % emodel
        emodel_path = os.path.join(circ_emodel_dir, emodel_fname)

        memodel_mechanisms_dir = os.path.join(memodel_dir, "mechanisms")
        shutil.copy(morph_path, memodel_morph_dir)
        shutil.copy(emodel_path, memodel_dir)
        shutil.copytree(
            workflow_config.get("paths", "mechanisms_dir"), memodel_mechanisms_dir
        )

        for script_path in script_paths:
            shutil.copy(script_path, memodel_dir)

        template_vars = {}

        template_vars["constants.hoc"] = {
            "template_name": emodel,
            "gid": self.gid,
            "morph_dir": "morphology",
            "morph_fname": morph_fname,
        }
        template_vars["current_amps.dat"] = {
            "holding": holding,
            "amp1": 1.50 * threshold,
            "amp2": 2.00 * threshold,
            "amp3": 2.50 * threshold,
        }

        for template_fn, vars in template_vars.items():
            template_path = os.path.join(templates_dir, template_fn)
            template = open(template_path).read()
            content = template.format(**vars)

            output_path = os.path.join(memodel_dir, template_fn)
            open(output_path, "w").write(content)

        print("Created dir for %s" % memodel_name)

        self.output().done()


class SSCX2020(luigi.WrapperTask):
    """The skeleton task to perform the workflow."""

    def requires(self):
        """The ParseCircuit method is required."""
        return ParseCircuit()
