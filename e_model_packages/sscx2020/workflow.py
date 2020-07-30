"""Workflow to build e-model packages."""

import os
import json
import collections
import shutil
import subprocess
import luigi
import numpy as np
from luigi.contrib.simulate import RunAnywayTarget
from bluepy.v2 import Cell as bpcell
from e_model_packages.sscx2020.utils import (
    read_circuit,
    NpEncoder,
    get_mecombo_emodels,
    combine_names,
    cwd,
)
from e_model_packages.sscx2020.config_decorator import ConfigDecorator


workflow_config = ConfigDecorator(luigi.configuration.get_config())


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
        output_dir = workflow_config.get("paths", "output")
        memodels_dir = os.path.join(output_dir, "memodel_dirs")
        mtype_dir = os.path.join(memodels_dir, self.mtype)
        metype_dir = os.path.join(mtype_dir, self.etype)
        memodel_name = combine_names(self.mtype, self.etype, self.gidx)
        memodel_dir = os.path.join(metype_dir, memodel_name)

        return luigi.LocalTarget(memodel_dir)

    @staticmethod
    def makedirs(memodel_dir, memodel_morph_dir):
        """Make directories"""
        os.makedirs(memodel_dir)
        os.makedirs(memodel_morph_dir)
        os.makedirs(os.path.join(memodel_dir, "hoc_recordings"))
        os.makedirs(os.path.join(memodel_dir, "python_recordings"))

    def copy_morph_emodel(
            self, circuit, blueconfig,
            mecombo_emodels, memodel_dir, memodel_morph_dir
        ):
        """Copy morphology and emodel"""
        circ_morph_dir = os.path.join(blueconfig.Run["MorphologyPath"], "ascii")
        circ_emodel_dir = blueconfig.Run["METypePath"]

        cell = circuit.cells.get(self.gid)
        mecombo = cell.me_combo

        morph_fname = "%s.asc" % cell.morphology
        morph_path = os.path.join(circ_morph_dir, morph_fname)

        emodel = mecombo_emodels[mecombo]
        emodel_path = os.path.join(circ_emodel_dir, "%s.hoc" % emodel)

        shutil.copy(morph_path, memodel_morph_dir)
        shutil.copy(emodel_path, memodel_dir)

        return mecombo, morph_fname, emodel

    @staticmethod
    def copy_mechanisms(memodel_dir):
        """Copy mechanisms into output directory"""
        memodel_mechanisms_dir = os.path.join(memodel_dir, "mechanisms")
        shutil.copytree(
            workflow_config.get("paths", "mechanisms_dir"), memodel_mechanisms_dir
        )

    @staticmethod
    def copy_scripts(memodel_dir):
        """Copy scripts"""
        scripts_dir = workflow_config.get("paths", "scripts_dir")
        script_files = workflow_config.get("files", "scripts")

        for script_file in script_files:
            script_path = os.path.join(scripts_dir, script_file)
            shutil.copy(script_path, memodel_dir)

    @staticmethod
    def write_down_using_templates(memodel_dir, templates_dir, template_vars):
        """Fill in and write constants.hoc & current_amp.dat templates given templates & vars"""
        for template_fn, variables in template_vars.items():
            template_path = os.path.join(templates_dir, template_fn)
            template = open(template_path).read()
            content = template.format(**variables)

            output_path = os.path.join(memodel_dir, template_fn)
            open(output_path, "w").write(content)

    def fill_in_templates(
            self, mecombo_thresholds, mecombo_hypamps, mecombo,
            emodel, morph_fname, memodel_dir
        ):
        """Fill in and write constants.hoc & current_amp.dat templates"""
        templates_dir = workflow_config.get("paths", "templates_dir")

        threshold = mecombo_thresholds[mecombo]
        holding = mecombo_hypamps[mecombo]

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

        self.write_down_using_templates(memodel_dir, templates_dir, template_vars)

    def run(self):
        """Create me-model directories."""
        circuit_config_path = workflow_config.get("paths", "circuit")
        circuit, blueconfig = read_circuit(circuit_config_path)

        mecombo_emodels, mecombo_thresholds, mecombo_hypamps = get_mecombo_emodels(
            blueconfig
        )

        memodel_dir = self.output().path
        memodel_morph_dir = os.path.join(memodel_dir, "morphology")

        # make dirs
        self.makedirs(memodel_dir, memodel_morph_dir)

        # morph & emodel
        mecombo, morph_fname, emodel = self.copy_morph_emodel(
            circuit, blueconfig, mecombo_emodels,
            memodel_dir, memodel_morph_dir
        )

        #mechanisms
        self.copy_mechanisms(memodel_dir)

        self.copy_scripts(memodel_dir)

        # template
        self.fill_in_templates(
            mecombo_thresholds, mecombo_hypamps, mecombo,
            emodel, morph_fname, memodel_dir
        )


class RunHoc(luigi.Task):
    """Task to run the hoc files for an emodel.

    Attributes:
        mtype: morphological type
        etype: electrophysiological type
        gidx: index of cell
    """

    mtype = luigi.Parameter()
    etype = luigi.Parameter()
    gidx = luigi.IntParameter()

    def output(self):
        """Produces the hoc recordings."""
        output_list = []

        inner_folder_name = combine_names(self.mtype, self.etype, self.gidx)
        recording_path = os.path.join(self.mtype, self.etype, inner_folder_name)

        workflow_output_dir = workflow_config.get("paths", "output")
        script_path = os.path.join(workflow_output_dir, "memodel_dirs", recording_path)
        output_path = os.path.join(script_path, "hoc_recordings")

        for idx in range(3):
            output_list.append(
                luigi.LocalTarget(
                    os.path.join(output_path, "soma_voltage_step%d.dat" % (idx + 1))
                )
            )

        return output_list

    def run(self):
        """Executes the hoc script."""
        inner_folder_name = combine_names(self.mtype, self.etype, self.gidx)
        recording_path = os.path.join(self.mtype, self.etype, inner_folder_name)

        workflow_output_dir = workflow_config.get("paths", "output")
        hoc_path = os.path.join(workflow_output_dir, "memodel_dirs", recording_path)
        with cwd(hoc_path):
            subprocess.call(["sh", "./run_hoc.sh"])


class RunPyScript(luigi.Task):
    """Task to run the python script for an emodel.

    Attributes:
        mtype: morphological type
        etype: electrophysiological type
        gidx: index of cell
    """

    mtype = luigi.Parameter()
    etype = luigi.Parameter()
    gidx = luigi.IntParameter()

    def output(self):
        """Produces the python recordings."""
        output_list = []

        inner_folder_name = combine_names(self.mtype, self.etype, self.gidx)
        recording_path = os.path.join(self.mtype, self.etype, inner_folder_name)

        workflow_output_dir = workflow_config.get("paths", "output")
        script_path = os.path.join(workflow_output_dir, "memodel_dirs", recording_path)
        output_path = os.path.join(script_path, "python_recordings")

        for idx in range(3):
            output_list.append(
                luigi.LocalTarget(
                    os.path.join(output_path, "soma_voltage_step%d.dat" % (idx + 1))
                )
            )

        return output_list

    def run(self):
        """Executes the python script."""
        inner_folder_name = combine_names(self.mtype, self.etype, self.gidx)
        recording_path = os.path.join(self.mtype, self.etype, inner_folder_name)

        workflow_output_dir = workflow_config.get("paths", "output")
        script_path = os.path.join(workflow_output_dir, "memodel_dirs", recording_path)
        with cwd(script_path):
            subprocess.call(["sh", "./run_py.sh"])


class CompareVoltages(luigi.Task):
    """Task to compare the voltages produced via python and hoc.

    Attributes:
        mtype: morphological type
        etype: electrophysiological type
        gidx: index of cell
    """

    mtype = luigi.Parameter()
    etype = luigi.Parameter()
    gidx = luigi.IntParameter()

    def requires(self):
        """The RunHoc and RunPyScript methods are required."""
        required_tasks = []
        required_tasks.append(RunHoc(self.mtype, self.etype, self.gidx))
        required_tasks.append(RunPyScript(self.mtype, self.etype, self.gidx))
        required_tasks.append(ParseCircuit())

        return required_tasks

    def output(self):
        """Does not produce output."""
        return RunAnywayTarget(self)

    def run(self):
        """Reads both dat files produced and checks if they are close.

        Raises a ValueError otherwise.
        """
        threshold = 1e-3
        recording_files = self.input()
        hoc_recs = recording_files[0]
        py_recs = recording_files[1]
        for hoc_rec, py_rec in zip(hoc_recs, py_recs):
            hoc_voltage = np.loadtxt(hoc_rec.path)
            py_voltage = np.loadtxt(py_rec.path)

            rms = np.sqrt(np.mean((hoc_voltage - py_voltage) ** 2))
            if rms > threshold:
                raise ValueError(
                    "the difference between hoc recordings and python is %f" % rms
                )
        print("the rms values are bounded within the %f threshold" % threshold)
        self.output().done()


class SSCX2020(luigi.WrapperTask):
    """The skeleton task to perform the workflow."""

    def requires(self):
        """The ParseCircuit method is required."""
        return ParseCircuit()
