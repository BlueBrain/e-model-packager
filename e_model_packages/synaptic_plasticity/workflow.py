"""Workflow to extract glusynapse cells."""
import csv
import os
import shutil
import subprocess
from pathlib import Path

import luigi
from e_model_packages.sscx2020.config_decorator import ConfigDecorator
from e_model_packages.sscx2020.utils import cwd
from e_model_packages.synaptic_plasticity.extractors import extract_all
from e_model_packages.synaptic_plasticity.utils import get_output_path

workflow_config = ConfigDecorator(luigi.configuration.get_config())


class PrepareMEModelDirectory(luigi.Task):
    """Task to prepare the e-model directory."""

    layers = luigi.Parameter()
    pregid = luigi.IntParameter()
    postgid = luigi.IntParameter()
    source_dir = luigi.Parameter()

    @property
    def output_folder(self):
        """The directory containing the output files."""
        output_dir = workflow_config.get("paths", "output")

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        return get_output_path(output_dir, self.layers, self.pregid, self.postgid)

    def output(self):
        """Produces a .completed.txt file after all steps are done."""
        completed_file = os.path.join(self.output_folder, ".completed.txt")
        return luigi.LocalTarget(completed_file)

    def makedirs(self):
        """Make directories."""
        memodel_dir = self.output_folder
        os.makedirs(memodel_dir)

        os.makedirs(os.path.join(memodel_dir, "morphology"))
        os.makedirs(os.path.join(memodel_dir, "synapses"))
        os.makedirs(os.path.join(memodel_dir, "config"))
        os.makedirs(os.path.join(memodel_dir, "protocols"))

    def copy_scripts(self):
        """Copy scripts."""
        scripts_dir = workflow_config.get("paths", "scripts_dir")
        script_files = workflow_config.get("files", "memodel_scripts")

        # check if multuple files to copy -> list of str
        if isinstance(script_files, list):
            for script_file in script_files:
                script_path = os.path.join(scripts_dir, script_file)
                shutil.copy(script_path, self.output_folder)
        # or only one file -> str
        else:
            script_path = os.path.join(scripts_dir, script_files)
            shutil.copy(script_path, self.output_folder)

    def copy_mechanisms(self):
        """Copy mechanisms into output directory."""
        memodel_mechanisms_dir = os.path.join(self.output_folder, "mechanisms")
        shutil.copytree(
            workflow_config.get("paths", "mechanisms_dir"),
            memodel_mechanisms_dir,
        )

    def copy_spike_train(self):
        """Copy spike train data from pre-cell."""
        spike_train_path = os.path.join(self.source_dir, "out.dat")
        output_dir = os.path.join(self.output_folder, "protocols")
        shutil.copy(spike_train_path, output_dir)

    def copy_templates(self):
        """Copy mechanisms into output directory."""
        output_templates_dir = os.path.join(self.output_folder, "templates")
        shutil.copytree(
            workflow_config.get("paths", "templates_to_copy_dir"), output_templates_dir
        )

    def run(self):
        """Create directory and copy data."""
        # make dirs
        self.makedirs()

        # copy mechanisms
        self.copy_mechanisms()

        # scripts
        self.copy_scripts()

        # templates
        self.copy_templates()

        # spike train
        self.copy_spike_train()

        # extract data from circuit
        circuitpath = workflow_config.get("paths", "circuitpath")
        extra_recipe = workflow_config.get("paths", "extra_recipe")
        extract_all(
            self.source_dir, self.output_folder, self.postgid, circuitpath, extra_recipe
        )

        Path(self.output().path).touch()


class RunPyScript(luigi.Task):
    """Task to run the python script for an emodel."""

    layers = luigi.Parameter()
    pregid = luigi.IntParameter()
    postgid = luigi.IntParameter()
    source_dir = luigi.Parameter()

    def requires(self):
        """Requires the output paths to be made."""
        return PrepareMEModelDirectory(
            self.layers, self.pregid, self.postgid, self.source_dir
        )

    def output(self):
        """Produces the python recordings."""
        workflow_output_dir = workflow_config.get("paths", "output")
        memodel_dir = get_output_path(
            workflow_output_dir, self.layers, self.pregid, self.postgid
        )
        output_path = os.path.join(memodel_dir, "output.h5")

        return luigi.LocalTarget(output_path)

    def run(self):
        """Executes the python script."""
        output_dir = workflow_config.get("paths", "output")
        memodel_dir = get_output_path(
            output_dir, self.layers, self.pregid, self.postgid
        )

        with cwd(memodel_dir):
            subprocess.call(["sh", "./run.sh"])


class RunWorkflow(luigi.WrapperTask):
    """Task to extract all cells."""

    def requires(self):
        """Call PrepareMEModelDirectory for each cell."""
        args = {}

        index_dir = workflow_config.get("paths", "index")
        layers = workflow_config.get("circuit", "layers")

        # find each unique set of layer, pregid, postgid
        for layer in layers:
            index_file_name = "index_" + layer + ".csv"
            index_file_path = os.path.join(index_dir, index_file_name)
            with open(index_file_path, newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    source_dir = os.path.dirname(row["path"])
                    key = "_".join((layer, str(row["pregid"], str(row["postgid"]))))
                    if key not in args:
                        args[key] = (
                            layer,
                            int(row["pregid"]),
                            int(row["postgid"]),
                            source_dir,
                        )

        tasks = [PrepareMEModelDirectory(*arg) for _, arg in args.items()]

        return tasks
