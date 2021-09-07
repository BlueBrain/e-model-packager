"""Workflow to extract glusynapse cells."""
import csv
import os
import shutil
import subprocess
from pathlib import Path

import configparser
import json

import luigi
from emodelrunner.load import load_config
from e_model_packages.circuit import BluepyCircuit
from e_model_packages.sscx2020.config_decorator import ConfigDecorator
from e_model_packages.sscx2020.utils import cwd
from e_model_packages.synaptic_plasticity.extractors import extract_all
from e_model_packages.synaptic_plasticity.precell_configuration import (
    get_amp_duration_spikedelay,
)
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
        os.makedirs(os.path.join(memodel_dir, "protocols"))
        os.makedirs(os.path.join(memodel_dir, "config"))
        os.makedirs(os.path.join(memodel_dir, "config", "params"))
        os.makedirs(os.path.join(memodel_dir, "config", "recipes"))

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

    @staticmethod
    def copy_unoptimized_params(input_dir, output_dir, emodels):
        """Copy unoptimized params folder."""
        for emodel in emodels:
            # load recipes
            recipes_path = os.path.join(input_dir, "recipes/recipes.json")
            with open(recipes_path, "r", encoding="utf-8") as recipes_file:
                recipe = json.load(recipes_file)[emodel]

            # get params path
            params_path = recipe["params"]

            # copy unoptimized params
            input_params_path = Path(input_dir).parent / params_path
            output_params_path = os.path.join(output_dir, params_path)
            if not os.path.isfile(output_params_path):
                shutil.copy(input_params_path, output_params_path)

    @staticmethod
    def get_recipes_and_final_dicts(input_dir, emodels):
        """Get trimmed recipes and final dicts."""
        recipes_out = {}
        final_out = {}
        for emodel in emodels:
            # recipes
            recipes_path = os.path.join(input_dir, "recipes/recipes.json")
            with open(recipes_path, "r", encoding="utf-8") as recipes_file:
                recipe = json.load(recipes_file)[emodel]
            if emodel not in recipes_out:
                recipes_out[emodel] = recipe

            # optimized params
            with open(
                os.path.join(input_dir, "params/final.json"), "r", encoding="utf-8"
            ) as final_file:
                final = json.load(final_file)[emodel]
            if emodel not in final_out:
                final_out[emodel] = final

        return recipes_out, final_out

    def copy_config_data(self, input_dir, output_dir, emodels):
        """Copy params, final and recipes from config."""
        # copy params
        self.copy_unoptimized_params(input_dir, output_dir, emodels)

        # get recipes and final
        recipes_out, final_out = self.get_recipes_and_final_dicts(input_dir, emodels)

        # write recipes
        recipes_out_path = os.path.join(output_dir, "config", "recipes", "recipes.json")
        with open(recipes_out_path, "w", encoding="utf-8") as recipes_out_file:
            json.dump(recipes_out, recipes_out_file)

        # write final
        final_out_path = os.path.join(output_dir, "config/params/final.json")
        with open(final_out_path, "w", encoding="utf-8") as final_out_file:
            json.dump(final_out, final_out_file)

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

        # copy config data
        input_dir = workflow_config.get("paths", "emodel_config_dir")
        circuit = BluepyCircuit(os.path.join(self.source_dir, "BlueConfig"))
        precell_emodel = circuit.get_emodel_attributes(self.pregid).name
        postcell_emodel = circuit.get_emodel_attributes(self.postgid).name
        self.copy_config_data(
            input_dir, self.output_folder, [postcell_emodel, precell_emodel]
        )

        # extract data from circuit
        circuitpath = workflow_config.get("paths", "circuitpath")
        extra_recipe = workflow_config.get("paths", "extra_recipe")
        extract_all(
            self.source_dir,
            self.output_folder,
            self.pregid,
            self.postgid,
            circuitpath,
            extra_recipe,
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
        return [
            PrepareMEModelDirectory(
                self.layers, self.pregid, self.postgid, self.source_dir
            ),
            PrecellConfig(self.layers, self.pregid, self.postgid, self.source_dir),
        ]

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


class PrecellConfigTarget(luigi.Target):
    """Checks that the amplitude check has not been performed yet."""

    def __init__(self, layers, pregid, postgid, configfile="config_pairsim.ini"):
        """Constructor."""
        self.layers = layers
        self.pregid = pregid
        self.postgid = postgid
        self.configfile = configfile

    def exists(self):
        """Check if the spike delay is written in the configfile."""
        output_dir = workflow_config.get("paths", "output")
        memodel_dir = get_output_path(
            output_dir, self.layers, self.pregid, self.postgid
        )
        configfile_path = os.path.join(memodel_dir, "config", self.configfile)

        if not os.path.isfile(configfile_path):
            return False

        config = load_config(config_path=configfile_path)
        if not config.has_option("Protocol", "precell_amplitude"):
            return False
        if not config.has_option("Protocol", "precell_spikedelay"):
            return False
        if not config.has_option("Protocol", "precell_width"):
            return False

        return True


class PrecellConfig(luigi.Task):
    """Create config s.t. the precell runs as expected during repeated spikes."""

    layers = luigi.Parameter()
    pregid = luigi.IntParameter()
    postgid = luigi.IntParameter()
    source_dir = luigi.Parameter()
    amp = luigi.FloatParameter(default=1.0)
    max_step_duration = luigi.IntParameter(default=15)
    max_amp = luigi.FloatParameter(default=8.0)

    def requires(self):
        """Requires the output paths to be made."""
        return PrepareMEModelDirectory(
            self.layers,
            self.pregid,
            self.postgid,
            self.source_dir,
        )

    def output(self):
        """Output."""
        return PrecellConfigTarget(
            layers=self.layers, pregid=self.pregid, postgid=self.postgid
        )

    def run(self):
        """Run cell and record spike delay."""
        output_dir = workflow_config.get("paths", "output")
        memodel_dir = get_output_path(
            output_dir, self.layers, self.pregid, self.postgid
        )

        burst_freq = int(Path(self.source_dir).name.split("Hz")[0])
        # special case, fire only once
        if burst_freq == 0:
            burst_interval = 5000
        else:
            burst_interval = 1000.0 / burst_freq

        step_duration, amp, spikedelay = get_amp_duration_spikedelay(
            memodel_dir,
            amp=self.amp,
            max_step_duration=self.max_step_duration,
            burst_interval=burst_interval,
            max_amp=self.max_amp,
        )

        # write spike delay
        configfile_path = os.path.join(memodel_dir, "config", "config_pairsim.ini")
        new_config = configparser.ConfigParser()
        # do not use load_config in order to not load the default values
        new_config.read(configfile_path)
        new_config["Protocol"]["precell_width"] = str(step_duration)
        new_config["Protocol"]["precell_amplitude"] = str(amp)
        new_config["Protocol"]["precell_spikedelay"] = str(spikedelay)

        with open(configfile_path, "w", encoding="utf-8") as configfile:
            new_config.write(configfile)


class RunWorkflow(luigi.WrapperTask):
    """Task to extract all cells."""

    def requires(self):
        """Create MEModelDirectory for each cell."""
        args = {}

        index_dir = workflow_config.get("paths", "index")
        layers = workflow_config.get("circuit", "layers")

        # find each unique set of layer, pregid, postgid
        for layer in layers:
            index_file_name = "index_" + layer + ".csv"
            index_file_path = os.path.join(index_dir, index_file_name)
            with open(index_file_path, newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    source_dir = os.path.dirname(row["path"])
                    key = "_".join((layer, str(row["pregid"]), str(row["postgid"])))
                    if key not in args:
                        args[key] = (
                            layer,
                            int(row["pregid"]),
                            int(row["postgid"]),
                            source_dir,
                        )

        tasks = [
            f(*arg)
            for _, arg in args.items()
            for f in (PrecellConfig, PrepareMEModelDirectory)
        ]

        return tasks
