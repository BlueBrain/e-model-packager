"""Workflow to extract glusynapse cells."""

import csv
import glob
import os
import shutil
import subprocess
from pathlib import Path

import configparser
import json
from schema import SchemaError

import luigi
from emodelrunner.load import load_config
from e_model_packager.synaptic_plasticity.circuit import BluepyCircuit
from e_model_packager.config_decorator import ConfigDecorator
from e_model_packager.utils import cwd
from e_model_packager.synaptic_plasticity.extractors import extract_all
from e_model_packager.synaptic_plasticity.precell_configuration import (
    check_for_special_cell,
    get_amp_duration_spikedelay,
)
from e_model_packager.synaptic_plasticity.utils import get_output_path

workflow_config = ConfigDecorator(luigi.configuration.get_config())


class PrepareMEModelDirectory(luigi.Task):
    """Task to prepare the e-model directory."""

    layers = luigi.Parameter()
    pregid = luigi.IntParameter()
    postgid = luigi.IntParameter()
    source_dirs = luigi.ListParameter()

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
        # if folder is already created, delete it
        # (it is generally from a previously failed run)
        if Path(memodel_dir).is_dir():
            shutil.rmtree(memodel_dir)
        os.makedirs(memodel_dir)

        os.makedirs(os.path.join(memodel_dir, "morphology"))
        os.makedirs(os.path.join(memodel_dir, "synapses"))
        os.makedirs(os.path.join(memodel_dir, "protocols"))
        os.makedirs(os.path.join(memodel_dir, "config"))
        os.makedirs(os.path.join(memodel_dir, "config", "params"))

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
        # pylint: disable=not-an-iterable
        for source_dir in self.source_dirs:
            # prot_name has the form 'PulseFrequency_SpikeTrainDT'
            prot_name = Path(source_dir).stem

            spike_train_path = os.path.join(source_dir, "out.dat")
            output_dir = os.path.join(
                self.output_folder, "protocols", f"spiketrain_{prot_name}.dat"
            )
            shutil.copy(spike_train_path, output_dir)

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
    def get_final_dict(input_dir, emodels):
        """Get trimmed recipes and final dicts."""
        final_out = {}
        for emodel in emodels:
            # optimized params
            with open(
                os.path.join(input_dir, "params/final.json"), "r", encoding="utf-8"
            ) as final_file:
                final = json.load(final_file)[emodel]
            if emodel not in final_out:
                final_out[emodel] = final

        return final_out

    def copy_config_data(self, input_dir, output_dir, emodels):
        """Copy params, final and recipes from config."""
        # copy params
        self.copy_unoptimized_params(input_dir, output_dir, emodels)

        # get recipes and final
        final_out = self.get_final_dict(input_dir, emodels)

        # write final
        final_out_path = os.path.join(output_dir, "config/params/final.json")
        with open(final_out_path, "w", encoding="utf-8") as final_out_file:
            json.dump(final_out, final_out_file, indent=4)

    def run(self):
        """Create directory and copy data."""
        # pylint: disable=unsubscriptable-object
        # make dirs
        self.makedirs()

        # copy mechanisms
        self.copy_mechanisms()

        # scripts
        self.copy_scripts()

        # spike train
        self.copy_spike_train()

        # copy config data
        input_dir = workflow_config.get("paths", "emodel_config_dir")

        circuit = BluepyCircuit(os.path.join(self.source_dirs[0], "BlueConfig"))
        precell_emodel = circuit.get_emodel_attributes(self.pregid).name
        postcell_emodel = circuit.get_emodel_attributes(self.postgid).name
        self.copy_config_data(
            input_dir, self.output_folder, [postcell_emodel, precell_emodel]
        )

        # extract data from circuit
        circuitpath = workflow_config.get("paths", "circuitpath")
        extra_recipe = workflow_config.get("paths", "extra_recipe")
        recipes_path = os.path.join(input_dir, "recipes/recipes.json")
        extract_all(
            self.source_dirs,
            self.output_folder,
            self.pregid,
            self.postgid,
            circuitpath,
            extra_recipe,
            recipes_path,
        )

        Path(self.output().path).touch()


class RunPyScript(luigi.Task):
    """Task to run the python script for an emodel."""

    layers = luigi.Parameter()
    pregid = luigi.IntParameter()
    postgid = luigi.IntParameter()
    source_dirs = luigi.ListParameter()
    config_path = luigi.Parameter()

    def requires(self):
        """Requires the output paths to be made."""
        return [
            PrepareMEModelDirectory(
                self.layers, self.pregid, self.postgid, self.source_dirs
            ),
            PrecellConfig(self.layers, self.pregid, self.postgid, self.source_dirs),
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
            subprocess.call(["sh", "./run.sh", self.config_path])


class PrecellConfigTarget(luigi.Target):
    """Checks that the amplitude check has not been performed yet."""

    def __init__(self, layers, pregid, postgid):
        """Constructor."""
        self.layers = layers
        self.pregid = pregid
        self.postgid = postgid

    def exists(self):
        """Check if the spike delay is written in the configfiles."""
        output_dir = workflow_config.get("paths", "output")
        memodel_dir = get_output_path(
            output_dir, self.layers, self.pregid, self.postgid
        )

        if not Path(memodel_dir).is_dir():
            return False

        with cwd(memodel_dir):
            config_paths = glob.glob(os.path.join("config", "*.ini"))

        if not config_paths:
            return False

        for config_path in config_paths:
            # go to memodel directory to prevent the config validator
            # from raising path not found errors
            with cwd(memodel_dir):
                try:
                    print(config_path)
                    _ = load_config(config_path=config_path)
                # config validator raises an error if a key is not present.
                # we want to return False if
                # precell_amplitude, precell_spikedelay or precell_width are not present.
                except SchemaError:
                    return False

        return True


class PrecellConfig(luigi.Task):
    """Create config s.t. the precell runs as expected during repeated spikes."""

    layers = luigi.Parameter()
    pregid = luigi.IntParameter()
    postgid = luigi.IntParameter()
    source_dirs = luigi.ListParameter()
    amp = luigi.FloatParameter(default=1.0)
    max_step_duration = luigi.IntParameter(default=15)
    max_amp = luigi.FloatParameter(default=8.0)

    def requires(self):
        """Requires the output paths to be made."""
        return PrepareMEModelDirectory(
            self.layers,
            self.pregid,
            self.postgid,
            self.source_dirs,
        )

    def output(self):
        """Output."""
        return PrecellConfigTarget(
            layers=self.layers, pregid=self.pregid, postgid=self.postgid
        )

    def run(self):
        """Run cell and record spike delay."""
        # pylint: disable=not-an-iterable
        output_dir = workflow_config.get("paths", "output")
        memodel_dir = get_output_path(
            output_dir, self.layers, self.pregid, self.postgid
        )

        # take the highest frequency, since it is the hardest to adapt the protocol to
        burst_freq = 0
        for source_dir in self.source_dirs:
            new_burst_freq = int(Path(source_dir).name.split("Hz")[0])
            if new_burst_freq > burst_freq:
                burst_freq = new_burst_freq

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

        # write protocol data in each config
        config_paths = glob.glob(os.path.join(memodel_dir, "config", "*.ini"))
        for config_path in config_paths:
            new_config = configparser.ConfigParser()
            # do not use load_config in order to not load the default values
            new_config.read(config_path)
            new_config["Protocol"]["precell_width"] = str(step_duration)
            new_config["Protocol"]["precell_amplitude"] = str(amp)
            new_config["Protocol"]["precell_spikedelay"] = str(spikedelay)
            new_config = check_for_special_cell(
                new_config, self.layers, self.pregid, self.postgid
            )

            with open(config_path, "w", encoding="utf-8") as configfile:
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
                        args[key] = [
                            layer,
                            int(row["pregid"]),
                            int(row["postgid"]),
                            [source_dir],
                        ]
                    else:
                        args[key][3].append(source_dir)

        tasks = [
            f(*arg)
            for _, arg in args.items()
            for f in (PrecellConfig, PrepareMEModelDirectory)
        ]

        return tasks
