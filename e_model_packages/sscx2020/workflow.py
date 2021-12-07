"""Workflow to build e-model packages."""
# pylint: disable=wrong-import-position
# pylint: disable=wrong-import-order
# pylint: disable=import-error
# pylint: disable=too-many-lines

import configparser
import json
import os

if "PMI_RANK" in os.environ:
    del os.environ["PMI_RANK"]
import shutil
import subprocess
import sys
from pathlib import Path
import logging

import numpy as np
from tqdm import tqdm
import pandas as pd

import luigi


from e_model_packages.sscx2020.utils import (
    NpEncoder,
    cwd,
    get_output_path,
    LocalTargetCustom,
)
from e_model_packages.sscx2020.config_decorator import ConfigDecorator
from e_model_packages.circuit import BluepyCircuit, BluepySimulation, SynapseExtractor
from e_model_packages.nwb.create_nwb import create_nwb, write_nwb

from emodelrunner.run import main as run_emodel
from emodelrunner.load import load_sscx_config, get_hoc_paths_args
from emodelrunner.create_hoc import get_hoc, write_hocs, copy_features_hoc
from emodelrunner.factsheets.output import (
    write_metype_json_from_config,
    write_emodel_json,
)

from luigi_tools.task import RemoveCorruptedOutputMixin

sys.path.append(os.path.join("e_model_packages", "sscx2020", "extra_data", "scripts"))


workflow_config = ConfigDecorator(luigi.configuration.get_config())
# pylint: disable=too-many-locals
logging.basicConfig(level=logging.INFO)


class SmartTask(RemoveCorruptedOutputMixin, luigi.Task):
    """A smarter task that automatically removes output of failed tasks.

    This is to ensure that no corrupted or incomplete output
    gets generated if the Task fails unexpectedly.
    This is the default behaviour in other wfms such as snakemake.
    """

    RemoveCorruptedOutputMixin.clean_failed = luigi.BoolParameter(
        significant=False,
        default=True,
        description="Trigger to remove the outputs of the failed tasks.",
        parsing=luigi.BoolParameter.EXPLICIT_PARSING,
    )


class MemodelParameters(SmartTask):
    """Parameter class to contain common MeModel parameters across various tasks.

    Luigi design pattern to address the parameter explosion problem.
    Reference https://luigi.readthedocs.io/en/stable/api/luigi.util.html

    Attributes:
        mtype: morphological type
        etype: electrophysiological type
        region: circuit region
        gid: id of cell in the circuit
        gidx: index of cell
    """

    mtype = luigi.Parameter()
    etype = luigi.Parameter()
    region = luigi.Parameter()
    gid = luigi.IntParameter()
    gidx = luigi.IntParameter()


class RunScriptMixin:
    """Class with a function to easily get the output targets from runs."""

    def get_output_targets_from_run(self):
        """Get output targets for a python or hoc run."""
        if self.configfile in ["config_singlestep.ini", "config_singlestep_short.ini"]:
            path = os.path.join(self.output_path, "*.Step_150.soma.v.dat")
            return LocalTargetCustom(path)

        elif self.configfile in ["config_synapses.ini", "config_synapses_short.ini"]:
            path = os.path.join(self.output_path, "*.Synapses_Vecstim.soma.v.dat")
            return LocalTargetCustom(path)

        elif self.configfile in ["config_multistep.ini", "config_multistep_short.ini"]:
            output_list = []
            for idx in range(3):
                path = os.path.join(
                    self.output_path, f"*.Step_{150 + idx * 50}.soma.v.dat"
                )
                output_list.append(LocalTargetCustom(path))
            return output_list

        elif self.configfile == "config_factsheets.ini":
            path = os.path.join(self.output_path, "*.RmpRiTau.soma.v.dat")
            return LocalTargetCustom(path)

        elif self.configfile == "config_recipe_protocols.ini":
            # files all main protocols have in common
            output_list = []
            output_files = [
                "*.RMP.soma.v.dat",
                "*.Rin.soma.v.dat",
                "*.IV_-100.soma.v.dat",
                "*.bpo_holding_current.dat",
                "*.bpo_threshold_current.dat",
            ]
            for outfile in output_files:
                path = os.path.join(self.output_path, outfile)
                output_list.append(LocalTargetCustom(path))
            return output_list

        raise Exception(f"Configfile {self.configfile} was not expected.")


class CollectMEModels(luigi.WrapperTask):
    """Yield the model preparation tasks."""

    def requires(self):
        """The required Tasks."""
        metype_gids_path = workflow_config.get("paths", "metype_gids")
        df = pd.read_csv(metype_gids_path)

        df = df[["gid", "mtype", "etype", "region"]]
        grouped_df = df.groupby(["mtype", "etype", "region"])["gid"].apply(list)

        tasks = []
        for metype_region in tqdm(grouped_df.index):
            mtype, etype, region = metype_region
            for idx, gid in enumerate(grouped_df.loc[metype_region]):
                gidx = idx + 1
                tasks.append(
                    CreateHoc(
                        mtype=mtype,
                        etype=etype,
                        region=region,
                        gid=gid,
                        gidx=gidx,
                        configfile="config_multistep.ini",
                    )
                )
                tasks.append(
                    CreateFactsheets(
                        mtype=mtype,
                        etype=etype,
                        region=region,
                        gid=gid,
                        gidx=gidx,
                        configfile="config_multistep.ini",
                    )
                )
                tasks.append(
                    CreateNWB(
                        mtype=mtype,
                        etype=etype,
                        region=region,
                        gid=gid,
                        gidx=gidx,
                        configfile="config_multistep.ini",
                    )
                )
        return tasks


class CreateNWB(MemodelParameters):
    """Applies the protocols and saves the results in an NWB.

    Attributes:
        configfile : name of emodel config file containing protocol info.
    """

    configfile = luigi.Parameter(default="config_multistep.ini")

    @property
    def emodel_name(self):
        """Emodel name in cells standard."""
        circuit = BluepyCircuit(workflow_config.get("paths", "circuit"))
        emodel = circuit.get_emodel_attributes(self.gid)
        return emodel.name

    def requires(self):
        """Requires the output paths to be made."""
        return RunPyScript(
            mtype=self.mtype,
            etype=self.etype,
            region=self.region,
            gid=self.gid,
            gidx=self.gidx,
            configfile=self.configfile,
        )

    @property
    def memodel_dir(self):
        """Directory containing the memodel."""
        output_dir = workflow_config.get("paths", "output")
        memodel_dir = get_output_path(
            self.mtype, self.etype, self.region, self.gidx, output_dir
        )
        return memodel_dir

    def output(self):
        """NWB output."""
        file_name = f"{self.region}_{self.mtype}_{self.etype}_{self.gidx}.nwb"
        return luigi.LocalTarget(Path(self.memodel_dir) / file_name)

    def run(self):
        """Creates and saves the nwb containing protocol responses."""
        recordings_path = Path(self.memodel_dir) / "python_recordings"
        voltage_recording_paths = sorted(
            Path(recordings_path).glob("*.Step_*.soma.v.dat")
        )
        current_recording_paths = sorted(
            Path(recordings_path).glob("current_*.Step_*.dat")
        )

        voltage_recordings = [np.loadtxt(volt) for volt in voltage_recording_paths]
        current_recordings = [np.loadtxt(curr) for curr in current_recording_paths]

        nwb = create_nwb(self.emodel_name, voltage_recordings, current_recordings)

        write_nwb(nwb, self.output().path)


class CreateFactsheets(MemodelParameters):
    """Task to create a me-type factsheet json file."""

    configfile = luigi.Parameter(default="config_factsheet.ini")

    def requires(self):
        """Requires the script to have been copied in the main output directory."""
        tasks = RunPyScript(
            mtype=self.mtype,
            etype=self.etype,
            region=self.region,
            gid=self.gid,
            gidx=self.gidx,
            configfile=self.configfile,
        )

        return tasks

    def output(self):
        """The JSON output."""
        output_dir = workflow_config.get("paths", "output")
        memodel_dir = get_output_path(
            self.mtype, self.etype, self.region, self.gidx, output_dir
        )
        targets = {}
        keys = ["metype", "emodel"]
        factsheet_names = ["me_type_factsheeet.json", "e_model_factsheeet.json"]
        for key, factsheet_name in zip(keys, factsheet_names):
            targets[key] = luigi.LocalTarget(
                os.path.join(memodel_dir, "factsheets", factsheet_name)
            )

        return targets

    def run(self):
        """Creates the me-type json file."""
        output_dir = workflow_config.get("paths", "output")
        memodel_dir = get_output_path(
            self.mtype, self.etype, self.region, self.gidx, output_dir
        )
        factsheets_dir = "factsheets"
        protocol_key = "RmpRiTau"

        with cwd(memodel_dir):
            config = load_sscx_config(
                config_path=os.path.join("config", self.configfile)
            )
            # is not exactly the same as self.mtype
            prefix = config.get("Morphology", "mtype")
            voltage_path = Path("python_recordings") / (
                prefix + "." + protocol_key + ".soma.v.dat"
            )
            morph_path = config.get("Paths", "morph_path")
            metype_output_path = os.path.join(factsheets_dir, "me_type_factsheet.json")

            emodel = config.get("Cell", "emodel")
            features_path = config.get("Paths", "features_path")
            units_path = config.get("Paths", "units_path")
            unoptimized_params_path = config.get("Paths", "unoptimized_params_path")
            optimized_params_path = config.get("Paths", "params_path")

            write_metype_json_from_config(
                config,
                voltage_path,
                morph_path,
                metype_output_path,
                protocol_key=protocol_key,
            )

            with open(features_path, "r", encoding="utf-8") as features_file:
                features_dict = json.load(features_file)

            with open(units_path, "r", encoding="utf-8") as units_file:
                feature_units_dict = json.load(units_file)

            with open(
                unoptimized_params_path, "r", encoding="utf-8"
            ) as unoptimized_params_file:
                unoptimized_params_dict = json.load(unoptimized_params_file)

            with open(
                optimized_params_path, "r", encoding="utf-8"
            ) as optimized_params_file:
                optimized_params_dict = json.load(optimized_params_file)

            emodel_output_path = os.path.join(factsheets_dir, "e_model_factsheet.json")

            write_emodel_json(
                emodel,
                prefix,
                features_dict,
                feature_units_dict,
                unoptimized_params_dict,
                optimized_params_dict,
                emodel_output_path,
            )

        # remove extra output
        mechanisms_compilation_dir = os.path.join(memodel_dir, "x86_64")
        if os.path.isdir(mechanisms_compilation_dir):
            shutil.rmtree(os.path.join(memodel_dir, "x86_64"))


class PrepareMEModelDirectory(MemodelParameters):
    """Task to prepare the e-model directory."""

    @property
    def output_folder(self):
        """The directory containing the output files."""
        output_dir = workflow_config.get("paths", "output")

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        memodel_dir = get_output_path(
            self.mtype, self.etype, self.region, self.gidx, output_dir
        )
        return memodel_dir

    def output(self):
        """Produces an empty file after all steps are done."""
        completed_file = os.path.join(
            self.output_folder, ".prep-memodel-dir-completed.txt"
        )
        return luigi.LocalTarget(completed_file)

    def makedirs(self, memodel_morph_dir, synapses_dir):
        """Make directories."""
        memodel_dir = self.output_folder
        os.makedirs(memodel_dir)
        os.makedirs(synapses_dir)

        os.makedirs(memodel_morph_dir)
        os.makedirs(os.path.join(memodel_dir, "hoc_recordings"))
        os.makedirs(os.path.join(memodel_dir, "python_recordings"))
        os.makedirs(os.path.join(memodel_dir, "config"))
        os.makedirs(os.path.join(memodel_dir, "config", "features"))
        os.makedirs(os.path.join(memodel_dir, "config", "params"))
        os.makedirs(os.path.join(memodel_dir, "config", "protocols"))

    def write_cell_info(self, morphology, layer, output_dir):
        """Create cell_info.json file and write it."""
        cell_info_path = os.path.join(output_dir, "cell_info.json")

        me_type = "_".join((self.mtype, self.etype))
        name = "_".join((me_type, str(self.gidx)))

        cell_info = {
            "cell name": name,
            "e-type": self.etype,
            "region": self.region,
            "gid": self.gid,
            "layer": layer,
            "m-type": self.mtype,
            "me-type": me_type,
            "morphology": morphology,
        }

        with open(cell_info_path, "w", encoding="utf-8") as out_file:
            json.dump(cell_info, out_file, indent=4, cls=NpEncoder)

    @staticmethod
    def copy_morph(morph_fname, circ_morph_dir, memodel_morph_dir):
        """Copy morphology."""
        morph_path = os.path.join(circ_morph_dir, morph_fname)

        shutil.copy(morph_path, memodel_morph_dir)

    @staticmethod
    def copy_config_data(
        input_dir, output_dir, emodel, params_path, features_path, protocol_path
    ):
        """Copy params, features and recipes from config."""
        # unoptimized params
        input_params_path = Path(input_dir).parent / params_path
        output_params_path = Path(output_dir) / params_path
        shutil.copy(input_params_path, output_params_path)

        # features
        input_features_path = Path(input_dir).parent / features_path
        output_features_path = Path(output_dir) / features_path
        shutil.copy(input_features_path, output_features_path)

        # units
        input_units_path = Path(input_dir) / "features" / "units.json"
        output_units_path = Path(output_dir) / "config" / "features" / "units.json"
        shutil.copy(input_units_path, output_units_path)

        # protocols
        input_protocol_path = Path(input_dir).parent / protocol_path
        output_protocol_path = Path(output_dir) / protocol_path
        shutil.copy(input_protocol_path, output_protocol_path)

        # optimized params
        with open(
            Path(input_dir) / "params" / "final.json", "r", encoding="utf-8"
        ) as final_file:
            final = json.load(final_file)[emodel]

        final_out = {emodel: final}
        final_out_path = Path(output_dir) / "config" / "params" / "final.json"
        with open(final_out_path, "w", encoding="utf-8") as final_out_file:
            json.dump(final_out, final_out_file)

    @staticmethod
    def copy_config_files(input_dir, output_config_dir):
        """Copy the .ini config files into output config directory."""
        luigi_config_filenames = workflow_config.get("files", "emodel_config_files")
        for config_filename in luigi_config_filenames:
            input_filepath = os.path.join(input_dir, config_filename)
            output_filepath = os.path.join(output_config_dir, config_filename)
            shutil.copy(input_filepath, output_filepath)

    @staticmethod
    def copy_protocol_files(input_dir, output_config_dir):
        """Copy the protocol files into output config protocol directory."""
        luigi_config_filenames = workflow_config.get("files", "protocols_files")
        for config_filename in luigi_config_filenames:
            input_filepath = os.path.join(input_dir, "protocols", config_filename)
            output_filepath = os.path.join(
                output_config_dir, "protocols", config_filename
            )
            shutil.copy(input_filepath, output_filepath)

    @staticmethod
    def add_recipe_data_to_config(
        output_config_dir,
        params_path,
        features_path,
        protocol_path,
        mtype_from_recipe,
    ):
        """Add params, features and protocol paths and mtype to config files."""
        for file_ in os.scandir(output_config_dir):
            if file_.is_file() and file_.path.split(".")[-1] == "ini":
                new_config = configparser.ConfigParser()
                new_config.read(file_.path)
                if "Paths" not in new_config:
                    new_config["Paths"] = {}

                if "features_path" not in new_config["Paths"]:
                    new_config["Paths"]["features_path"] = features_path
                if "unoptimized_params_path" not in new_config["Paths"]:
                    new_config["Paths"]["unoptimized_params_path"] = params_path
                if "prot_path" not in new_config["Paths"]:
                    new_config["Paths"]["prot_path"] = protocol_path

                if "Morphology" not in new_config:
                    new_config["Morphology"] = {}
                new_config["Morphology"]["mtype"] = mtype_from_recipe

                # write config file
                with open(file_.path, "w", encoding="utf-8") as configfile:
                    new_config.write(configfile)

    def copy_config(self, output_dir, emodel):
        """Copy config files into output directory."""
        input_dir = workflow_config.get("paths", "emodel_config_dir")
        output_config_dir = os.path.join(output_dir, "config")
        # recipes
        with open(
            Path(input_dir) / "recipes" / "recipes.json", "r", encoding="utf-8"
        ) as recipes_file:
            recipe = json.load(recipes_file)[emodel]
        params_path = recipe["params"]
        features_path = recipe["features"]
        protocol_path = recipe["protocol"]
        recipe_morph = recipe["morphology"]
        if isinstance(recipe_morph, list):
            mtype_from_recipe = recipe_morph[0][0]
        else:
            mtype_from_recipe = "_"

        # protocols, params, features
        self.copy_config_data(
            input_dir, output_dir, emodel, params_path, features_path, protocol_path
        )

        # emodel config files
        self.copy_config_files(input_dir, output_config_dir)

        # protocol files
        self.copy_protocol_files(input_dir, output_config_dir)

        # add paths and mtype to config files
        self.add_recipe_data_to_config(
            output_config_dir,
            params_path,
            features_path,
            protocol_path,
            mtype_from_recipe,
        )

    @staticmethod
    def copy_templates(output_dir):
        """Copy mechanisms into output directory."""
        output_templates_dir = os.path.join(output_dir, "templates")
        shutil.copytree(
            workflow_config.get("paths", "templates_to_copy_dir"), output_templates_dir
        )

    def copy_mechanisms(self):
        """Copy mechanisms into output directory."""
        memodel_mechanisms_dir = os.path.join(self.output_folder, "mechanisms")
        shutil.copytree(
            workflow_config.get("paths", "mechanisms_dir"), memodel_mechanisms_dir
        )

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

    @staticmethod
    def get_apical_point_isec(apical_dir, morphology_fname):
        """Return the index of the section of the apical point."""
        apic_filepath = os.path.join(apical_dir, "apical_points_isec.json")
        with open(apic_filepath, "r", encoding="utf-8") as apic_file:
            apical_point_isecs = json.load(apic_file)

        morph_name = Path(morphology_fname).stem

        try:
            apical_point_isec = apical_point_isecs[morph_name]
        except KeyError:
            apical_point_isec = -1

        return apical_point_isec

    def fill_in_step_protocols(self, threshold, holding):
        """Fill in emodel config.

        Args:
        threshold(float): threshold current.
        holding(float): holding current.
        """
        output_config_dir = os.path.join(self.output_folder, "config", "protocols")

        for file_ in os.scandir(output_config_dir):
            if file_.is_file() and file_.path.split(".")[-1] == "json":
                with open(file_.path, "r", encoding="utf-8") as protocol_file:
                    new_protocol = json.load(protocol_file)

                if "Step_150" in new_protocol and "Main" not in new_protocol:
                    new_protocol["Step_150"]["stimuli"]["step"]["amp"] = (
                        1.50 * threshold
                    )
                    new_protocol["Step_150"]["stimuli"]["holding"]["amp"] = holding
                if "Step_200" in new_protocol and "Main" not in new_protocol:
                    new_protocol["Step_200"]["stimuli"]["step"]["amp"] = (
                        2.00 * threshold
                    )
                    new_protocol["Step_200"]["stimuli"]["holding"]["amp"] = holding
                if "Step_250" in new_protocol and "Main" not in new_protocol:
                    new_protocol["Step_250"]["stimuli"]["step"]["amp"] = (
                        2.50 * threshold
                    )
                    new_protocol["Step_250"]["stimuli"]["holding"]["amp"] = holding

                with open(file_.path, "w", encoding="utf-8") as f:
                    json.dump(new_protocol, f)

    @staticmethod
    def fill_in_config_default_values(config_dict):
        """Fill in config dict with default values.

        Args:
            config_dict (dict): the configuration dictionary
        """
        if "Cell" not in config_dict:
            config_dict["Cell"] = {}
        config_dict["Cell"]["celsius"] = "34"
        config_dict["Cell"]["v_init"] = "-80"

        if "Morphology" not in config_dict:
            config_dict["Morphology"] = {}
        config_dict["Morphology"]["do_replace_axon"] = "True"

        if "Sim" not in config_dict:
            config_dict["Sim"] = {}
        config_dict["Sim"]["dt"] = "0.025"
        config_dict["Sim"]["cvode_active"] = "False"

        if "Synapses" not in config_dict:
            config_dict["Synapses"] = {}
        config_dict["Synapses"]["seed"] = "846515"
        config_dict["Synapses"]["rng_settings_mode"] = "Random123"
        config_dict["Synapses"]["hoc_synapse_template_name"] = "hoc_synapses"

        if "Paths" not in config_dict:
            config_dict["Paths"] = {}
        config_dict["Paths"]["memodel_dir"] = "."
        config_dict["Paths"]["output_dir"] = "%(memodel_dir)s/python_recordings"
        config_dict["Paths"]["params_path"] = "%(memodel_dir)s/config/params/final.json"
        config_dict["Paths"][
            "units_path"
        ] = "%(memodel_dir)s/config/features/units.json"
        config_dict["Paths"]["templates_dir"] = "%(memodel_dir)s/templates"
        config_dict["Paths"][
            "cell_template_path"
        ] = "%(templates_dir)s/cell_template_neurodamus.jinja2"
        config_dict["Paths"][
            "run_hoc_template_path"
        ] = "%(templates_dir)s/run_hoc.jinja2"
        config_dict["Paths"][
            "createsimulation_template_path"
        ] = "%(templates_dir)s/createsimulation.jinja2"
        config_dict["Paths"][
            "synapses_template_path"
        ] = "%(templates_dir)s/synapses.jinja2"
        config_dict["Paths"][
            "replace_axon_hoc_path"
        ] = "%(templates_dir)s/replace_axon_hoc.hoc"
        config_dict["Paths"]["syn_dir_for_hoc"] = "%(memodel_dir)s/synapses"
        config_dict["Paths"]["syn_dir"] = "%(memodel_dir)s/synapses"
        config_dict["Paths"]["syn_data_file"] = "synapses.tsv"
        config_dict["Paths"]["syn_conf_file"] = "synconf.txt"
        config_dict["Paths"]["syn_hoc_file"] = "synapses.hoc"
        config_dict["Paths"]["syn_mtype_map"] = "mtype_map.tsv"
        config_dict["Paths"]["simul_hoc_file"] = "createsimulation.hoc"
        config_dict["Paths"]["cell_hoc_file"] = "cell.hoc"
        config_dict["Paths"]["run_hoc_file"] = "run.hoc"

    def fill_in_emodel_config(self, emodel, morph_fname, apical_point_isec):
        """Fill in emodel config.

        Args:
        emodel(str): emodel name.
        morph_fname(str): morphology filename.
        apical_point_isec(int): section index of the apical point.
        """
        output_config_dir = os.path.join(self.output_folder, "config")

        for file_ in os.scandir(output_config_dir):
            if file_.is_file() and file_.path.split(".")[-1] == "ini":
                new_config = configparser.ConfigParser()
                new_config.read(file_.path)

                self.fill_in_config_default_values(new_config)

                if "Protocol" not in new_config:
                    new_config["Protocol"] = {}
                # add apical point isec to config
                new_config["Protocol"]["apical_point_isec"] = str(apical_point_isec)

                if "Cell" not in new_config:
                    new_config["Cell"] = {}
                new_config["Cell"]["emodel"] = emodel
                new_config["Cell"]["gid"] = str(self.gid)

                if "Paths" not in new_config:
                    new_config["Paths"] = {}
                new_config["Paths"]["morph_path"] = os.path.join(
                    "morphology", morph_fname
                )

                # write config file
                with open(file_.path, "w", encoding="utf-8") as configfile:
                    new_config.write(configfile)

    def run(self):
        """Create me-model directories."""
        circuit_config_path = workflow_config.get("paths", "circuit")
        circuit = BluepyCircuit(circuit_config_path)

        memodel_dir = self.output_folder
        memodel_morph_dir = os.path.join(memodel_dir, "morphology")
        synapses_dir = os.path.join(memodel_dir, "synapses")

        # make dirs
        self.makedirs(memodel_morph_dir, synapses_dir)

        # get cell data
        cell = circuit.get_cell_attributes(self.gid)

        # create cell_info.json
        self.write_cell_info(cell.morphology, cell.layer, memodel_dir)

        simulation = BluepySimulation(circuit_config_path)
        # copy morphology
        self.copy_morph(cell.morphology_fname, simulation.morph_dir, memodel_morph_dir)

        # synapses
        syn_extractor = SynapseExtractor(circuit_config_path, self.gid)
        syn_extractor.load_synapses()

        synapse_tsv_filename = os.path.join(synapses_dir, "synapses.tsv")
        with open(synapse_tsv_filename, "w", encoding="utf-8") as synapse_tsv_file:
            synapse_tsv_file.write(syn_extractor.synapse_tsv_content)

        mtype_filename = os.path.join(synapses_dir, "mtype_map.tsv")
        with open(mtype_filename, "w", encoding="utf-8") as mtype_file:
            mtype_file.write(syn_extractor.mtype_map_content)

        synconf_filename = os.path.join(synapses_dir, "synconf.txt")
        with open(synconf_filename, "w", encoding="utf-8") as synconf_file:
            synconf_file.write(syn_extractor.synconf)

        # copy mechanisms
        self.copy_mechanisms()

        # scripts
        self.copy_scripts()

        cell_emodel = circuit.get_emodel_attributes(self.gid)

        # config
        self.copy_config(memodel_dir, emodel=cell_emodel.name)

        # templates to be copied
        self.copy_templates(memodel_dir)

        # get the apicla point section index
        apical_point_isec = self.get_apical_point_isec(
            simulation.morph_parent_dir, cell.morphology_fname
        )

        # protocols to be filled
        self.fill_in_step_protocols(
            cell_emodel.threshold_current, cell_emodel.holding_current
        )

        # config to be filled
        self.fill_in_emodel_config(
            cell_emodel.name,
            cell.morphology_fname,
            apical_point_isec,
        )

        Path(self.output().path).touch()


class CreateHoc(MemodelParameters):
    """Task to create the hoc file of an emodel.

    Attributes:
        configfile : name of config file in /config to use when creating hoc
    """

    configfile = luigi.Parameter(default=None)

    def requires(self):
        """Requires the output paths to be made."""
        return PrepareMEModelDirectory(
            mtype=self.mtype,
            etype=self.etype,
            region=self.region,
            gid=self.gid,
            gidx=self.gidx,
        )

    @property
    def output_folder(self):
        """Returns the path to the outputs directory."""
        workflow_output_dir = workflow_config.get("paths", "output")
        return get_output_path(
            self.mtype, self.etype, self.region, self.gidx, workflow_output_dir
        )

    def output(self):
        """Produces the hoc file."""
        output_path = self.output_folder
        filenames = ["cell.hoc", "run.hoc", "createsimulation.hoc"]

        return [
            luigi.LocalTarget(os.path.join(output_path, fname)) for fname in filenames
        ]

    def run(self):
        """Creates the hoc script."""
        with cwd(self.output_folder):
            config = load_sscx_config(
                config_path=os.path.join("config", self.configfile)
            )
            cell_hoc, syn_hoc, simul_hoc, run_hoc, main_protocol_hoc = get_hoc(
                config=config
            )

            hoc_paths = get_hoc_paths_args(config)
            if main_protocol_hoc:
                copy_features_hoc(config)
            write_hocs(
                hoc_paths,
                cell_hoc,
                simul_hoc,
                run_hoc,
                syn_hoc,
                main_protocol_hoc,
            )


class RunHoc(MemodelParameters, RunScriptMixin):
    """Task to run the hoc files for an emodel.

    Attributes:
        configfile : name of config file in /config to use when creating hoc
    """

    configfile = luigi.Parameter(default="config_multistep.ini")
    has_rerun_create_hoc = False

    def requires(self):
        """Requires the hoc file to have been created."""
        # The first time that luigi comes to check
        # if CreateHoc is complete, destroy the target.
        # This forces luigi to rerun CreateHoc
        # and still leaves the target complete afterwards.
        if not self.has_rerun_create_hoc:
            targets = CreateHoc(
                mtype=self.mtype,
                etype=self.etype,
                region=self.region,
                gid=self.gid,
                gidx=self.gidx,
                configfile=self.configfile,
            ).output()
            for target in targets:
                if target.exists():
                    target.remove()

            self.has_rerun_create_hoc = True

        return CreateHoc(
            mtype=self.mtype,
            etype=self.etype,
            region=self.region,
            gid=self.gid,
            gidx=self.gidx,
            configfile=self.configfile,
        )

    @property
    def output_path(self):
        """Directory containing the output."""
        workflow_output_dir = workflow_config.get("paths", "output")
        script_path = get_output_path(
            self.mtype, self.etype, self.region, self.gidx, workflow_output_dir
        )
        output_path = os.path.join(script_path, "hoc_recordings")
        return output_path

    def output(self):
        """Produces the hoc recordings."""
        return self.get_output_targets_from_run()

    def run(self):
        """Executes the hoc script."""
        workflow_output_dir = workflow_config.get("paths", "output")
        hoc_path = get_output_path(
            self.mtype, self.etype, self.region, self.gidx, workflow_output_dir
        )
        with cwd(hoc_path):
            subprocess.call(["sh", "./run_hoc.sh"])


class RunPyScript(MemodelParameters, RunScriptMixin):
    """Task to run the python script for an emodel.

    Attributes:
        configfile : name of config file in /config to use when running script
    """

    configfile = luigi.Parameter(default="config_multistep.ini")

    def requires(self):
        """Requires the output paths to be made."""
        return PrepareMEModelDirectory(
            mtype=self.mtype,
            etype=self.etype,
            region=self.region,
            gid=self.gid,
            gidx=self.gidx,
        )

    @property
    def output_path(self):
        """Directory containing the output."""
        workflow_output_dir = workflow_config.get("paths", "output")
        script_path = get_output_path(
            self.mtype, self.etype, self.region, self.gidx, workflow_output_dir
        )
        output_path = os.path.join(script_path, "python_recordings")
        return output_path

    def output(self):
        """Produces the python recordings."""
        return self.get_output_targets_from_run()

    def run(self):
        """Executes the python script."""
        output_dir = workflow_config.get("paths", "output")
        memodel_dir = get_output_path(
            self.mtype, self.etype, self.region, self.gidx, output_dir
        )

        with cwd(memodel_dir):
            subprocess.call(["sh", "./compile_mechanisms.sh", self.configfile])
            run_emodel(config_path=os.path.join("config", self.configfile))


class CreateSystemLog(SmartTask):
    """Task to log the modules and python packages used in the execution."""

    def output(self):
        """A log file to be written."""
        workflow_output_dir = workflow_config.get("paths", "output")
        return luigi.LocalTarget(os.path.join(workflow_output_dir, "system-state.log"))

    def run(self):
        """Writes down the loaded modules, pip packages and python version."""
        output_dir = workflow_config.get("paths", "output")
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        module_list = subprocess.run(
            ["modulecmd", "bash", "list"], capture_output=True, check=True
        )
        py_version = subprocess.run(
            ["python", "--version"], capture_output=True, check=True
        )
        pip_list = subprocess.run(["pip", "list"], capture_output=True, check=True)

        modules = " ".join(
            [x.decode("utf-8") for x in [module_list.stdout, module_list.stderr]]
        )
        python_ver, pip = [x.stdout.decode("utf-8") for x in [py_version, pip_list]]

        with self.output().open("w") as outfile:
            outfile.write(f"{modules}\n{python_ver}\n{pip}")


class DoRecordings(MemodelParameters):
    """Launch both RunHoc and RunPyScript.

    Attributes:
        configfile : name of config file in /config to use when running script / creating hoc
    """

    configfile = luigi.Parameter(default=None)

    def requires(self):
        """Launch both RunHoc and RunPyScript."""
        tasks = [
            RunHoc(
                mtype=self.mtype,
                etype=self.etype,
                region=self.region,
                gid=self.gid,
                gidx=self.gidx,
                configfile=self.configfile,
            ),
            RunPyScript(
                mtype=self.mtype,
                etype=self.etype,
                region=self.region,
                gid=self.gid,
                gidx=self.gidx,
                configfile=self.configfile,
            ),
        ]
        return tasks


class SSCX2020(luigi.WrapperTask):
    """The skeleton task to perform the workflow."""

    def requires(self):
        """The required Tasks."""
        return [CreateSystemLog(), CollectMEModels()]
