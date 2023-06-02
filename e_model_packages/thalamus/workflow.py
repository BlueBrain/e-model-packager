"""Workflow to build Thalamus e-model packages."""

import logging
import shutil
import json
from pathlib import Path
import os
import configparser
import subprocess
import collections

import numpy as np
import luigi
from luigi_tools.target import OutputLocalTarget
from emodelrunner.factsheets.morphology_features import (
    ThalamusMorphologyFactsheetBuilder,
)
from emodelrunner.factsheets.output import (
    get_stim_params_from_config_for_physiology_factsheet,
)
from emodelrunner.factsheets.physiology_features import physiology_factsheet_info
from emodelrunner.run import main as run_emodel
from emodelrunner.load import load_config

from e_model_packages.io import NpEncoder
from e_model_packages.utils import cwd
from e_model_packages.circuit import BluepyCircuit, BluepySimulation, SynapseExtractor
from e_model_packages.common_tasks import SmartTask, CreateSystemLog
from e_model_packages.config_decorator import ConfigDecorator
from e_model_packages.nwb.create_nwb import create_nwb, write_nwb

logging.basicConfig(level=logging.INFO)

workflow_config = ConfigDecorator(luigi.configuration.get_config())


class PackageTarget(OutputLocalTarget):
    """Specific target for first category outputs."""

    output_dir = workflow_config.get("paths", "output")
    # pylint: disable=unused-private-member
    __prefix = output_dir


def memodel_target(mtype, etype, gid, fname):
    """Return the target (fname) inside memodel directory."""
    memodel_file = Path(mtype) / etype / str(gid) / fname
    memodel_file = str(memodel_file).replace(":", "-")
    return PackageTarget(memodel_file)


class MemodelParameters(SmartTask):
    """Parameter class to contain common MeModel parameters across various tasks.

    Attributes:
        mtype: morphological type
        etype: electrophysiological type
        gid: id of cell in the circuit
    """

    mtype = luigi.Parameter()
    etype = luigi.Parameter()
    gid = luigi.IntParameter()


class CollectMEModels(luigi.WrapperTask):
    """Submits the model package preparation tasks."""

    configfile = luigi.Parameter()
    ngids = luigi.IntParameter()

    def requires(self):
        """Returns the tasks required."""
        circuit_config_path = workflow_config.get("paths", "circuit")

        circuit = BluepyCircuit(circuit_config_path)
        metype_gids = circuit.extract_circuit_metype_gids(self.ngids)
        metype_gids = self.remove_unwanted_morphologies(metype_gids)

        tasks = []
        for (mtype, etype), gids in metype_gids.items():
            logging.info(
                "Collecting emodel with mtype=%s, etype=%s and the gids:", mtype, etype
            )
            logging.info(gids)
            for gid in gids:
                tasks.extend(
                    (
                        CreateFactsheets(
                            mtype=mtype,
                            etype=etype,
                            gid=gid,
                            configfile=self.configfile,
                        ),
                        PrepareMEModelDirectory(mtype=mtype, etype=etype, gid=gid),
                        CreateNWB(
                            mtype=mtype,
                            etype=etype,
                            gid=gid,
                            configfile=self.configfile,
                        ),
                    )
                )
        return tasks

    @staticmethod
    def remove_unwanted_morphologies(metype_gids):
        """Removes the packages with undesired morphologies.

        Args:
            metype_gids (dict): key: mtype, etype tuple value: cell ids.

        Returns:
            dict: filtered version of the metype_gids input.
        """
        list_to_remove = [
            "LPLR_definitive_AD5-Clasca_20160111",
            "R281HI-22-03-18-dorsalneuron_somatodendritic60X_correctshrink",
            "R281HI-6-6-16_ventralneuron",
            "EP23HI-LPLC_shrinkcorrect_cont_PWcoord",
            "R1818-VP-RET-03-05-18shrinkcorrect2.40",
            "MS1742_VPL-Ret-2-v6-botons-contours_SHRINKcorr2.56",
            "TCneuron_EP36-S1_7-3-18",
        ]

        circuit_config_path = workflow_config.get("paths", "circuit")
        circuit = BluepyCircuit(circuit_config_path)

        for gids in metype_gids.values():
            for gid in gids.copy():
                cell = circuit.get_cell_attributes(gid)
                morph_name = cell.morphology_fname
                if any(n in morph_name for n in list_to_remove):
                    gids.remove(gid)
        return metype_gids


class CreateFactsheets(MemodelParameters):
    """Task that creates factsheets for an emodel package."""

    configfile = luigi.Parameter()

    def requires(self):
        """The directory must be present."""
        return [
            PrepareMEModelDirectory(mtype=self.mtype, etype=self.etype, gid=self.gid),
            RunPyScript(
                mtype=self.mtype,
                etype=self.etype,
                gid=self.gid,
                configfile=self.configfile,
            ),
        ]

    def output(self):
        """Morphology factsheet json."""
        return {
            "morphology": memodel_target(
                self.mtype, self.etype, self.gid, "factsheets/morphology_factsheet.json"
            ),
            "etype": memodel_target(
                self.mtype, self.etype, self.gid, "factsheets/etype_factsheet.json"
            ),
        }

    def run(self):
        """Write the factsheets."""
        self.save_morphology_factsheet()
        self.save_etype_factsheet()

    def save_morphology_factsheet(self):
        """Dump morphology factsheet json."""
        morphology = (
            PrepareMEModelDirectory(mtype=self.mtype, etype=self.etype, gid=self.gid)
            .output()["morphology"]
            .path
        )
        factsheet_builder = ThalamusMorphologyFactsheetBuilder(morphology)
        thal_morphometrics_dict = factsheet_builder.factsheet_dict()

        thal_morphometrics_dict = self.round_factsheet_values(
            thal_morphometrics_dict, n_digits=1
        )

        with open(self.output()["morphology"].path, "w", encoding="utf-8") as out_file:
            json.dump(thal_morphometrics_dict, out_file, indent=4, cls=NpEncoder)

    def save_etype_factsheet(self):
        """Write etype factsheet json."""
        protocol_key = "RMP"
        memodel_dir = Path(self.output()["etype"].path).parent.parent
        with cwd(memodel_dir):
            config = load_config(config_path=os.path.join("config", self.configfile))
            # is not exactly the same as self.mtype
            prefix = config.get("Morphology", "mtype")
            voltage_path = (
                Path("python_recordings") / f"{prefix}.{protocol_key}.soma.v.dat"
            )

            prot_path = config.get("Paths", "prot_path")

            (
                current_amplitude,
                stim_start,
                stim_duration,
            ) = get_stim_params_from_config_for_physiology_factsheet(
                prot_path, protocol_key
            )
            etype_factsheet_output = Path(self.output()["etype"].path)
            rel_output_path = (
                Path(etype_factsheet_output.parent.name) / etype_factsheet_output.name
            )

            voltage_data = np.loadtxt(voltage_path)

            physiology = physiology_factsheet_info(
                time=voltage_data[:, 0],
                voltage=voltage_data[:, 1],
                current_amplitude=current_amplitude,
                stim_start=stim_start,
                stim_duration=stim_duration,
            )

            physiology = self.round_factsheet_values(physiology, n_digits=1)

            with open(rel_output_path, "w", encoding="utf-8") as out_file:
                json.dump(physiology, out_file, indent=4, cls=NpEncoder)

    @staticmethod
    def round_factsheet_values(factsheet: dict, n_digits: int) -> dict:
        """Round the values into smaller decimal points."""
        factsheet_copy = factsheet.copy()
        for feature_value in factsheet_copy["values"]:
            feature_value["value"] = round(feature_value["value"], n_digits)
        return factsheet_copy


class PrepareMEModelDirectory(MemodelParameters):
    """Creates the MEModelDirectory with static files."""

    @property
    def cell_attributes(self):
        """Return the Cell attributes."""
        circuit_config_path = workflow_config.get("paths", "circuit")
        circuit = BluepyCircuit(circuit_config_path)
        return circuit.get_cell_attributes(self.gid)

    @property
    def emodel_attributes(self):
        """Returns the emodel attributes."""
        circuit_config_path = workflow_config.get("paths", "circuit")
        circuit = BluepyCircuit(circuit_config_path)
        return circuit.get_emodel_attributes(self.gid)

    def output(self):
        """Returns multiple outputs in a dict."""
        return {
            "morphology": memodel_target(
                self.mtype,
                self.etype,
                self.gid,
                f"morphology/{self.cell_attributes.morphology_fname}",
            ),
            "cell_info": memodel_target(
                self.mtype, self.etype, self.gid, "cell_info.json"
            ),
            "mechanisms": memodel_target(
                self.mtype, self.etype, self.gid, "mechanisms"
            ),
            "static_files": [
                memodel_target(self.mtype, self.etype, self.gid, x)
                for x in workflow_config.get("files", "package_files")
            ],
            "synapses": memodel_target(
                self.mtype, self.etype, self.gid, "synapses/synapses.tsv"
            ),
            "mtype_map": memodel_target(
                self.mtype, self.etype, self.gid, "synapses/mtype_map.tsv"
            ),
            "synconf": memodel_target(
                self.mtype, self.etype, self.gid, "synapses/synconf.tsv"
            ),
            "config_dir": memodel_target(self.mtype, self.etype, self.gid, "config"),
            "completed_file": memodel_target(
                self.mtype, self.etype, self.gid, ".memodel-dir-created.txt"
            ),
            "python_recordings_dir": memodel_target(
                self.mtype, self.etype, self.gid, "python_recordings"
            ),
        }

    def run(self):
        """Writes the package data."""
        circuit_config_path = workflow_config.get("paths", "circuit")

        simulation = BluepySimulation(circuit_config_path)
        morph_path = Path(simulation.morph_dir) / self.cell_attributes.morphology_fname
        # copy morphology
        shutil.copy(morph_path, self.output()["morphology"].path)
        self.write_cell_info(self.output()["cell_info"].path)

        # copy mechanisms
        shutil.copytree(
            workflow_config.get("paths", "mechanisms_dir"),
            self.output()["mechanisms"].path,
        )

        self.copy_static_files()

        self.extract_synapses(circuit_config_path)

        # config
        self.copy_emodel_config(emodel=self.emodel_attributes.name)

        # config to be filled
        self.fill_in_emodel_config(
            self.emodel_attributes.name,
            self.cell_attributes.morphology_fname,
        )

        python_recordings_dir = Path(self.output()["python_recordings_dir"].path)
        python_recordings_dir.mkdir(parents=True, exist_ok=True)

        Path(self.output()["completed_file"].path).touch()

    def extract_synapses(self, circuit_config_path):
        """Extracts the synapses from the circuit.

        Args:
            circuit_config_path (str): Path to circuit config.
        """
        # synapses
        syn_extractor = SynapseExtractor(circuit_config_path, self.gid)
        syn_extractor.load_synapses()
        syn_extractor.write_synapses_to_files(
            self.output()["synapses"].path,
            self.output()["mtype_map"].path,
            self.output()["synconf"].path,
        )

    def write_cell_info(self, cell_info_path):
        """Create cell_info.json file and write it."""
        me_type = "_".join((self.mtype, self.etype))
        name = "_".join((me_type, str(self.gid)))

        cell_info = {
            "cell name": name,
            "e-type": self.etype,
            "gid": self.gid,
            "layer": self.cell_attributes.layer,
            "m-type": self.mtype,
            "me-type": me_type,
            "morphology": self.cell_attributes.morphology_fname,
        }

        with open(cell_info_path, "w", encoding="utf-8") as out_file:
            json.dump(cell_info, out_file, indent=4, cls=NpEncoder)

    def copy_static_files(self) -> None:
        """Copies static package files."""
        for s_file in self.output()["static_files"]:
            source = (
                Path(workflow_config.get("paths", "static_files_dir"))
                / Path(s_file.path).name
            )
            destination = s_file.path
            shutil.copy(source, destination)

    def copy_emodel_config(self, emodel):
        """Copy config files into output directory."""
        emodel_config_dir = workflow_config.get("paths", "emodel_config_dir")
        emodel_config_dir = Path(emodel_config_dir) / emodel / "config"
        output_config_dir = self.output()["config_dir"].path
        output_dir = Path(output_config_dir).parent
        # recipes
        with open(
            Path(emodel_config_dir) / "recipes" / "recipes.json", "r", encoding="utf-8"
        ) as recipes_file:
            recipe = json.load(recipes_file)[emodel]
        params_path = recipe["params"]
        features_path = recipe["features"]
        protocol_path = recipe["protocol"]

        # protocols, params, features
        self.copy_emodel_config_files(
            emodel_config_dir,
            output_dir,
            emodel,
            params_path,
            features_path,
            protocol_path,
        )

        self.copy_local_config_files(output_config_dir)

        # add paths and mtype to config files
        self.add_recipe_data_to_config(
            output_config_dir,
            params_path,
            features_path,
            protocol_path,
            self.mtype,
        )

    @staticmethod
    def copy_emodel_config_files(
        input_dir, output_dir, emodel, params_path, features_path, protocol_path
    ):
        """Copy params, features and protocols from config."""
        # unoptimized params, features and protocols
        for path_name in [params_path, features_path, protocol_path]:
            input_path = Path(input_dir).parent / path_name
            output_path = Path(output_dir) / path_name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(input_path, output_path)

        # optimized params
        with open(
            Path(workflow_config.get("paths", "final_json_dir")) / "final.json",
            "r",
            encoding="utf-8",
        ) as final_file:
            final = json.load(final_file)[f"{emodel}_legacy"]

        final_out_path = Path(output_dir) / "config" / "params" / "final.json"
        with open(final_out_path, "w", encoding="utf-8") as final_out_file:
            json.dump({emodel: final}, final_out_file)

    @staticmethod
    def copy_local_config_files(output_config_dir):
        """Copies the local config and protocols to the destination."""
        input_dir = workflow_config.get("paths", "local_config_ini_dir")
        luigi_config_filenames = workflow_config.get("files", "emodel_config_files")

        # single ini file case
        if isinstance(luigi_config_filenames, str):
            luigi_config_filenames = [luigi_config_filenames]
        for config_filename in luigi_config_filenames:
            input_filepath = Path(input_dir) / config_filename
            output_filepath = Path(output_config_dir) / config_filename
            shutil.copy(input_filepath, output_filepath)

        # copy protocols
        shutil.copytree(
            Path(input_dir) / "protocols",
            Path(output_config_dir) / "protocols",
            dirs_exist_ok=True,
        )

    @staticmethod
    def add_recipe_data_to_config(
        output_config_dir,
        params_path,
        features_path,
        protocol_path,
        mtype,
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
                new_config["Morphology"]["mtype"] = mtype

                # write config file
                with open(file_.path, "w", encoding="utf-8") as configfile:
                    new_config.write(configfile)

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

        if "Paths" not in config_dict:
            config_dict["Paths"] = {}
        config_dict["Paths"]["memodel_dir"] = "."
        config_dict["Paths"]["output_dir"] = "%(memodel_dir)s/python_recordings"
        config_dict["Paths"]["params_path"] = "%(memodel_dir)s/config/params/final.json"
        config_dict["Paths"]["syn_dir"] = "%(memodel_dir)s/synapses"
        config_dict["Paths"]["syn_data_file"] = "synapses.tsv"
        config_dict["Paths"]["syn_conf_file"] = "synconf.txt"
        config_dict["Paths"]["syn_mtype_map"] = "mtype_map.tsv"

    def fill_in_emodel_config(self, emodel, morph_fname):
        """Fill in emodel config.

        Args:
        emodel(str): emodel name.
        morph_fname(str): morphology filename.
        """
        output_config_dir = self.output()["config_dir"].path

        for file_ in os.scandir(output_config_dir):
            if file_.is_file() and file_.path.split(".")[-1] == "ini":
                new_config = configparser.ConfigParser()
                new_config.read(file_.path)

                self.fill_in_config_default_values(new_config)

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


class RunPyScript(MemodelParameters):
    """Task to run the python script for an emodel.

    Attributes:
        configfile : name of config file in /config to use when running script
    """

    configfile = luigi.Parameter()

    def requires(self):
        """Requires the output paths to be made."""
        return PrepareMEModelDirectory(
            mtype=self.mtype,
            etype=self.etype,
            gid=self.gid,
        )

    def output(self):
        """Produces the python recordings."""
        return memodel_target(
            self.mtype,
            self.etype,
            self.gid,
            "python_recordings/.python-recordings-created.txt",
        )

    def run(self):
        """Executes the python script."""
        memodel_dir = Path(self.output().path).parent.parent
        with cwd(memodel_dir):
            subprocess.call(["sh", "./compile_mechanisms.sh"])
            run_emodel(config_path=os.path.join("config", self.configfile))

        Path(self.output().path).touch()


class CreateNWB(MemodelParameters):
    """Applies the protocols and saves the results in an NWB.

    Attributes:
        configfile : name of emodel config file containing protocol info.
    """

    configfile = luigi.Parameter()

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
            gid=self.gid,
            configfile=self.configfile,
        )

    def output(self):
        """NWB output."""
        return memodel_target(
            self.mtype,
            self.etype,
            self.gid,
            f"{self.mtype}_{self.etype}_{self.gid}.nwb",
        )

    def run(self):
        """Creates and saves the nwb containing protocol responses."""
        # fmt: off
        recordings_path = Path(
            RunPyScript(
                mtype=self.mtype,
                etype=self.etype,
                gid=self.gid,
                configfile=self.configfile,
            ).output().path).parent
        # fmt: on
        voltage_recording_paths = sorted(Path(recordings_path).glob("*.v.dat"))
        current_recording_paths = sorted(Path(recordings_path).glob("current_*.dat"))

        voltage_recordings = [np.loadtxt(volt) for volt in voltage_recording_paths]
        current_recordings = [np.loadtxt(curr) for curr in current_recording_paths]
        protocol_names = [x.name.split(".")[1] for x in voltage_recording_paths]

        zipped_protocols = zip(protocol_names, voltage_recordings, current_recordings)

        Protocol = collections.namedtuple("Protocol", ["name", "voltage", "current"])
        protocol_responses = [
            Protocol(name=x[0], voltage=x[1], current=x[2]) for x in zipped_protocols
        ]

        if not protocol_responses:
            raise ValueError("No protocol responses are found.")

        nwb = create_nwb(
            self.emodel_name, protocol_responses, "Simulated Thalamus cell"
        )

        write_nwb(nwb, self.output().path)


class ThalamusMicroStudioPackages(luigi.WrapperTask):
    """The skeleton task to perform the workflow."""

    def requires(self):
        """The required Tasks."""
        return [CreateSystemLog(), CollectMEModels()]
