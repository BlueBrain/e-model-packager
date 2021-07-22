"""Workflow to build e-model packages."""
# pylint: disable=wrong-import-position
# pylint: disable=wrong-import-order
# pylint: disable=import-error

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

from tqdm import tqdm
import pandas as pd

import luigi


from e_model_packages.sscx2020.utils import (
    NpEncoder,
    cwd,
    get_output_path,
)
from e_model_packages.sscx2020.config_decorator import ConfigDecorator
from e_model_packages.circuit import BluepyCircuit, BluepySimulation, SynapseExtractor

from emodelrunner.run import main as run_emodel
from emodelrunner.load import load_config, get_hoc_paths_args
from emodelrunner.create_hoc import get_hoc, write_hocs
from emodelrunner.write_factsheets import (
    write_metype_json_from_config,
    write_emodel_json_from_config,
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
        nwb_env = workflow_config.get("nwb", "env")
        nwb_script = workflow_config.get("nwb", "script")
        nwb_output = self.output().path
        recordings_path = Path(self.memodel_dir) / "python_recordings"
        cmd = (
            f"{nwb_env} {nwb_script} --emodel_name={self.emodel_name}"
            f" --recordings_path={recordings_path} --output_file={nwb_output}"
        )
        cmd = cmd.split(" ")
        subprocess.run(cmd, check=True)


class CreateFactsheets(MemodelParameters):
    """Task to create a me-type factsheet json file."""

    configfile = luigi.Parameter(default="config_singlestep.ini")

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
        targets = []
        for fname in [
            "me_type_factsheeet.json",
            "e_model_factsheeet.json",
            "morphology_factsheeet.json",
        ]:
            targets.append(
                luigi.LocalTarget(os.path.join(memodel_dir, "factsheets", fname))
            )
        return targets

    def run(self):
        """Creates the me-type json file."""
        output_dir = workflow_config.get("paths", "output")
        memodel_dir = get_output_path(
            self.mtype, self.etype, self.region, self.gidx, output_dir
        )

        factsheets_dir = "factsheets"
        with cwd(memodel_dir):
            config = load_config(filename=self.configfile)
            write_metype_json_from_config(config, factsheets_dir)
            write_emodel_json_from_config(config, factsheets_dir)

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

        with open(cell_info_path, "w") as out_file:
            json.dump(cell_info, out_file, indent=4, cls=NpEncoder)

    @staticmethod
    def copy_morph(morph_fname, circ_morph_dir, memodel_morph_dir):
        """Copy morphology."""
        morph_path = os.path.join(circ_morph_dir, morph_fname)

        shutil.copy(morph_path, memodel_morph_dir)

    @staticmethod
    def copy_config(output_dir):
        """Copy python recordings config into output directory."""
        input_dir = workflow_config.get("paths", "emodel_config_dir")
        output_config_dir = os.path.join(output_dir, "config")
        shutil.copytree(
            input_dir,
            output_config_dir,
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

    def fill_in_emodel_config(self, threshold, holding, emodel, morph_fname):
        """Fill in emodel config.

        Args:
        threshold(float): threshold current.
        holding(float): holding current.
        emodel(str): emodel name.
        morph_fname(str): morphology filename.
        """
        output_config_dir = os.path.join(self.output_folder, "config")

        for file_ in os.scandir(output_config_dir):
            if file_.is_file() and file_.path.split(".")[-1] == "ini":
                new_config = configparser.ConfigParser()
                new_config.read(file_.path)

                if "Protocol" not in new_config:
                    new_config["Protocol"] = {}
                new_config["Protocol"]["hold_amp"] = str(holding)
                new_config["Protocol"]["stimulus_amp1"] = str(1.50 * threshold)
                new_config["Protocol"]["stimulus_amp2"] = str(2.00 * threshold)
                new_config["Protocol"]["stimulus_amp3"] = str(2.50 * threshold)

                if "Cell" not in new_config:
                    new_config["Cell"] = {}
                new_config["Cell"]["celsius"] = "34"
                new_config["Cell"]["v_init"] = "-80"
                new_config["Cell"]["emodel"] = emodel
                new_config["Cell"]["gid"] = str(self.gid)

                if "Sim" not in new_config:
                    new_config["Sim"] = {}
                new_config["Sim"]["dt"] = "0.025"

                if "Paths" not in new_config:
                    new_config["Paths"] = {}
                new_config["Paths"]["morph_dir"] = "morphology"
                new_config["Paths"]["morph_file"] = morph_fname

                # write config file
                with open(file_.path, "w") as configfile:
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
        with open(synapse_tsv_filename, "w") as synapse_tsv_file:
            synapse_tsv_file.write(syn_extractor.synapse_tsv_content)

        mtype_filename = os.path.join(synapses_dir, "mtype_map.tsv")
        with open(mtype_filename, "w") as mtype_file:
            mtype_file.write(syn_extractor.mtype_map_content)

        synconf_filename = os.path.join(synapses_dir, "synconf.txt")
        with open(synconf_filename, "w") as synconf_file:
            synconf_file.write(syn_extractor.synconf)

        # copy mechanisms
        self.copy_mechanisms()

        # scripts
        self.copy_scripts()

        # config
        self.copy_config(memodel_dir)

        # templates to be copied
        self.copy_templates(memodel_dir)

        cell_emodel = circuit.get_emodel_attributes(self.gid)

        # config to be filled
        self.fill_in_emodel_config(
            cell_emodel.threshold_current,
            cell_emodel.holding_current,
            cell_emodel.name,
            cell.morphology_fname,
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
            run_hoc_filename = "run.hoc"
            config = load_config(filename=self.configfile)
            cell_hoc, syn_hoc, simul_hoc, run_hoc = get_hoc(
                config=config, syn_temp_name="hoc_synapses"
            )

            hoc_paths = get_hoc_paths_args(config)
            write_hocs(
                hoc_paths,
                cell_hoc,
                simul_hoc,
                run_hoc,
                run_hoc_filename,
                syn_hoc,
            )


class RunHoc(MemodelParameters):
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

    def output(self):
        """Produces the hoc recordings."""
        output_list = []

        workflow_output_dir = workflow_config.get("paths", "output")
        script_path = get_output_path(
            self.mtype, self.etype, self.region, self.gidx, workflow_output_dir
        )
        output_path = os.path.join(script_path, "hoc_recordings")

        if self.configfile in ["config_synapses.ini", "config_synapses_short.ini"]:
            output_list.append(
                luigi.LocalTarget(os.path.join(output_path, "soma_voltage_vecstim.dat"))
            )

        else:
            for idx in range(3):
                output_list.append(
                    luigi.LocalTarget(
                        os.path.join(output_path, "soma_voltage_step%d.dat" % (idx + 1))
                    )
                )

        return output_list

    def run(self):
        """Executes the hoc script."""
        workflow_output_dir = workflow_config.get("paths", "output")
        hoc_path = get_output_path(
            self.mtype, self.etype, self.region, self.gidx, workflow_output_dir
        )
        with cwd(hoc_path):
            subprocess.call(["sh", "./run_hoc.sh"])


class RunPyScript(MemodelParameters):
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
        output_list = []

        if self.configfile in ["config_singlestep.ini", "config_singlestep_short.ini"]:
            output_list.append(
                luigi.LocalTarget(
                    os.path.join(self.output_path, "soma_voltage_step1.dat")
                )
            )
        elif self.configfile in ["config_synapses.ini", "config_synapses_short.ini"]:
            output_list.append(
                luigi.LocalTarget(
                    os.path.join(self.output_path, "soma_voltage_vecstim.dat")
                )
            )
        else:
            for idx in range(3):
                output_list.append(
                    luigi.LocalTarget(
                        os.path.join(
                            self.output_path, "soma_voltage_step%d.dat" % (idx + 1)
                        )
                    )
                )

        return output_list

    def run(self):
        """Executes the python script."""
        output_dir = workflow_config.get("paths", "output")
        memodel_dir = get_output_path(
            self.mtype, self.etype, self.region, self.gidx, output_dir
        )

        with cwd(memodel_dir):
            if self.configfile:
                subprocess.call(["sh", "./compile_mechanisms.sh", self.configfile])
                run_emodel(config_file=self.configfile)
            else:
                subprocess.call(["sh", "./compile_mechanisms.sh"])
                run_emodel(config_file=None)


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
