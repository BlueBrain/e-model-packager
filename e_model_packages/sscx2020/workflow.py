"""Workflow to build e-model packages."""
# pylint: disable=wrong-import-position
# pylint: disable=wrong-import-order
# pylint: disable=import-error

import json
import os

if "PMI_RANK" in os.environ:
    del os.environ["PMI_RANK"]
import shutil
import subprocess
import sys
from pathlib import Path
import logging
import pickle
from tqdm import tqdm

import luigi

from e_model_packages.sscx2020.utils import (
    NpEncoder,
    cwd,
    get_output_path,
    create_single_step_config,
)
from e_model_packages.sscx2020.config_decorator import ConfigDecorator
from e_model_packages.circuit import BluepyCircuit, BluepySimulation, SynapseExtractor

from emodelrunner.load import load_config
from emodelrunner.create_hoc import get_hoc, write_hocs
from emodelrunner.write_factsheets import (
    write_metype_json,
    write_etype_json,
    write_morph_json,
)

from luigi_tools.task import RemoveCorruptedOutputMixin

sys.path.append(os.path.join("e_model_packages", "sscx2020", "extra_data", "scripts"))
from old_run import main as old_python_main


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


class ExtractCircuitInfo(SmartTask):
    """Extracts the metype, region and gids from the circuit.

    Args:
        ngids (int): Number of gids to retrieve
    """

    ngids = luigi.IntParameter()

    def output(self):
        """The JSON output."""
        output_dir = Path(workflow_config.get("paths", "output"))
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        return {
            "json": luigi.LocalTarget(output_dir / "metype_region_gids.json"),
            "pickle": luigi.LocalTarget(output_dir / "metype_region_gids.pickle"),
        }

    def run(self):
        """Write the JSON."""
        metype_region_gids_dict = {}
        circuit_config_path = workflow_config.get("paths", "circuit")
        circuit = BluepyCircuit(circuit_config_path)

        regions = workflow_config.get("circuit", "regions")

        metype_region_gids = circuit.extract_circuit_metype_region_gids(
            self.ngids, regions
        )
        with open(self.output()["pickle"].path, "wb") as pickle_handle:
            pickle.dump(metype_region_gids, pickle_handle)

        for (mtype, etype, region), gids in metype_region_gids.items():
            metype_region_gids_dict[mtype] = {etype: {region: gids}}

        with self.output()["json"].open("w") as out_file:
            json.dump(metype_region_gids_dict, out_file, indent=4, cls=NpEncoder)


class CollectMEModels(SmartTask):
    """Yield the model preparation tasks."""

    ngids = luigi.IntParameter(default=5)
    task_complete = False

    def requires(self):
        """Metype, region and gids info should be extracted."""
        return ExtractCircuitInfo(ngids=self.ngids)

    def run(self):
        """Spawn the memodel jobs."""
        output_dir = Path(workflow_config.get("paths", "output"))
        with open(output_dir / "metype_region_gids.pickle", "rb") as pickle_file:
            metype_region_gids = pickle.load(pickle_file)

        tasks = []
        logging.info("Generating the model preparation tasks...")
        for (mtype, etype, region), gids in tqdm(metype_region_gids.items()):
            for gidx, gid in enumerate(gids):
                gidx = gidx + 1  # 1 indexed for users
                tasks.append(
                    CreateHoc(
                        mtype=mtype, etype=etype, region=region, gid=gid, gidx=gidx
                    )
                )
                tasks.append(
                    CreateMETypeJson(
                        mtype=mtype, etype=etype, region=region, gid=gid, gidx=gidx
                    )
                )

        self.task_complete = True
        yield tasks

    def complete(self):
        """Override the complete method."""
        return self.task_complete


class CreateMETypeJson(MemodelParameters):
    """Task to create a me-type factsheet json file."""

    configfile = None

    def requires(self):
        """Requires the script to have been copied in the main output directory."""
        tasks = RunPyScript(
            mtype=self.mtype,
            etype=self.etype,
            region=self.region,
            gid=self.gid,
            gidx=self.gidx,
            configfile=self.configfile,
            run_single_step=True,
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
            "e_type_factsheeet.json",
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
            write_metype_json(config, factsheets_dir)
            write_etype_json(config, factsheets_dir)
            write_morph_json(config, factsheets_dir)

        # remove extra output
        os.remove(
            os.path.join(memodel_dir, "python_recordings", "soma_voltage_step1.dat")
        )
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
        """Produces a .completed.txt file after all steps are done."""
        completed_file = os.path.join(self.output_folder, ".completed.txt")
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

    def fill_in_templates(
        self,
        threshold,
        holding,
        emodel,
        morph_fname,
    ):
        """Fill in and write constants.json & current_amp.json templates.

        Args:
        threshold(float): threshold current.
        holding(float): holding current.
        emodel(str): emodel name.
        morph_fname(str): morphology filename.
        """
        output_dir = "config"

        # current amps
        current_amps = {
            "holding": holding,
            "amps": [1.50 * threshold, 2.00 * threshold, 2.50 * threshold],
        }

        currents_amp_path = os.path.join(
            self.output_folder, output_dir, "current_amps.json"
        )
        with open(currents_amp_path, "w") as out_file:
            json.dump(current_amps, out_file, indent=4, cls=NpEncoder)

        # constants
        constants = {
            "celsius": 34,
            "v_init": -80,
            "tstop": 3000,
            "dt": 0.025,
            "template_name": emodel,
            "gid": self.gid,
            "morph_dir": "morphology",
            "morph_fname": morph_fname,
        }

        constants_path = os.path.join(self.output_folder, output_dir, "constants.json")
        with open(constants_path, "w") as out_file:
            json.dump(constants, out_file, indent=4, cls=NpEncoder)

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

        # templates to be filled
        self.fill_in_templates(
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

            write_hocs(
                config,
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

    configfile = luigi.Parameter(default=None)
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

        if self.configfile is None:
            for idx in range(3):
                output_list.append(
                    luigi.LocalTarget(
                        os.path.join(output_path, "soma_voltage_step%d.dat" % (idx + 1))
                    )
                )

        elif self.configfile == "config_synapses.ini":
            output_list.append(
                luigi.LocalTarget(os.path.join(output_path, "soma_voltage_vecstim.dat"))
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
        run_single_step: set to True to only run one single step protocol
    """

    configfile = luigi.Parameter(default=None)
    run_single_step = luigi.BoolParameter(default=False)

    def requires(self):
        """Requires the output paths to be made."""
        return PrepareMEModelDirectory(
            mtype=self.mtype,
            etype=self.etype,
            region=self.region,
            gid=self.gid,
            gidx=self.gidx,
        )

    def output(self):
        """Produces the python recordings."""
        output_list = []

        workflow_output_dir = workflow_config.get("paths", "output")
        script_path = get_output_path(
            self.mtype, self.etype, self.region, self.gidx, workflow_output_dir
        )
        output_path = os.path.join(script_path, "python_recordings")

        if self.configfile is None:
            if self.run_single_step:
                output_list.append(
                    luigi.LocalTarget(
                        os.path.join(output_path, "soma_voltage_step1.dat")
                    )
                )
            else:
                for idx in range(3):
                    output_list.append(
                        luigi.LocalTarget(
                            os.path.join(
                                output_path, "soma_voltage_step%d.dat" % (idx + 1)
                            )
                        )
                    )

        elif self.configfile == "config_synapses.ini":
            output_list.append(
                luigi.LocalTarget(os.path.join(output_path, "soma_voltage_vecstim.dat"))
            )

        return output_list

    def run(self):
        """Executes the python script."""
        output_dir = workflow_config.get("paths", "output")
        memodel_dir = get_output_path(
            self.mtype, self.etype, self.region, self.gidx, output_dir
        )
        if self.run_single_step:
            new_config = "config_single_step.ini"
            config_dir = os.path.join(memodel_dir, "config")
            create_single_step_config(self.configfile, new_config, config_dir)

            with cwd(memodel_dir):
                subprocess.call(["sh", "./run_py.sh", new_config])

            os.remove(os.path.join(config_dir, new_config))
        else:
            with cwd(memodel_dir):
                if self.configfile:
                    subprocess.call(["sh", "./run_py.sh", self.configfile])
                else:
                    subprocess.call(["sh", "./run_py.sh"])


class RunOldPyScript(MemodelParameters):
    """Task to run the python script for an emodel.

    Attributes:
        configfile : name of config file in /config to use when running script
    """

    configfile = luigi.Parameter(default=None)

    def requires(self):
        """Requires the hoc file to have been created."""
        return CreateHoc(
            mtype=self.mtype,
            etype=self.etype,
            region=self.region,
            gid=self.gid,
            gidx=self.gidx,
            configfile=self.configfile,
        )

    def output(self):
        """Produces the python recordings."""
        output_list = []

        workflow_output_dir = workflow_config.get("paths", "output")
        script_path = get_output_path(
            self.mtype, self.etype, self.region, self.gidx, workflow_output_dir
        )
        output_path = os.path.join(script_path, "old_python_recordings")

        if self.configfile is None:
            for idx in range(3):
                output_list.append(
                    luigi.LocalTarget(
                        os.path.join(output_path, "soma_voltage_step%d.dat" % (idx + 1))
                    )
                )

        elif self.configfile == "config_synapses.ini":
            output_list.append(
                luigi.LocalTarget(os.path.join(output_path, "soma_voltage_vecstim.dat"))
            )

        return output_list

    def run(self):
        """Executes the python script."""
        output_dir = workflow_config.get("paths", "output")
        memodel_dir = get_output_path(
            self.mtype, self.etype, self.region, self.gidx, output_dir
        )
        with cwd(memodel_dir):
            # compile mechanisms if needed
            if os.path.exists(os.path.join("x86_64", "special")):
                subprocess.call(["nrnivmodl", "mechanisms"])
            # run old_python
            old_python_main(self.configfile)


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
            RunOldPyScript(
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
