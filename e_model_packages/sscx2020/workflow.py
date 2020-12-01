"""Workflow to build e-model packages."""
# pylint: disable=wrong-import-position
# pylint: disable=wrong-import-order
# pylint: disable=import-error
import collections
import json
import os
import re
import shutil
import subprocess
import sys

import luigi
from bluepy.v2 import Cell as bpcell
import bglibpy

from e_model_packages.sscx2020.utils import (
    read_circuit,
    NpEncoder,
    get_mecombo_emodel,
    cwd,
    get_output_path,
    create_single_step_config,
)
from e_model_packages.sscx2020.config_decorator import ConfigDecorator

# for loading create_hoc, old_run & factsheets modules
# that are supposed to be loaded from the memodel directory.
sys.path.append(os.path.join("e_model_packages", "sscx2020", "extra_data", "scripts"))
from load import load_config
from create_hoc import get_hoc, write_hocs
from old_run import main as old_python_main
from write_factsheets import write_metype_json, write_etype_json, write_morph_json


workflow_config = ConfigDecorator(luigi.configuration.get_config())
# pylint: disable=too-many-locals


class MemodelParameters(luigi.Task):
    """Parameter class to contain common MeModel parameters across various tasks.

    Luigi design pattern to address the parameter explosion problem.
    Reference https://luigi.readthedocs.io/en/stable/api/luigi.util.html

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


class ParseCircuit(luigi.Task):
    """Parse the circuit to get the number of mtypes etypes and cells."""

    gids_per_metype = luigi.IntParameter(default=5)
    mtype_etype_gids = collections.defaultdict(dict)

    mtype = luigi.Parameter(default=None)
    etype = luigi.Parameter(default=None)
    gidx = luigi.IntParameter(default=None)

    def requires(self):
        """The BuildCircuit task is a dependency of this task."""
        circuit_config_path = workflow_config.get("paths", "circuit")
        circuit, _ = read_circuit(circuit_config_path)

        # if mtype, etype, gidx not set, run required task for all metypes
        if None in [self.mtype, self.etype, self.gidx]:
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
                    tasks.append(CreateHoc(mtype, etype, gid, gidx))
                    tasks.append(CreateMETypeJson(mtype, etype, gid, gidx))
        else:
            gids = list(
                circuit.cells.ids({bpcell.MTYPE: self.mtype, bpcell.ETYPE: self.etype})
            )
            gid = gids[self.gidx - 1]
            self.mtype_etype_gids[self.mtype][self.etype] = [
                gids[self.gidx - 1],
            ]

            tasks = [
                PrepareMEModelDirectory(self.mtype, self.etype, gid, self.gidx),
                CreateHoc(self.mtype, self.etype, gid, self.gidx),
                CreateMETypeJson(self.mtype, self.etype, gid, self.gidx),
            ]

        return tasks

    def output(self):
        """The JSON output."""
        output_dir = workflow_config.get("paths", "output")
        return luigi.LocalTarget(os.path.join(output_dir, "metype_gids.json"))

    def run(self):
        """Write the JSON."""
        with self.output().open("w") as out_file:
            json.dump(self.mtype_etype_gids, out_file, indent=4, cls=NpEncoder)


class CreateMETypeJson(MemodelParameters):
    """Task to create a me-type factsheet json file."""

    configfile = None

    def requires(self):
        """Requires the script to have been copied in the main output directory."""
        tasks = [
            RunPyScript(
                self.mtype,
                self.etype,
                self.gid,
                self.gidx,
                self.configfile,
                run_single_step=True,
            )
        ]
        return tasks

    def output(self):
        """The JSON output."""
        output_dir = workflow_config.get("paths", "output")
        memodel_dir = get_output_path(self.mtype, self.etype, self.gidx, output_dir)
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
        memodel_dir = get_output_path(self.mtype, self.etype, self.gidx, output_dir)

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


class PrepareOutputDirectory(luigi.Task):
    """Task to prepare the output directory.

    Create the main output directory.
    """

    def output(self):
        """Copy files."""
        output_dir = workflow_config.get("paths", "output")

        return luigi.LocalTarget(output_dir)

    def run(self):
        """Create the output folder."""
        output_dir = self.output().path

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)


class PrepareMEModelDirectory(MemodelParameters):
    """Task to prepare the e-model directory."""

    def requires(self):
        """Requires the script to have been copied in the main output directory."""
        tasks = [PrepareOutputDirectory()]
        return tasks

    def output(self):
        """Does not produce output."""
        output_dir = workflow_config.get("paths", "output")
        memodel_dir = get_output_path(self.mtype, self.etype, self.gidx, output_dir)

        return luigi.LocalTarget(memodel_dir)

    def makedirs(self, memodel_morph_dir, synapses_dir):
        """Make directories."""
        memodel_dir = self.output().path
        os.makedirs(memodel_dir)
        os.makedirs(synapses_dir)

        os.makedirs(memodel_morph_dir)
        os.makedirs(os.path.join(memodel_dir, "hoc_recordings"))
        os.makedirs(os.path.join(memodel_dir, "python_recordings"))

    def get_cell_data_from_circuit(self, circuit):
        """Get cell data from bluepy circuit."""
        cell = circuit.cells.get(self.gid)
        return cell.me_combo, cell.morphology, "L{}".format(cell.layer)

    def write_cell_info(self, morphology, layer, output_dir):
        """Create cell_info.json file and write it."""
        cell_info_path = os.path.join(output_dir, "cell_info.json")

        me_type = "_".join((self.mtype, self.etype))
        name = "_".join((me_type, str(self.gidx)))

        cell_info = {
            "cell name": name,
            "e-type": self.etype,
            "gid": self.gid,
            "layer": layer,
            "m-type": self.mtype,
            "me-type": me_type,
            "morphology": morphology,
        }

        with open(cell_info_path, "w") as out_file:
            json.dump(cell_info, out_file, indent=4, cls=NpEncoder)

    @staticmethod
    def copy_morph(morph_fname, blueconfig, memodel_morph_dir):
        """Copy morphology."""
        circ_morph_dir = os.path.join(blueconfig.Run["MorphologyPath"], "ascii")
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

    @staticmethod
    def copy_GUI_utils(output_dir):
        """Copy mechanisms into output directory."""
        output_GUI_dir = os.path.join(output_dir, "GUI_utils")
        shutil.copytree(workflow_config.get("paths", "GUI_dir"), output_GUI_dir)

    def copy_mechanisms(self):
        """Copy mechanisms into output directory."""
        memodel_mechanisms_dir = os.path.join(self.output().path, "mechanisms")
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
                shutil.copy(script_path, self.output().path)
        # or only one file -> str
        else:
            script_path = os.path.join(scripts_dir, script_files)
            shutil.copy(script_path, self.output().path)

    @staticmethod
    def convert_sec_name(sec_name):
        """Convert section name into sectionlist_id and section_list_index."""
        match = re.match(r"(.*)\[(.*)\]", sec_name)
        if match is None:
            raise Exception("Couldnt match section name %s" % sec_name)

        sectionlist_name = match.groups()[0]
        sectionlist_index = int(match.groups()[1])

        sectionlist_names = ["soma", "dend", "apic", "axon"]

        return sectionlist_names.index(sectionlist_name), sectionlist_index

    @staticmethod
    def generate_synconf_content(synconf_dict, synconf_ordering):
        """Generate content for synconf.txt."""
        synconf_content = ""
        for command in synconf_ordering:
            gids = synconf_dict[command]
            synconf_content += "%s\n%s\n" % (
                command,
                " ".join([str(x) for x in gids] + [str(-1e15)]),
            )

        return synconf_content

    def write_synapses(self, blueconfig, synapse_dir):
        """Save the synapses from the circuit to a tsv."""
        ssim = bglibpy.SSim(blueconfig, record_dt=0.1)
        circuit = ssim.bc_simulation.circuit
        ssim.instantiate_gids([self.gid], synapse_detail=2, add_replay=True)

        cell_info_dict = ssim.cells[self.gid].info_dict
        cell = ssim.cells[self.gid]

        n_of_synapses = len(cell_info_dict["synapses"].items())

        # n_of_cols is actually not related to nmb of keys
        n_of_cols = 14

        synapse_tsv_content = "%d %d\n" % (n_of_synapses, n_of_cols)

        synconf_dict = collections.defaultdict(list)
        synconf_ordering = []
        mtype_map = []

        for (synapse_id, synapse_dict), (_, synapse) in zip(
            cell_info_dict["synapses"].items(), cell.synapses.items()
        ):
            if synapse_dict["syn_type"] > 100:
                # 119 or synapse_dict['syn_type'] == 113:
                tau_d = synapse_dict["synapse_parameters"]["tau_d_AMPA"]
            elif synapse_dict["syn_type"] < 100:
                # or synapse_dict['syn_type'] == 9:
                tau_d = synapse_dict["synapse_parameters"]["tau_d_GABAA"]
            else:
                raise Exception("Unknown synapse type %d" % synapse_dict["syn_type"])

            delay = cell_info_dict["connections"][synapse_id]["post_netcon"]["delay"]
            weight = cell_info_dict["connections"][synapse_id]["post_netcon"]["weight"]

            pre_gid = synapse_dict["pre_cell_id"]
            post_sec_sectionlist_id, post_sec_sectionlist_index = self.convert_sec_name(
                synapse_dict["post_sec_name"]
            )

            # assign pre-cell mtype to an id
            pre_mtype = circuit.cells.get(pre_gid).mtype
            if pre_mtype in mtype_map:
                # can use index. one occurence of pre_mtype & list is not long
                pre_mtype_id = mtype_map.index(pre_mtype)
            else:
                pre_mtype_id = len(mtype_map)
                mtype_map.append(pre_mtype)

            # get synapse id without the ('', ) part.
            _, sid = synapse_id

            # do not save in scientific notation : hoc files can't read it.
            synapse_tsv_content += "%s\n" % "\t".join(
                [
                    str(x)
                    for x in [
                        sid,
                        pre_gid,
                        post_sec_sectionlist_id,
                        post_sec_sectionlist_index,
                        "%.3f" % synapse_dict["post_segx"],
                        synapse_dict["syn_type"],
                        synapse_dict["synapse_parameters"]["Dep"],
                        synapse_dict["synapse_parameters"]["Fac"],
                        synapse_dict["synapse_parameters"]["Use"],
                        tau_d,
                        delay,
                        weight,
                        synapse.hsynapse.Nrrp,
                        pre_mtype_id,
                    ]
                ]
            )
            for command in synapse_dict["synapseconfigure_cmds"]:
                if command not in synconf_ordering:
                    synconf_ordering.append(command)
                synconf_dict[command].append(sid)

        synapse_tsv_filename = os.path.join(synapse_dir, "synapses.tsv")
        with open(synapse_tsv_filename, "w") as synapse_tsv_file:
            synapse_tsv_file.write(synapse_tsv_content)

        synconf_filename = os.path.join(synapse_dir, "synconf.txt")
        with open(synconf_filename, "w") as synconf_file:
            synconf_file.write(
                self.generate_synconf_content(synconf_dict, synconf_ordering)
            )

        mtype_map_content = ""
        for idx, pre_mtype in enumerate(mtype_map):
            mtype_map_content += f"{idx} {pre_mtype}\n"

        mtype_filename = os.path.join(synapse_dir, "mtype_map.tsv")
        with open(mtype_filename, "w") as mtype_file:
            mtype_file.write(mtype_map_content)

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
            self.output().path, output_dir, "current_amps.json"
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

        constants_path = os.path.join(self.output().path, output_dir, "constants.json")
        with open(constants_path, "w") as out_file:
            json.dump(constants, out_file, indent=4, cls=NpEncoder)

    def run(self):
        """Create me-model directories."""
        circuit_config_path = workflow_config.get("paths", "circuit")
        circuit, blueconfig = read_circuit(circuit_config_path)

        memodel_dir = self.output().path
        memodel_morph_dir = os.path.join(memodel_dir, "morphology")
        synapses_dir = os.path.join(memodel_dir, "synapses")

        # make dirs
        self.makedirs(memodel_morph_dir, synapses_dir)

        # get cell data
        mecombo, morphology, layer = self.get_cell_data_from_circuit(circuit)

        # create cell_info.json
        self.write_cell_info(morphology, layer, memodel_dir)

        # copy morphology
        morph_fname = "{}.asc".format(morphology)
        self.copy_morph(morph_fname, blueconfig, memodel_morph_dir)

        # synapses
        self.write_synapses(circuit_config_path, synapses_dir)

        # copy mechanisms
        self.copy_mechanisms()

        # scripts
        self.copy_scripts()

        # config
        self.copy_config(memodel_dir)

        # templates to be copied
        self.copy_templates(memodel_dir)

        # GUI scripts
        self.copy_GUI_utils(memodel_dir)

        emodel, threshold_current, holding_current = get_mecombo_emodel(
            blueconfig, mecombo
        )

        # templates to be filled
        self.fill_in_templates(
            threshold_current,
            holding_current,
            emodel,
            morph_fname,
        )


class CreateHoc(MemodelParameters):
    """Task to create the hoc file of an emodel.

    Attributes:
        configfile : name of config file in /config to use when creating hoc
    """

    configfile = luigi.Parameter(default=None)

    def requires(self):
        """Requires the output paths to be made."""
        return [
            PrepareMEModelDirectory(
                mtype=self.mtype, etype=self.etype, gid=self.gid, gidx=self.gidx
            )
        ]

    def get_output_path(self):
        """Returns the path to the outputs directory."""
        workflow_output_dir = workflow_config.get("paths", "output")
        return get_output_path(self.mtype, self.etype, self.gidx, workflow_output_dir)

    def output(self):
        """Produces the hoc file."""
        output_path = self.get_output_path()

        circuit_config_path = workflow_config.get("paths", "circuit")
        circuit_, blueconfig_ = read_circuit(circuit_config_path)
        cell_ = circuit_.cells.get(self.gid)
        mecombo_ = cell_.me_combo
        emodel, _, _ = get_mecombo_emodel(blueconfig_, mecombo_)

        filename = emodel + ".hoc"
        filenames = [filename, "run.hoc", "createsimulation.hoc"]

        targets = []
        for fname in filenames:
            targets.append(luigi.LocalTarget(os.path.join(output_path, fname)))

        return targets

    def run(self):
        """Creates the hoc script."""
        workflow_output_dir = self.get_output_path()
        with cwd(workflow_output_dir):
            run_hoc_filename = "run.hoc"
            config = load_config(filename=self.configfile)
            cell_hoc, emodel, syn_hoc, simul_hoc, run_hoc = get_hoc(
                config=config, syn_temp_name="hoc_synapses"
            )

            write_hocs(
                config,
                cell_hoc,
                emodel,
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
            gid=self.gid,
            gidx=self.gidx,
            configfile=self.configfile,
        )

    def output(self):
        """Produces the hoc recordings."""
        output_list = []

        workflow_output_dir = workflow_config.get("paths", "output")
        script_path = get_output_path(
            self.mtype, self.etype, self.gidx, workflow_output_dir
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
            self.mtype, self.etype, self.gidx, workflow_output_dir
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
        return [
            PrepareMEModelDirectory(
                mtype=self.mtype, etype=self.etype, gid=self.gid, gidx=self.gidx
            )
        ]

    def output(self):
        """Produces the python recordings."""
        output_list = []

        workflow_output_dir = workflow_config.get("paths", "output")
        script_path = get_output_path(
            self.mtype, self.etype, self.gidx, workflow_output_dir
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
        memodel_dir = get_output_path(self.mtype, self.etype, self.gidx, output_dir)
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
            gid=self.gid,
            gidx=self.gidx,
            configfile=self.configfile,
        )

    def output(self):
        """Produces the python recordings."""
        output_list = []

        workflow_output_dir = workflow_config.get("paths", "output")
        script_path = get_output_path(
            self.mtype, self.etype, self.gidx, workflow_output_dir
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
        memodel_dir = get_output_path(self.mtype, self.etype, self.gidx, output_dir)
        with cwd(memodel_dir):
            # compile mechanisms if needed
            if os.path.exists(os.path.join("x86_64", "special")):
                subprocess.call(["nrnivmodl", "mechanisms"])
            # run old_python
            old_python_main(self.configfile)


class CreateSystemLog(luigi.Task):
    """Task to log the modules and python packages used in the execution."""

    def requires(self):
        """Requires the main output directory to be present."""
        return PrepareOutputDirectory()

    def output(self):
        """A log file to be written."""
        workflow_output_dir = workflow_config.get("paths", "output")
        return luigi.LocalTarget(os.path.join(workflow_output_dir, "system-state.log"))

    def run(self):
        """Writes down the loaded modules, pip packages and python version."""
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
            RunHoc(self.mtype, self.etype, self.gid, self.gidx, self.configfile),
            RunPyScript(self.mtype, self.etype, self.gid, self.gidx, self.configfile),
            RunOldPyScript(
                self.mtype, self.etype, self.gid, self.gidx, self.configfile
            ),
        ]
        return tasks


class SSCX2020(luigi.WrapperTask):
    """The skeleton task to perform the workflow."""

    def requires(self):
        """The ParseCircuit method is required."""
        return [CreateSystemLog(), ParseCircuit()]
