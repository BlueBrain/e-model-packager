"""Workflow to build e-model packages."""
import collections
import json
import os
import re
import shutil
import subprocess

import luigi
from bluepy.v2 import Cell as bpcell
import bglibpy

from e_model_packages.sscx2020.utils import (
    read_circuit,
    NpEncoder,
    get_mecombo_emodels,
    cwd,
    get_output_path,
)
from e_model_packages.sscx2020.config_decorator import ConfigDecorator


workflow_config = ConfigDecorator(luigi.configuration.get_config())
# pylint: disable=too-many-locals


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
        else:
            gids = list(
                circuit.cells.ids(
                    {bpcell.MTYPE: self.mtype, bpcell.ETYPE: self.etype},
                    limit=self.gids_per_metype,
                )
            )
            gid = gids[self.gidx - 1]
            self.mtype_etype_gids[self.mtype][self.etype] = [
                self.gidx - 1,
                gids[self.gidx - 1],
            ]

            tasks = [PrepareMEModelDirectory(self.mtype, self.etype, gid, self.gidx)]

        return tasks

    def output(self):
        """The JSON output."""
        output_dir = workflow_config.get("paths", "output")
        return luigi.LocalTarget(os.path.join(output_dir, "metype_gids.json"))

    def run(self):
        """Write the JSON."""
        with self.output().open("w") as out_file:
            json.dump(self.mtype_etype_gids, out_file, indent=4, cls=NpEncoder)


class PrepareOutputDirectory(luigi.Task):
    """Task to prepare the output directory.

    Copy scripts, config, templates and params files in the main output directory.
    """

    def output(self):
        """Copy files."""
        targets = []
        output_dir = workflow_config.get("paths", "output")
        files = workflow_config.get("files", "general_scripts")

        for f in files:
            targets.append(luigi.LocalTarget(os.path.join(output_dir, f)))
        targets.append(luigi.LocalTarget(os.path.join(output_dir, "config")))
        targets.append(luigi.LocalTarget(os.path.join(output_dir, "templates")))

        return targets

    @staticmethod
    def copy_templates(output_dir):
        """Copy mechanisms into output directory."""
        output_templates_dir = os.path.join(output_dir, "templates")
        shutil.copytree(
            workflow_config.get("paths", "templates_to_copy_dir"), output_templates_dir
        )

    @staticmethod
    def copy_config(output_dir):
        """Copy python recordings config into output directory."""
        output_config_dir = os.path.join(output_dir, "config")
        shutil.copytree(
            workflow_config.get("paths", "emodel_config_dir"), output_config_dir,
        )

    @staticmethod
    def copy_scripts(output_dir):
        """Copy scripts."""
        scripts_dir = workflow_config.get("paths", "scripts_dir")
        script_files = workflow_config.get("files", "general_scripts")

        for script_file in script_files:
            script_path = os.path.join(scripts_dir, script_file)
            shutil.copy(script_path, output_dir)

    def run(self):
        """Copy scripts, config and templates."""
        output_dir = workflow_config.get("paths", "output")

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        # teamplates to be copied
        self.copy_templates(output_dir)

        # python recordings config
        self.copy_config(output_dir)

        # scripts
        self.copy_scripts(output_dir)


class PrepareConfig(luigi.Task):
    """Task to prepare the e-model directory.

    Attributes:
        mtype: morphological type
        etype: electrophysiological type
        gid: id of cell in the circuit
        gidx: index of cell
    """

    mtype = luigi.Parameter()
    etype = luigi.Parameter()
    gidx = luigi.IntParameter()

    def requires(self):
        """Requires the output directory to have been created."""
        return PrepareOutputDirectory()

    def output(self):
        """Write config file."""
        output_dir = workflow_config.get("paths", "output")
        config_file = os.path.join(output_dir, "config", "config.ini")

        return luigi.LocalTarget(config_file)

    def run(self):
        """Write mtype, etype, gidx in config file."""
        config_dir = workflow_config.get("paths", "emodel_config_dir")
        config_path = os.path.join(config_dir, "config_example.ini")

        with open(self.output().path, "w") as out_file:
            with open(config_path, "r") as in_file:
                for line in in_file:
                    if "mtype" in line.split("="):
                        line = "mtype={}\n".format(self.mtype)
                    elif "etype" in line.split("="):
                        line = "etype={}\n".format(self.etype)
                    elif "gidx" in line.split("="):
                        line = "gidx={}\n".format(self.gidx)

                    if line[0] != "#":
                        out_file.write(line)


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

    def requires(self):
        """Requires the script to have been copied in the main output directory."""
        tasks = [
            PrepareOutputDirectory(),
            PrepareConfig(self.mtype, self.etype, self.gidx),
        ]
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
        os.makedirs(os.path.join(memodel_dir, "old_python_recordings"))

    def copy_morph_emodel(self, circuit, blueconfig, memodel_morph_dir):
        """Copy morphology and emodel."""
        circ_morph_dir = os.path.join(blueconfig.Run["MorphologyPath"], "ascii")

        cell = circuit.cells.get(self.gid)
        mecombo = cell.me_combo

        morph_fname = "%s.asc" % cell.morphology
        morph_path = os.path.join(circ_morph_dir, morph_fname)

        shutil.copy(morph_path, memodel_morph_dir)

        return mecombo, morph_fname

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

        for script_file in script_files:
            script_path = os.path.join(scripts_dir, script_file)
            shutil.copy(script_path, self.output().path)

    def write_down_using_templates(self, templates_dir, template_vars):
        """Fill in and write constants.hoc & current_amp.dat templates given templates & vars."""
        for template_fn, variables in template_vars.items():
            template_path = os.path.join(templates_dir, template_fn)
            template = open(template_path).read()
            content = template.format(**variables)

            output_path = os.path.join(self.output().path, template_fn)
            open(output_path, "w").write(content)

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

        n_of_synapses = len(cell_info_dict["synapses"].items())

        a_key = list(cell_info_dict["synapses"].keys())[0]
        n_of_cols = len(cell_info_dict["synapses"][a_key])

        synapse_tsv_content = "%d %d\n" % (n_of_synapses, n_of_cols)

        synconf_dict = collections.defaultdict(list)
        synconf_ordering = []

        for synapse_id, synapse_dict in cell_info_dict["synapses"].items():
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
            pre_mtype = circuit.cells.get(pre_gid).mtype
            post_sec_sectionlist_id, post_sec_sectionlist_index = self.convert_sec_name(
                synapse_dict["post_sec_name"]
            )

            synapse_tsv_content += "%s\n" % "\t".join(
                [
                    str(x)
                    for x in [
                        synapse_dict["synapse_id"],
                        pre_gid,
                        pre_mtype,
                        post_sec_sectionlist_id,
                        post_sec_sectionlist_index,
                        "%.3f" % synapse_dict["post_segx"],
                        synapse_dict["syn_type"],
                        "%.20e" % synapse_dict["synapse_parameters"]["Dep"],
                        "%.20e" % synapse_dict["synapse_parameters"]["Fac"],
                        "%.20e" % synapse_dict["synapse_parameters"]["Use"],
                        "%.20e" % tau_d,
                        delay,
                        weight,
                    ]
                ]
            )
            for command in synapse_dict["synapseconfigure_cmds"]:
                if command not in synconf_ordering:
                    synconf_ordering.append(command)
                synconf_dict[command].append(synapse_id)

        synapse_tsv_filename = os.path.join(synapse_dir, "synapses.tsv")
        with open(synapse_tsv_filename, "w") as synapse_tsv_file:
            synapse_tsv_file.write(synapse_tsv_content)

        synconf_filename = os.path.join(synapse_dir, "synconf.txt")
        with open(synconf_filename, "w") as synconf_file:
            synconf_file.write(
                self.generate_synconf_content(synconf_dict, synconf_ordering)
            )

    def fill_in_templates(
        self, mecombo_thresholds, mecombo_hypamps, mecombo, emodel, morph_fname,
    ):
        """Fill in and write constants.hoc & current_amp.dat templates."""
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

        self.write_down_using_templates(templates_dir, template_vars)

    def run(self):
        """Create me-model directories."""
        circuit_config_path = workflow_config.get("paths", "circuit")
        circuit, blueconfig = read_circuit(circuit_config_path)

        mecombo_emodels, mecombo_thresholds, mecombo_hypamps = get_mecombo_emodels(
            blueconfig
        )

        memodel_dir = self.output().path
        memodel_morph_dir = os.path.join(memodel_dir, "morphology")
        synapses_dir = os.path.join(memodel_dir, "synapses")

        # make dirs
        self.makedirs(memodel_morph_dir, synapses_dir)

        # morph & mecombo
        mecombo, morph_fname = self.copy_morph_emodel(
            circuit, blueconfig, memodel_morph_dir
        )

        # synapses
        self.write_synapses(circuit_config_path, synapses_dir)

        # copy mechanisms
        self.copy_mechanisms()

        # scripts
        self.copy_scripts()

        # templates to be filled
        emodel = mecombo_emodels[mecombo]
        self.fill_in_templates(
            mecombo_thresholds, mecombo_hypamps, mecombo, emodel, morph_fname,
        )


class CreateHoc(luigi.Task):
    """Task to create the hoc file of an emodel.

    Attributes:
        mtype: morphological type
        etype: electrophysiological type
        gidx: index of cell
    """

    mtype = luigi.Parameter()
    etype = luigi.Parameter()
    gidx = luigi.IntParameter()

    def requires(self):
        """Requires the output paths to be made."""
        return ParseCircuit(mtype=self.mtype, etype=self.etype, gidx=self.gidx)

    def get_output_path(self):
        """Returns the path to the outputs directory."""
        workflow_output_dir = workflow_config.get("paths", "output")
        return get_output_path(self.mtype, self.etype, self.gidx, workflow_output_dir)

    def output(self):
        """Produces the hoc file."""
        output_path = self.get_output_path()
        with open(os.path.join(output_path, "constants.hoc"), "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.split("=")
            if line[0] == "template_fname":
                tmp = line[1].rstrip()
                filename = tmp.strip('"')

        return luigi.LocalTarget(os.path.join(output_path, filename))

    def run(self):
        """Createss the hoc script."""
        workflow_output_dir = workflow_config.get("paths", "output")
        with cwd(workflow_output_dir):
            subprocess.call(["python", "create_hoc.py"])


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

    def requires(self):
        """Requires the hoc file to have been created."""
        return CreateHoc(mtype=self.mtype, etype=self.etype, gidx=self.gidx)

    def output(self):
        """Produces the hoc recordings."""
        output_list = []

        workflow_output_dir = workflow_config.get("paths", "output")
        script_path = get_output_path(
            self.mtype, self.etype, self.gidx, workflow_output_dir
        )
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
        workflow_output_dir = workflow_config.get("paths", "output")
        hoc_path = get_output_path(
            self.mtype, self.etype, self.gidx, workflow_output_dir
        )
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

    def requires(self):
        """Requires the output paths to be made."""
        return ParseCircuit(mtype=self.mtype, etype=self.etype, gidx=self.gidx)

    def output(self):
        """Produces the python recordings."""
        output_list = []

        workflow_output_dir = workflow_config.get("paths", "output")
        script_path = get_output_path(
            self.mtype, self.etype, self.gidx, workflow_output_dir
        )
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
        workflow_output_dir = workflow_config.get("paths", "output")
        with cwd(workflow_output_dir):
            subprocess.call(["sh", "./run_py.sh"])


class RunOldPyScript(luigi.Task):
    """Task to run the python script for an emodel.

    Attributes:
        mtype: morphological type
        etype: electrophysiological type
        gidx: index of cell
    """

    mtype = luigi.Parameter()
    etype = luigi.Parameter()
    gidx = luigi.IntParameter()

    def requires(self):
        """Requires the hoc file to have been created."""
        return CreateHoc(mtype=self.mtype, etype=self.etype, gidx=self.gidx)

    def output(self):
        """Produces the python recordings."""
        output_list = []

        workflow_output_dir = workflow_config.get("paths", "output")
        script_path = get_output_path(
            self.mtype, self.etype, self.gidx, workflow_output_dir
        )
        output_path = os.path.join(script_path, "old_python_recordings")

        for idx in range(3):
            output_list.append(
                luigi.LocalTarget(
                    os.path.join(output_path, "soma_voltage_step%d.dat" % (idx + 1))
                )
            )

        return output_list

    def run(self):
        """Executes the python script."""
        workflow_output_dir = workflow_config.get("paths", "output")
        with cwd(workflow_output_dir):
            subprocess.call(["sh", "./run_old_py.sh"])


class DoRecordings(luigi.WrapperTask):
    """Launch both RunHoc and RunPyScript.

    Attributes:
        mtype: morphological type
        etype: electrophysiological type
        gidx: index of cell
    """

    mtype = luigi.Parameter()
    etype = luigi.Parameter()
    gidx = luigi.IntParameter()

    def requires(self):
        """Launch both RunHoc and RunPyScript."""
        tasks = [
            RunHoc(self.mtype, self.etype, self.gidx),
            RunPyScript(self.mtype, self.etype, self.gidx),
            RunOldPyScript(self.mtype, self.etype, self.gidx),
        ]
        return tasks


class SSCX2020(luigi.WrapperTask):
    """The skeleton task to perform the workflow."""

    def requires(self):
        """The ParseCircuit method is required."""
        return ParseCircuit()
