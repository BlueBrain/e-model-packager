"""Create me-model packages for nmc portal."""

from __future__ import print_function

import os
import shutil
import json
import collections
import numpy as np

import bluepy
from bluepy.v2 import Cell as bpcell


from bluepy_configfile.configfile import BlueConfig


class Settings(object):
    """Sets the circuit config."""

    def __init__(self, circuit_config):
        """Constructor."""
        self.circuit_config = circuit_config


class NpEncoder(json.JSONEncoder):
    """Class to encode np.integer as python int."""

    def default(self, obj):
        """Convert numpy integer to int."""
        if isinstance(obj, np.integer):
            return int(obj)
        else:
            return super(NpEncoder, self).default(obj)


def read_circuit(settings):
    """Read circuit info."""
    circuit = bluepy.Circuit(settings.circuit_config).v2
    blueconfig = BlueConfig(open(settings.circuit_config))

    return circuit, blueconfig


def select_gids(circuit, settings, gids_per_metype=5):
    """Select gids for each me-type."""
    metype_gids = {}

    metypes_df = circuit.cells.get(
        properties=[bpcell.MTYPE, bpcell.ETYPE, bpcell.LAYER]
    ).drop_duplicates()
    metypes = [(row["mtype"], row["etype"]) for _, row in metypes_df.iterrows()]
    print("Found %d me-types" % len(metypes))

    for mtype, etype in metypes:
        metype_gids[(mtype, etype)] = list(
            circuit.cells.ids(
                {bpcell.MTYPE: mtype, bpcell.ETYPE: etype}, limit=gids_per_metype
            )
        )
        print(
            "Found %d %s gids for: %s_%s"
            % (
                len(metype_gids[(mtype, etype)]),
                metype_gids[(mtype, etype)],
                mtype,
                etype,
            )
        )

    metype_gids_json_fn = os.path.join(settings["output_dir"], "metype_gids.json")

    mtype_etype_gids = collections.defaultdict(dict)
    for (mtype, etype), gids in metype_gids.items():
        mtype_etype_gids[mtype][etype] = gids

    json.dump(mtype_etype_gids, open(metype_gids_json_fn, "w"), indent=4, cls=NpEncoder)

    return metype_gids


def create_memodel_dirs(config, circuit, blueconfig, metype_gids):
    """Create me-model directories."""
    output_dir = config["output_dir"]
    memodels_dir = os.path.join(output_dir, "memodel_dirs")
    circ_morph_dir = os.path.join(blueconfig.Run["MorphologyPath"], "ascii")
    circ_emodel_dir = blueconfig.Run["METypePath"]

    scripts_dir = config["scripts_dir"]
    script_fns = config["scripts"]
    script_paths = []

    templates_dir = config["templates_dir"]

    for script_fn in script_fns:
        script_path = os.path.join(scripts_dir, script_fn)
        script_paths.append(script_path)

    mecombo_emodels, mecombo_thresholds, mecombo_hypamps = get_mecombo_emodels(
        blueconfig
    )

    for ((mtype, etype), gids) in metype_gids.items():
        mtype_dir = os.path.join(memodels_dir, mtype)
        metype_dir = os.path.join(mtype_dir, etype)

        metype = "%s_%s" % (mtype, etype)
        for index, gid in enumerate(gids):
            memodel_name = "%s_%d" % (metype, index)
            memodel_dir = os.path.join(metype_dir, memodel_name)
            os.makedirs(memodel_dir)
            memodel_morph_dir = os.path.join(memodel_dir, "morphology")
            os.makedirs(memodel_morph_dir)

            hocrec_dir = os.path.join(memodel_dir, "hoc_recordings")
            pyrec_dir = os.path.join(memodel_dir, "python_recordings")

            os.makedirs(hocrec_dir)
            os.makedirs(pyrec_dir)

            cell = circuit.cells.get(gid)
            morph = cell.morphology
            mecombo = cell.me_combo

            threshold = mecombo_thresholds[mecombo]
            holding = mecombo_hypamps[mecombo]

            morph_fname = "%s.asc" % morph
            morph_path = os.path.join(circ_morph_dir, morph_fname)

            emodel = mecombo_emodels[mecombo]
            emodel_fname = "%s.hoc" % emodel
            emodel_path = os.path.join(circ_emodel_dir, emodel_fname)

            memodel_mechanisms_dir = os.path.join(memodel_dir, "mechanisms")
            shutil.copy(morph_path, memodel_morph_dir)
            shutil.copy(emodel_path, memodel_dir)
            shutil.copytree(config["mechanisms"], memodel_mechanisms_dir)

            for script_path in script_paths:
                shutil.copy(script_path, memodel_dir)

            template_vars = {}

            template_vars["constants.hoc"] = {
                "template_name": emodel,
                "gid": gid,
                "morph_dir": "morphology",
                "morph_fname": morph_fname,
            }
            template_vars["current_amps.dat"] = {
                "holding": holding,
                "amp1": 1.50 * threshold,
                "amp2": 2.00 * threshold,
                "amp3": 2.50 * threshold,
            }

            for template_fn, vars in template_vars.items():
                template_path = os.path.join(templates_dir, template_fn)
                template = open(template_path).read()
                content = template.format(**vars)

                output_path = os.path.join(memodel_dir, template_fn)
                open(output_path, "w").write(content)

            print("Created dir for %s" % memodel_name)


def get_mecombo_emodels(blueconfig):
    """Create a dict matching me_combo names to template_names."""
    mecombo_filename = blueconfig.Run["MEComboInfoFile"]

    with open(mecombo_filename) as mecombo_file:
        mecombo_content = mecombo_file.read()

    mecombo_emodels = {}
    mecombo_thresholds = {}
    mecombo_hypamps = {}

    for line in mecombo_content.split("\n")[1:-1]:
        mecombo_info = line.split("\t")
        emodel = mecombo_info[4]
        me_combo = mecombo_info[5]
        threshold = float(mecombo_info[6])
        hypamp = float(mecombo_info[7])
        mecombo_emodels[me_combo] = emodel
        mecombo_thresholds[me_combo] = threshold
        mecombo_hypamps[me_combo] = hypamp

    return mecombo_emodels, mecombo_thresholds, mecombo_hypamps


def run(config):
    """Run with config."""
    output_dir = config["output_dir"]
    os.makedirs(output_dir)

    settings = Settings(circuit_config=config["circuit"])

    circuit, blueconfig = read_circuit(settings)

    metype_gids = select_gids(circuit, config)

    create_memodel_dirs(config, circuit, blueconfig, metype_gids)


def main():
    """Main."""
    with open("nmc_emodels_cfg.json") as json_file:
        config = json.load(json_file)

    run(config["config"])


if __name__ == "__main__":
    main()
