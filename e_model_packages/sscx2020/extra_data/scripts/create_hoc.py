"""Creates .hoc from cell."""
import os

import bluepyopt.ephys as ephys

from load import (
    load_constants,
    load_params,
    find_param_file,
    load_mechanisms,
    define_parameters,
    get_axon_hoc,
)
from mymorphology import NrnFileMorphologyCustom


def write_hoc(hoc, template_name, hoc_dir=""):
    """Write hoc file."""
    # write out result
    hoc_file_name = "{}.hoc".format(template_name)
    emodel_hoc_path = os.path.join(hoc_dir, hoc_file_name)
    with open(emodel_hoc_path, "w") as emodel_hoc_file:
        emodel_hoc_file.write(hoc)


def get_hoc():
    """Returns hoc file and emodel."""
    template_dir = "templates"
    template = "cell_template_neurodamus.jinja2"

    constants_path = "constants.hoc"
    etype, morph_dir, morph_fname, dt, gid = load_constants(constants_path)
    morph_path = os.path.join(morph_dir, morph_fname)

    recipes_path = os.path.join("config", "recipes", "recipes.json")
    params_filename = find_param_file(recipes_path, etype)
    mechs = load_mechanisms(params_filename)

    params_path = os.path.join("config", "params", "final.json")
    release_params = load_params(params_filename=params_path, etype=etype)
    params = define_parameters(params_filename)

    axon_hoc_path = os.path.join("templates", "replace_axon_hoc.hoc")
    replace_axon_hoc = get_axon_hoc(axon_hoc_path)
    morph = NrnFileMorphologyCustom(
        morph_path,
        do_replace_axon=True,
        replace_axon_hoc=replace_axon_hoc,
        do_set_nseg=40,
    )

    cell = ephys.models.CellModel(
        name=etype, morph=morph, mechs=mechs, params=params, gid=gid
    )

    hoc = cell.create_hoc(release_params, template=template, template_dir=template_dir)

    return hoc, etype


if __name__ == "__main__":
    hoc, emodel = get_hoc()
    write_hoc(hoc, emodel)
