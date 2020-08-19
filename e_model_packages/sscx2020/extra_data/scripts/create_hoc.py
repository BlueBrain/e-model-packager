"""Creates .hoc from cell."""
import os

from load import (
    load_config,
    create_cell,
    load_constants,
)


def write_hoc(hoc, template_name, hoc_dir=""):
    """Write hoc file."""
    # write out result
    hoc_file_name = "{}.hoc".format(template_name)
    emodel_hoc_path = os.path.join(hoc_dir, hoc_file_name)
    with open(emodel_hoc_path, "w") as emodel_hoc_file:
        emodel_hoc_file.write(hoc)


def get_hoc():
    """Returns hoc file and emodel."""
    config = load_config()

    template_dir = config.get("Paths", "templates_dir")
    template = config.get("Paths", "create_hoc_template_file")

    # get emodel
    constants_path = os.path.join(
        config.get("Paths", "constants_dir"), config.get("Paths", "constants_file")
    )
    emodel, _, _, _, _ = load_constants(constants_path)

    cell, release_params, _ = create_cell(config)

    hoc = cell.create_hoc(release_params, template=template, template_dir=template_dir)

    return hoc, emodel, config.get("Paths", "memodel_dir")


if __name__ == "__main__":
    hoc, emodel, hoc_dir = get_hoc()
    write_hoc(hoc, emodel, hoc_dir)
