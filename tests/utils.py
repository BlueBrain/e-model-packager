import csv
import os
import random


def get_random_cell_index(workflow_config):
    """Get random layer, postgid, pregid and source_dir."""
    index_dir = workflow_config.get("paths", "index")
    layers = workflow_config.get("circuit", "layers").split(",")

    layer = random.choice(layers)

    index_file_name = "index_" + layer + ".csv"
    index_file_path = os.path.join(index_dir, index_file_name)
    with open(index_file_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        # turning reader into might cause memory problems if the file is very big
        # but the files used here should not cause any problem
        rows = list(reader)
        row = random.choice(rows)

    return {
        "layers": layer,
        "pregid": int(row["pregid"]),
        "postgid": int(row["postgid"]),
        "source_dir": os.path.dirname(row["path"]),
    }
