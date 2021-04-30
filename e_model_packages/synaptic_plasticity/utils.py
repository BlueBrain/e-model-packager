"""Contains the utility functions needed for the workflow."""
import os


def get_output_path(output_dir, layers, pregid, postgid):
    """Return cell output path given layers, pregid and postgid."""
    gids = str(pregid) + "-" + str(postgid)
    return os.path.join(output_dir, layers, gids)
