# -*- coding: utf-8 -*-
"""Plot shape of neuron."""
from __future__ import unicode_literals  # for micrometer display
from matplotlib.colors import to_rgb
from matplotlib import cm
from neuron.gui2.utilities import _segment_3d_pts


def auto_aspect(ax):
    """Sets the x, y, and z range symmetric around the center."""
    bounds = [ax.get_xlim(), ax.get_ylim()]
    half_delta_max = max([(item[1] - item[0]) / 2 for item in bounds])
    xmid = sum(bounds[0]) / 2
    ymid = sum(bounds[1]) / 2

    ax.set_xlim((xmid - half_delta_max, xmid + half_delta_max))
    ax.set_ylim((ymid - half_delta_max, ymid + half_delta_max))


def get_morph_lines(
    ax,
    sim,
    val_min=-90,
    val_max=30,
    sections=None,
    variable="v",
    cmap=cm.cool,
    do_plot=False,
    plot_3d=False,
    threshold_volt=4,
    old_vals=None,
    xaxis=2,
    yaxis=0,
    zaxis=1,
    linewidth=0.5,
):
    """Plots a 3D shapeplot.

    Args:
        ax: matplotlib.pyplot axis
        sim: bluepyopt.ephys.simulators.NrnSimulator object
        val_min(int): minimum value of voltage for colormap
        val_max(int): minimum value of voltage for colormap
        sections: list of h.Section() objects to be plotted. If None, all sections are loaded.
        variable(str): variable to be plotted. 'v' for voltage.
        cmap: matplotlib colormap object
        do_plot(bool): True to plot data. False to get actualised data.
        plot_3d (bool): set to True to plot the shape in 3D
        threshold_volt(int): voltage difference from which
            color should be changed on the cell shape.
        old_vals(list): variable values at the last display
        xaxis, yaxis, zaxis(int): 0 for x, 1 for y, 2 for z
        linewidth(float): width of line in shape plot

    Returns:
        lines = list of line objects making up shapeplot
        old_vals = list of voltages previously plotted
    """
    # Adapted from the NEURON package (fct _do_plot in __init__):
    # https://www.neuron.yale.edu/neuron/
    # where this part of the code was itself adapted from
    # https://github.com/ahwillia/PyNeuron-Toolbox/blob/master/PyNeuronToolbox/morphology.py
    # Accessed 2019-04-11, which had an MIT license

    # Default is to plot all sections.
    if sections is None:
        sections = list(sim.neuron.h.allsec())

    sim.neuron.h.define_shape()

    if do_plot:
        lines_list = []
    else:
        # get lines to be actualised
        lines_list = ax.lines

    val_range = val_max - val_min
    lines_to_update = []
    vals = []

    labels = ["x [μm]", "y [μm]", "z [μm]"]

    # get lines and variable values at each segment
    for i, sec in enumerate(sections):
        # Plot each segment as a line
        if do_plot:
            all_seg_pts = _segment_3d_pts(sec)

            for seg, data in zip(sec, all_seg_pts):
                val = getattr(seg, variable)
                vals.append(val)

                if plot_3d:
                    (line,) = ax.plot(
                        data[xaxis],
                        data[yaxis],
                        data[zaxis],
                        "-",
                        linewidth=linewidth,
                        color="black",
                    )
                else:
                    (line,) = ax.plot(
                        data[xaxis],
                        data[yaxis],
                        "-",
                        linewidth=linewidth,
                        color="black",
                    )

                ax.set_xlabel(labels[xaxis])
                ax.set_ylabel(labels[yaxis])
                if plot_3d:
                    ax.set_zlabel(labels[zaxis])
                if cmap:
                    col = cmap(
                        (min(max(val, val_min), val_max) - val_min) / (val_range)
                    )
                    line.set_color(col)
                lines_list.append(line)

        else:
            for seg in sec:
                val = getattr(seg, variable)
                vals.append(val)

    if do_plot:
        auto_aspect(ax)

    if old_vals is None:
        old_vals = [100] * len(vals)

    if val_range and old_vals and cmap:
        for i, (line, val, old_val) in enumerate(zip(lines_list, vals, old_vals)):
            if val is not None and abs(val - old_val) > threshold_volt:
                col = cmap((min(max(val, val_min), val_max) - val_min) / (val_range))
                line.set_color(col)
                lines_to_update.append(line)

                old_vals[i] = val

    return lines_to_update, old_vals
