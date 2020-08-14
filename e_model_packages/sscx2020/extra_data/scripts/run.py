"""Create python recordings."""

import os
import numpy as np

import bluepyopt.ephys as ephys

from load import (
    load_constants,
    load_params,
    load_mechanisms,
    find_param_file,
    define_protocols,
    define_parameters,
    get_axon_hoc,
)
from mymorphology import NrnFileMorphologyCustom

output_path = "python_recordings"
output_file = "soma_voltage_"

# load params & mechanisms
constants_path = "constants.hoc"
etype, morph_dir, morph_fname, dt, gid = load_constants(constants_path)
morph_path = os.path.join(morph_dir, morph_fname)

recipes_path = os.path.join("config", "recipes", "recipes.json")
params_filename = find_param_file(recipes_path, etype)
mechs = load_mechanisms(params_filename)


params_path = os.path.join("config", "params", "final.json")
release_params = load_params(params_filename=params_path, etype=etype)
params = define_parameters(params_filename)


# create morphology
axon_hoc_path = os.path.join("templates", "replace_axon_hoc.hoc")
replace_axon_hoc = get_axon_hoc(axon_hoc_path)
morph = NrnFileMorphologyCustom(
    morph_path, do_replace_axon=True, replace_axon_hoc=replace_axon_hoc, do_set_nseg=40,
)

# create cell
cell = ephys.models.CellModel(
    name=etype, morph=morph, mechs=mechs, params=params, gid=gid
)

# create protocols
amp_filename = "current_amps.dat"
protocols = define_protocols(amp_filename)

# simulator
nrn = ephys.simulators.NrnSimulator(dt=dt)

# run
print("Python Recordings Running...")

responses = protocols.run(cell_model=cell, param_values=release_params, sim=nrn)

for key, resp in responses.items():
    output = os.path.join(output_path, output_file + key + ".dat")

    time = np.array(resp["time"])
    soma_voltage = np.array(resp["voltage"])

    np.savetxt(output, np.transpose(np.vstack((time, soma_voltage))))

print("Python Recordings Done")
