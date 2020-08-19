"""Create python recordings."""

import os
import numpy as np

import bluepyopt.ephys as ephys

from load import (
    load_config,
    define_protocols,
    create_cell,
)

config = load_config()

output_dir = config.get("Paths", "output_dir")
output_file = config.get("Paths", "output_file")

cell, release_params, dt_tmp = create_cell(config)

# create protocols
amp_filename = os.path.join(
    config.get("Paths", "protocol_amplitudes_dir"),
    config.get("Paths", "protocol_amplitudes_file"),
)
protocols = define_protocols(amp_filename, config)

# simulator
if config.has_section("Sim") and config.has_option("Sim", "dt"):
    dt = config.getfloat("Sim", "dt")
else:
    dt = dt_tmp
nrn = ephys.simulators.NrnSimulator(dt=dt)

# run
print("Python Recordings Running...")

responses = protocols.run(cell_model=cell, param_values=release_params, sim=nrn)

for key, resp in responses.items():
    output_path = os.path.join(output_dir, output_file + key + ".dat")

    time = np.array(resp["time"])
    soma_voltage = np.array(resp["voltage"])

    np.savetxt(output_path, np.transpose(np.vstack((time, soma_voltage))))

print("Python Recordings Done")
